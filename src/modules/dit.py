# src/dit.py
import torch
import torch.nn as nn
import math
from .embeddings import TimestepEmbedding, Timesteps


# === AdaLN (Adaptive Layer Normalization) ===
# This class needs to be defined in dit.py
class AdaLN(nn.Module):
    def __init__(self, n_embd, n_cond):
        super().__init__()
        # Projects conditional input to scale and shift parameters for normalization
        self.linear = nn.Linear(n_cond, 2 * n_embd)

    def forward(self, x, cond):
        # x: [B, N, D_embd] (input tokens)
        # cond: [B, D_cond] (conditional embedding, e.g., timestep embedding)
        scale_shift = self.linear(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        # Reshape scale and shift to [B, 1, D_embd] to broadcast correctly
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return x * (1 + scale) + shift # Apply adaptive normalization


# === Patch Embedding ===
class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=6, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1,
                 time_cond_dim=None, # Dimension of the timestep embedding
                 cross_attn_cond_dim=None): # Dimension of the cross-attention conditional input
        super().__init__()
        # Use elementwise_affine=False as AdaLN will provide scale and shift
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_ln1 = AdaLN(dim, time_cond_dim) if time_cond_dim is not None else None

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-Attention block (optional, for conditional inputs like style/content features)
        self.cross_attn_norm = nn.LayerNorm(dim, elementwise_affine=False) if cross_attn_cond_dim is not None else None
        self.ada_ln_cross_attn = AdaLN(dim, time_cond_dim) if time_cond_dim is not None and cross_attn_cond_dim is not None else None
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True) if cross_attn_cond_dim is not None else None


        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_ln2 = AdaLN(dim, time_cond_dim) if time_cond_dim is not None else None

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, time_emb=None, cross_attn_cond=None):
        # Self-attention block
        h = self.norm1(x)
        if self.ada_ln1 is not None:
            h = self.ada_ln1(h, time_emb)
        x = x + self.attn(h, h, h)[0]

        # Cross-attention block (if enabled)
        if self.cross_attn is not None and cross_attn_cond is not None:
            h_cross = self.cross_attn_norm(x)
            if self.ada_ln_cross_attn is not None:
                h_cross = self.ada_ln_cross_attn(h_cross, time_emb)
            x = x + self.cross_attn(h_cross, cross_attn_cond, cross_attn_cond)[0]


        # MLP block
        h = self.norm2(x)
        if self.ada_ln2 is not None:
            h = self.ada_ln2(h, time_emb)
        x = x + self.mlp(h)
        return x


# === Diffusion Transformer (DiT) ===
class DiTModel(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=6,
                 hidden_size=768, depth=12, num_heads=12,
                 time_embed_dim=256, # Dimension for timestep embedding
                 cross_attn_cond_dim=None): # Dimension for cross-attention conditional input
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        self.pos_embed = PositionalEncoding(hidden_size, max_len=(image_size // patch_size) ** 2)

        # Timestep embedding
        self.time_proj = Timesteps(num_channels=hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(hidden_size, time_embed_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_size, num_heads, time_cond_dim=time_embed_dim,
                             cross_attn_cond_dim=cross_attn_cond_dim)
            for _ in range(depth)
        ])
        # Final LayerNorm before the head, also needs AdaLN
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ada_ln_final = AdaLN(hidden_size, time_embed_dim)

        # Ensure the head output matches what your pipeline expects (e.g., 3 channels for RGB, or latent channels)
        # If your pipeline expects noise in latent space, adjust the '3' below.
        self.head = nn.Linear(hidden_size, patch_size * patch_size * 3) # Example: predict RGB patch or latent patch

        self.patch_size = patch_size
        self.image_size = image_size
        self.out_channels = 3 # Make this configurable if needed for latent space output

    def forward(self, x, t=None, cond=None):
        # Process timestep embedding
        t_emb = self.time_proj(t)
        t_emb = self.time_embed(t_emb) # [B, time_embed_dim]

        # Process conditional input for cross-attention (e.g., style_hidden_states)
        # 'cond' here is expected to be [B, seq_len, dim] if used with Cross-Attention
        cross_attn_cond = cond # Assuming cond is already in the right format for cross-attention

        x = self.patch_embed(x)     # [B, N, D]
        x = self.pos_embed(x)       # [B, N, D]

        # Pass both timestep embedding and cross_attn_cond to each block
        for block in self.blocks:
            x = block(x, time_emb, cross_attn_cond)

        x = self.ada_ln_final(self.norm(x), t_emb) # Apply AdaLN to final layer norm
        x = self.head(x)            # [B, N, 3*patch^2]

        # Reshape output patches back to image
        B, N, P_squared_C = x.shape
        H_patches = W_patches = self.image_size // self.patch_size
        
        # Determine the number of output channels based on the head's output dimension
        # Assuming P_squared_C = patch_size * patch_size * out_channels
        out_channels_per_patch = P_squared_C // (self.patch_size * self.patch_size)

        x = x.transpose(1, 2).reshape(B, out_channels_per_patch, self.patch_size, self.patch_size, H_patches, W_patches)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, out_channels_per_patch, H_patches * self.patch_size, W_patches * self.patch_size)
        return x