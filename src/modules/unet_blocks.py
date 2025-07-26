# src/dit.py
import torch
import torch.nn as nn
import math
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from typing import Optional, Union, Tuple


# === Patch Embedding ===
class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=6, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class DiTOutput(BaseOutput):
    sample: torch.FloatTensor


# === Diffusion Transformer (DiT) ===
class DiTModel(ModelMixin):
    def __init__(self, image_size=128, patch_size=16, in_channels=6,
                 hidden_size=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        self.pos_embed = PositionalEncoding(hidden_size, max_len=(image_size // patch_size) ** 2)
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, patch_size * patch_size * 3)  # predict RGB patch
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs
    ) -> Union[DiTOutput, Tuple]:
        B = sample.size(0)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)

        # [B, C, H, W] → [B, N, D]
        x = self.patch_embed(sample)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)  # [B, N, patch^2*3]

        # Convert patches back to image
        H = W = self.image_size // self.patch_size
        x = x.transpose(1, 2).reshape(B, 3, self.patch_size, self.patch_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, 3, H * self.patch_size, W * self.patch_size)

        if not return_dict:
            return (x,)

        return DiTOutput(sample=x)
