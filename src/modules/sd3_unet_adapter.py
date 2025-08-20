# sd3_unet_adapter.py
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (
    DownBlock2D,
    UNetMidMCABlock2D,
    UpBlock2D,
    get_down_block,
    get_up_block,
)

logger = logging.get_logger(__name__)


# -----------------------------
# Helper: simple MLP adapters
# -----------------------------
class AdapterMLP(nn.Module):
    """
    Project arbitrary features (content/style) into a common token dim
    to be consumed by cross-attention (SD3-style).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, act="silu", dropout=0.0):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        act_layer = nn.SiLU() if act == "silu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, *, in_dim]  ->  [B, *, out_dim]
        return self.net(x)


# -----------------------------
# Output typing
# -----------------------------
@dataclass
class UNetOutput(BaseOutput):
    sample: torch.FloatTensor


# ===========================================================
#           SD3-Like UNet with Content/Style Adapters
# ===========================================================
class SD3AdapterUNet(ModelMixin, ConfigMixin):
    """
    A latent-space UNet (in_channels=4) with cross-attention that can consume:
      - content tokens (multi-scale content features projected)
      - style tokens (style embedding projected)
    You can either pass `encoder_hidden_states` directly (already-built tokens),
    or pass `content_feats` (list of multi-scale feature maps) + `style_embed`
    and let this module build tokens internally.

    Intended as a drop-in replacement for your original UNet to compare with an SD3-style backbone.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        # latent I/O like SD/SD3
        in_channels: int = 4,
        out_channels: int = 4,
        # time
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        # UNet depth
        down_block_types: Tuple[str, ...] = ("DownResBlock2D", "DownResBlock2D", "DownResBlock2D", "DownResBlock2D"),
        up_block_types: Tuple[str, ...] = ("UpResBlock2D", "UpResBlock2D", "UpResBlock2D", "UpResBlock2D"),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 1,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1.0,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        # cross-attn token dim (like SD3 text-encoder width)
        cross_attention_dim: int = 1024,
        attention_head_dim: int = 8,
        channel_attn: bool = False,
        # content multi-scale control (how many scales are fed)
        content_encoder_downsample_size: int = 4,
        content_start_channel: int = 16,
        # adapters
        content_token_dim: int = 512,
        style_token_dim: int = 512,
        token_proj_dim: int = 1024,  # must match cross_attention_dim
        token_dropout: float = 0.0,
        # reduction for MCA
        reduction: int = 32,
        # training policy
        freeze_backbone: bool = True,
    ):
        super().__init__()

        assert token_proj_dim == cross_attention_dim, \
            f"token_proj_dim({token_proj_dim}) must equal cross_attention_dim({cross_attention_dim})"

        self.sample_size = sample_size
        self.content_encoder_downsample_size = content_encoder_downsample_size
        self.freeze_backbone = freeze_backbone

        time_embed_dim = block_out_channels[0] * 4

        # ---------- input conv ----------
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # ---------- time embedding ----------
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # ---------- adapters for content/style ----------
        # Content arrives as multi-scale feature maps [B, C_i, H_i, W_i]
        # We global-pool each map -> [B, C_i] then project to tokens
        self.content_pool = nn.AdaptiveAvgPool2d(1)
        # Create per-scale linear adapters at runtime based on expected channels
        self._content_in_dims = []
        for i in range(content_encoder_downsample_size):
            ch = content_start_channel * (2 ** i)
            self._content_in_dims.append(ch)
        self.content_adapters = nn.ModuleList([
            AdapterMLP(in_dim=ch, out_dim=content_token_dim, hidden=None, act=act_fn, dropout=token_dropout)
            for ch in self._content_in_dims
        ])
        self.content_proj = AdapterMLP(content_token_dim, token_proj_dim, hidden=None, act=act_fn, dropout=token_dropout)

        # Style arrives as a vector [B, S]; project to a token
        self.style_adapter = AdapterMLP(in_dim=style_token_dim, out_dim=token_proj_dim, hidden=None, act=act_fn, dropout=token_dropout)

        # ---------- UNet blocks ----------
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None

        # Down path
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,         # <- enable cross-attn
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                content_channel=0,                               # latent backbone: no direct content channels in down path
                reduction=reduction,
                channel_attn=channel_attn,
            )
            self.down_blocks.append(down_block)

        # Mid block with MCA (kept to be compatible with your pipeline)
        self.mid_block = UNetMidMCABlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            channel_attn=channel_attn,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            content_channel=content_start_channel * (2 ** (content_encoder_downsample_size - 1)),
            reduction=reduction,
        )

        # Up path
        self.num_upsamplers = 0
        rev_channels = list(reversed(block_out_channels))
        output_channel = rev_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = rev_channels[i]
            input_channel = rev_channels[min(i + 1, len(block_out_channels) - 1)]

            add_upsample = not is_final
            if add_upsample:
                self.num_upsamplers += 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,        # <- enable cross-attn
                attn_num_head_channels=attention_head_dim,
                upblock_index=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Output head
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # Optionally freeze backbone (fine-tune adapters only)
        if self.freeze_backbone:
            self._freeze_backbone_parameters()

    # -----------------------------
    # Utils
    # -----------------------------
    def _freeze_backbone_parameters(self):
        # Freeze everything except adapters + output head norm/conv (you can adjust)
        keep = {self.content_adapters, self.content_proj, self.style_adapter}
        keep_ids = set()
        for m in keep:
            for p in m.parameters():
                keep_ids.add(id(p))

        for name, p in self.named_parameters():
            if id(p) not in keep_ids:
                p.requires_grad = False

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
            raise ValueError(
                f"slice_size {slice_size} must divide attention_head_dim {self.config.attention_head_dim}"
            )
        if slice_size is not None and slice_size > self.config.attention_head_dim:
            raise ValueError(
                f"slice_size {slice_size} must be <= attention_head_dim {self.config.attention_head_dim}"
            )

        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)
        if self.mid_block is not None:
            self.mid_block.set_attention_slice(slice_size)
        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DownBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    # -----------------------------
    # Token builder
    # -----------------------------
    def _build_tokens_from_content_style(
        self,
        content_feats: List[torch.Tensor],   # List[B, C_i, H_i, W_i], length = content_encoder_downsample_size
        style_embed: torch.Tensor,           # [B, style_token_dim]
    ) -> torch.Tensor:
        """
        Convert content multi-scale features + style embedding into a single
        token sequence for cross-attention: [B, T, cross_attention_dim]
        """
        B = style_embed.shape[0]

        # Content: pool and per-scale MLP
        content_tokens = []
        expected_scales = min(len(content_feats), len(self.content_adapters))
        for i in range(expected_scales):
            f = content_feats[i]                      # [B, C_i, H_i, W_i]
            pooled = self.content_pool(f).flatten(1)  # [B, C_i]
            c_tok = self.content_adapters[i](pooled)  # [B, content_token_dim]
            c_tok = self.content_proj(c_tok)          # [B, cross_attention_dim]
            content_tokens.append(c_tok.unsqueeze(1)) # [B, 1, D]

        if len(content_tokens) == 0:
            raise ValueError("content_feats is empty or adapters not configured.")

        content_tokens = torch.cat(content_tokens, dim=1)  # [B, T_c, D]

        # Style: single token
        style_token = self.style_adapter(style_embed).unsqueeze(1)  # [B, 1, D]

        # Concatenate [style | content]
        tokens = torch.cat([style_token, content_tokens], dim=1)    # [B, 1+T_c, D]
        return tokens

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(
        self,
        sample: torch.FloatTensor,                 # latent noisy sample [B, 4, H/8, W/8] typically
        timestep: Union[torch.Tensor, float, int],
        # Option A: pass pre-built tokens directly (like SD/SD3 text-embeds)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # [B, T, D]
        # Option B: pass raw features to build tokens inside
        content_feats: Optional[List[torch.Tensor]] = None,
        style_embed: Optional[torch.Tensor] = None,            # [B, style_token_dim]
        # Mid-block MCA expects an index for content scale (keep same API)
        content_encoder_downsample_size: int = 4,
        return_dict: bool = False,
    ) -> Union[UNetOutput, Tuple]:
        """
        Either provide `encoder_hidden_states` directly,
        or provide (`content_feats`, `style_embed`) to build tokens.
        """
        # Prepare tokens
        if encoder_hidden_states is None:
            if (content_feats is None) or (style_embed is None):
                raise ValueError("Provide either `encoder_hidden_states` or both `content_feats` and `style_embed`.")
            tokens = self._build_tokens_from_content_style(content_feats, style_embed)
        else:
            tokens = encoder_hidden_states  # already [B, T, D]

        # 1) time embedding
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        else:
            timesteps = timestep
            if len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2) pre
        sample = self.conv_in(sample)

        # 3) down path
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = any(s % default_overall_up_factor != 0 for s in sample.shape[-2:])
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if hasattr(down_block, "attentions") and down_block.attentions is not None:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=tokens,  # <- use tokens in cross-attn
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4) mid block (MCA retained; tokens for cross-attn)
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                index=content_encoder_downsample_size,
                encoder_hidden_states=tokens,
            )

        # 5) up path
        offset_out_sum = 0
        for i, up_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            upsample_size = None
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(up_block, "attentions") and up_block.attentions is not None:
                sample, offset_out = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=tokens,              # <- use tokens in cross-attn
                    style_structure_features=None,             # kept for API symmetry; not used here
                )
                offset_out_sum += offset_out
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6) post
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample, offset_out_sum)

        return UNetOutput(sample=sample)
