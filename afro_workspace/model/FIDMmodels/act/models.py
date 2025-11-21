import torch
from torch import nn, Tensor
from omegaconf import DictConfig
from typing import Optional


class ACTEncoder(nn.Module):
    """
    Multi-layer encoder built from ACTEncoderLayer.
    Expects config.dim_model to match the input feature dim.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)]
        )
        # If pre_norm, final norm is applied; otherwise identity to match post-norm behaviour.
        self.final_norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, seq_len, dim_model]
        pos_embed: Optional [B, seq_len, dim] or [seq_len, dim]
        attn_mask: optional boolean or float mask for attention (not used by default)
        """
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, attn_mask=attn_mask)
        x = self.final_norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with a clear pre-norm/post-norm split.
    Uses batch_first=True MultiheadAttention.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.dim_model = cfg.dim_model
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.dim_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)

        # activation function helper from project utils (string -> fn)
        from diffusion_policy_3d.model.FIDMmodels.utils.utils import get_activation_fn
        self.activation = get_activation_fn(cfg.feedforward_activation)

        self.pre_norm = cfg.pre_norm

    def forward(self, x: Tensor, pos_embed: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, seq_len, dim_model]
        pos_embed: optionally [B, seq_len, dim_model] or [seq_len, dim_model]
        """
        # prepare q/k
        if pos_embed is None:
            qk = x
        else:
            # broadcasting: if pos_embed shape [1, seq_len, D], it will broadcast with x
            qk = x + pos_embed

        if self.pre_norm:
            # Pre-norm variant: norm before sublayers
            x_norm = self.norm1(x)
            qk_norm = x_norm + (pos_embed if pos_embed is not None else 0.0)
            # attention
            attn_out = self.self_attn(qk_norm, qk_norm, value=x_norm, attn_mask=attn_mask)[0]
            x = x + self.dropout1(attn_out)

            # feed-forward
            x_norm = self.norm2(x)
            ff = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
            x = x + self.dropout2(ff)
            return x
        else:
            # Post-norm variant
            attn_out = self.self_attn(qk, qk, value=x, attn_mask=attn_mask)[0]
            x = x + self.dropout1(attn_out)
            x = self.norm1(x)

            ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout2(ff)
            x = self.norm2(x)
            return x


# import numpy as np
# import torch
# from omegaconf import DictConfig
# from torch import Tensor, nn
# from diffusion_policy_3d.model.FIDMmodels.utils.utils import get_activation_fn
# from typing import Optional

# class ACTEncoder(nn.Module):
#     """Convenience module for running multiple encoder layers, maybe followed by normalization."""

#     def __init__(self, config: DictConfig):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)]
#         )
#         self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

#     def forward(self, x: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
#         for layer in self.layers:
#             x = layer(x, pos_embed=pos_embed)
#         x = self.norm(x)
#         return x


# class ACTEncoderLayer(nn.Module):
#     def __init__(self, cfg: DictConfig):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=cfg.dim_model,
#             num_heads=cfg.n_heads,
#             dropout=cfg.dropout,
#             batch_first=True,
#         )

#         # Feed forward layers.
#         self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
#         self.dropout = nn.Dropout(cfg.dropout)
#         self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

#         self.norm1 = nn.LayerNorm(cfg.dim_model)
#         self.norm2 = nn.LayerNorm(cfg.dim_model)
#         self.dropout1 = nn.Dropout(cfg.dropout)
#         self.dropout2 = nn.Dropout(cfg.dropout)

#         self.activation = get_activation_fn(cfg.feedforward_activation)
#         self.pre_norm = cfg.pre_norm

#     def forward(self, x: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
#         skip = x
#         if self.pre_norm:
#             x = self.norm1(x)
#         q = k = x if pos_embed is None else x + pos_embed
#         x = self.self_attn(q, k, value=x)[
#             0
#         ]  # select just the output, not the attention weights
#         x = skip + self.dropout1(x)
#         if self.pre_norm:
#             skip = x
#             x = self.norm2(x)
#         else:
#             x = self.norm1(x)
#             skip = x
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         x = skip + self.dropout2(x)
#         if not self.pre_norm:
#             x = self.norm2(x)
#         return x

