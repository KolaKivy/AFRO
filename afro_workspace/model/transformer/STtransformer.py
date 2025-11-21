import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, num_layers=2, act=nn.GELU):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers-1):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    
class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
    
    def forward(self, x: Tensor, is_causal: bool = False, attn_mask: Tensor = None) -> Tensor:
        if is_causal:
            seq_len = x.shape[1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            x = self.attn(x, x, x, attn_mask=causal_mask)[0]
        else:
            x = self.attn(x, x, x, attn_mask=attn_mask)[0]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[1]].to(x.device)

class TemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(TemporalBlock, self).__init__()
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        # Temporal attention
        x_ = self.norm1(x)
        x_ = self.temporal_attn(x_, attn_mask=attn_mask)
        x = x + x_

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x

class TemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super(TemporalTransformer, self).__init__() 
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化到单一时间步

    def forward(self, x: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        x = self.ffn(x)  # [B, T, model_dim]
        x = self.pos_enc(x)  # [B, T, model_dim]

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=-1)
            x = nn.Linear(x.shape[-1], x.shape[-1])(x)

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)  # [B, T, model_dim]

        x = rearrange(x, "b t e -> b e t")  # [B, model_dim, T]
        x = self.pool(x).squeeze(-1)  # [B, model_dim]
        x = self.out(x)  # [B, out_dim]
        return x

class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        # Spatial attention
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_, attn_mask=attn_mask)
        x = x + x_

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x

class SpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化到单一空间节点

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        x = self.ffn(x)  # [B, S, model_dim]
        x = self.pos_enc(x)  # [B, S, model_dim]

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)  # [B, S, model_dim]

        # 池化空间维度以匹配 fdm_vis_decoder 输出
        x = rearrange(x, "b s e -> b e s")  # [B, model_dim, S]
        x = self.pool(x).squeeze(-1)  # [B, model_dim]
        x = self.out(x)  # [B, out_dim]
        return x

