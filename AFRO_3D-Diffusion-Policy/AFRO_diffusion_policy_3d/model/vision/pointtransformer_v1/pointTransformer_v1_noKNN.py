import torch
import torch.nn as nn
from termcolor import cprint
from typing import List, Dict, Tuple, Optional


class PurePointTransformerLayer(nn.Module):
    """Pure transformer layer that processes all points directly"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position encoding for 3D coordinates
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, pos):
        """
        Args:
            x: features [B, N, dim]
            pos: positions [B, N, 3]
        Returns:
            output: [B, N, dim]
        """
        B, N, C = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoder(pos)
        x_with_pos = x + pos_enc
        
        # Generate Q, K, V
        qkv = self.qkv(x_with_pos).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class PurePointTransformerBlock(nn.Module):
    """Pure transformer block with residual connections and LayerNorm"""
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PurePointTransformerLayer(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, pos):
        """
        Args:
            x: features [B, N, dim]
            pos: positions [B, N, 3]
        Returns:
            output: [B, N, dim]
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), pos)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class PurePointTransformerExtractor(nn.Module):
    """Pure transformer-based point cloud extractor without any grouping or clustering"""
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.0,
                 final_norm: str = 'layernorm',
                 use_projection: bool = True,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        cprint(f"[PurePointTransformerExtractor] in_channels: {in_channels}", 'cyan')
        cprint(f"[PurePointTransformerExtractor] out_channels: {out_channels}", 'cyan')
        cprint(f"[PurePointTransformerExtractor] embed_dim: {embed_dim}", 'cyan')
        cprint(f"[PurePointTransformerExtractor] num_layers: {num_layers}", 'cyan')
        cprint(f"[PurePointTransformerExtractor] num_heads: {num_heads}", 'cyan')
        
        # Input embedding
        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            PurePointTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output processing
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Final projection
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(embed_dim, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(embed_dim, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
            
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PurePointTransformerExtractor] not use projection", "yellow")
    
    def forward(self, xyz):
        """
        Input:
            xyz: point cloud data, [B, N, C] where C can be 3 (XYZ) or 6 (XYZRGB)
        Return:
            x: global feature, [B, out_channels]
        """
        B, N, C = xyz.shape
        
        if N == 0:
            return torch.zeros((B, self.out_channels), dtype=torch.float32, device=xyz.device)
        
        # Extract positions (always first 3 dimensions)
        pos = xyz[:, :, :3]  # [B, N, 3]
        
        # Input embedding
        x = self.input_proj(xyz)  # [B, N, embed_dim]
        x = self.input_norm(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, pos)  # [B, N, embed_dim]
        
        # Output normalization
        x = self.output_norm(x)  # [B, N, embed_dim]
        
        # Global pooling (max pooling across points)
        x = torch.max(x, dim=1)[0]  # [B, embed_dim]
        
        # Final projection
        x = self.final_projection(x)  # [B, out_channels]
        
        return x
