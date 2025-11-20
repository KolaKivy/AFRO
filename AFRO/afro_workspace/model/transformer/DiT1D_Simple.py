import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

# ---------- small util ----------
def modulate(x, shift, scale):
    # x: (B, T, D), shift, scale: (B, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ---------- timestep embedder ----------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # (B, hidden_size)

# ---------- DiT block (adaLN-Zero style) ----------
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        # AdaLN modulation: produces shift/scale/gate for MSA and MLP branches
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c: (B, hidden_size) for adaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # MSA
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

# ---------- Vector-version DiT with 3 parameterizations ----------
class DiTFeaturePredictor_simple(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=2.0,
        timestep_embed_dim=256,
        use_pos_embed=True,
        param_type: str = "v",    
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert param_type in ("x0", "eps", "v")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.param_type = param_type

        # tokens: [noisy(x_t), vis, act]
        self.num_tokens = 3
        self.noise_proj = nn.Linear(input_dim, hidden_size)
        self.vis_proj   = nn.Linear(64, hidden_size)
        self.act_proj   = nn.Linear(32, hidden_size)

        self.noise_norm = nn.LayerNorm(hidden_size)
        self.vis_norm   = nn.LayerNorm(hidden_size)
        self.act_norm   = nn.LayerNorm(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size)) if use_pos_embed else None

        # timestep + (vis,act) summary for adaLN
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=timestep_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.ca_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for blk in self.blocks:
            last = blk.adaLN_modulation[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.weight, 0.0)
                if last.bias is not None:
                    nn.init.constant_(last.bias, 0.0)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.output_layer.bias, 0.0)

    @staticmethod
    def alpha_sigma_from_alphas_cumprod(t: torch.Tensor, alphas_cumprod: torch.Tensor):
        alpha_bar = alphas_cumprod[t]                   # (B,)
        alpha_t = torch.sqrt(alpha_bar).unsqueeze(-1)   # (B,1)
        sigma_t = torch.sqrt(1.0 - alpha_bar).unsqueeze(-1)
        return alpha_t, sigma_t

    @staticmethod
    def target_from_param(param_type: str, x0, eps, alpha_t, sigma_t, x_t):
        if param_type == "x0":
            return x0
        elif param_type == "eps":
            return eps
        elif param_type == "v":
            return alpha_t * eps - sigma_t * x0
        else:
            raise ValueError(f"Unknown param_type {param_type}")

    @staticmethod
    def pred_to_x0(param_type: str, pred, alpha_t, sigma_t, x_t):
        if param_type == "x0":
            return pred
        elif param_type == "eps":
            return (x_t - sigma_t * pred) / alpha_t
        elif param_type == "v":
            return alpha_t * x_t - sigma_t * pred
        else:
            raise ValueError(f"Unknown param_type {param_type}")

    # ---------- 前向 ----------
    def forward(self, z, t, vis, act):
        B = z.shape[0]
        # assert z.shape[1] == self.input_dim and vis.shape == (B, self.input_dim) and act.shape == (B, self.input_dim)

        n0 = self.noise_norm(self.noise_proj(z))
        v0 = self.vis_norm(self.vis_proj(vis))
        a0 = self.act_norm(self.act_proj(act))

        x = torch.stack([n0, v0, a0], dim=1)           # (B,3,H)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        t_emb = self.t_embedder(t)                     # (B,H)
        c = self.cond_proj(t_emb) + self.ca_proj(torch.cat([v0, a0], dim=-1))  # (B,H)

        for blk in self.blocks:
            x = blk(x, c)

        token_out = x[:, 0, :]                         # (B,H)
        pred = self.output_layer(token_out)            # (B,D), in param space
        return pred



