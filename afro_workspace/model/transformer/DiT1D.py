import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

# ---------- small util ----------
def modulate(x, shift, scale):
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
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
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

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

# ---------- Vector-version DiT with 3 parameterizations ----------
class DiTFeaturePredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
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

        self.num_tokens = 1
        self.noise_proj = nn.Linear(input_dim, hidden_size)

        self.vis_proj   = nn.Linear(input_dim, hidden_size)
        self.act_proj   = nn.Linear(16, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size)) if use_pos_embed else None

        # timestep + (vis, act) -> c for AdaLN
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=timestep_embed_dim)
        self.ca_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, input_dim)
        self.pred_norm = nn.LayerNorm(input_dim, elementwise_affine=True)

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

    def forward(self, z, t, vis, act):
        n0 = self.noise_proj(z)                 # (B, H)
        x = n0.unsqueeze(1)                     # (B, 1, H)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :1, :]

        t_emb = self.t_embedder(t)              # (B, H)
        v = self.vis_proj(vis)                  # (B, H)
        a = self.act_proj(act)                  # (B, H)
        c = self.ca_proj(torch.cat([t_emb, v, a], dim=-1))  # (B, H)

        for blk in self.blocks:
            x = blk(x, c)

        token_out = x[:, 0, :]                  # (B, H)
        pred = self.output_layer(token_out)     # (B, D)
        return pred



# # import torch
# # import torch.nn as nn
# # import math
# # from timm.models.vision_transformer import Attention, Mlp

# # # ---------- small util ----------
# # def modulate(x, shift, scale):
# #     # x: (B, T, D), shift, scale: (B, D)
# #     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# # # ---------- timestep embedder (same as original) ----------
# # class TimestepEmbedder(nn.Module):
# #     def __init__(self, hidden_size, frequency_embedding_size=256):
# #         super().__init__()
# #         self.mlp = nn.Sequential(
# #             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
# #             nn.SiLU(),
# #             nn.Linear(hidden_size, hidden_size, bias=True),
# #         )
# #         self.frequency_embedding_size = frequency_embedding_size

# #     @staticmethod
# #     def timestep_embedding(t, dim, max_period=10000):
# #         half = dim // 2
# #         freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
# #         args = t[:, None].float() * freqs[None]
# #         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
# #         if dim % 2:
# #             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
# #         return embedding

# #     def forward(self, t):
# #         t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
# #         t_emb = self.mlp(t_freq)
# #         return t_emb  # (B, hidden_size)

# # # ---------- DiT block (adaLN-Zero style) ----------
# # class DiTBlock(nn.Module):
# #     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
# #         super().__init__()
# #         # Pre-norm without affine parameters (adaLN will handle modulation)
# #         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
# #         self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
# #         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
# #         mlp_hidden_dim = int(hidden_size * mlp_ratio)
# #         approx_gelu = lambda: nn.GELU(approximate="tanh")
# #         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

# #         # AdaLN modulation: produces shift/scale/gate for MSA and MLP branches
# #         self.adaLN_modulation = nn.Sequential(
# #             nn.SiLU(),
# #             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
# #         )

# #     def forward(self, x, c):
# #         # c: (B, hidden_size) after cond_proj (here c only contains t_emb info)
# #         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
# #         # MSA branch
# #         x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
# #         # MLP branch
# #         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
# #         return x

# # # ---------- Vector-version DiT: 输入 (z, t, vis, act) -> 输出 (B, d) ----------
# # class DiTFeaturePredictor(nn.Module):
# #     """
# #     DiT adapted for vector (latent) feature prediction.

# #     Inputs:
# #         z   : (B, d_in)  - noisy latent vector (x_t)
# #         t   : (B,)       - diffusion timestep scalar (int/float)
# #         vis : (B, d_in)  - visual feature (condition)
# #         act : (B, d_in)  - action feature (condition)

# #     Output:
# #         if learn_sigma=False: out (B, d_in)
# #         if learn_sigma=True:  out (B, d_in*2) => split to mean, logvar
# #     """
# #     def __init__(
# #         self,
# #         input_dim,              # d (visual/action/noise dim)
# #         hidden_size=256,        # transformer hidden dim
# #         depth=6,
# #         num_heads=8,
# #         mlp_ratio=4.0,
# #         learn_sigma=False,
# #         timestep_embed_dim=256,
# #         use_pos_embed=True,
# #     ):
# #         super().__init__()
# #         assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
# #         self.input_dim = input_dim
# #         self.hidden_size = hidden_size

# #         # Now use 3 tokens: [noise_token, vis_token, act_token]
# #         self.num_tokens = 3
# #         self.learn_sigma = learn_sigma
# #         self.out_dim = input_dim * (2 if learn_sigma else 1)

# #         # token projectors (separate so vis/act can be treated differently)
# #         self.vis_proj = nn.Linear(input_dim, hidden_size)
# #         self.act_proj = nn.Linear(input_dim, hidden_size)
# #         self.noise_proj = nn.Linear(input_dim, hidden_size)

# #         # positional / token-type embedding for tokens (optional)
# #         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size)) if use_pos_embed else None

# #         # timestep embedder (c)
# #         self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=timestep_embed_dim)

# #         # condition projector: now we only use timestep embedding as cond vector
# #         # concat dim = hidden_size (only t_emb)
# #         self.cond_proj = nn.Sequential(
# #             nn.Linear(hidden_size, hidden_size, bias=True),
# #             nn.SiLU(),
# #             nn.Linear(hidden_size, hidden_size, bias=True)
# #         )

# #         # transformer blocks
# #         self.blocks = nn.ModuleList([
# #             DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
# #         ])

# #         # output head (from noise token to predicted feature)
# #         self.output_layer = nn.Linear(hidden_size, self.out_dim)

# #         # initialization
# #         self.initialize_weights()

# #     def initialize_weights(self):
# #         # basic linear init
# #         for m in self.modules():
# #             if isinstance(m, nn.Linear):
# #                 nn.init.xavier_uniform_(m.weight)
# #                 if m.bias is not None:
# #                     nn.init.constant_(m.bias, 0.0)

# #         # zero-out adaLN modulation linear last layer weights/biases (like DiT)
# #         for blk in self.blocks:
# #             final_linear = blk.adaLN_modulation[-1]
# #             if isinstance(final_linear, nn.Linear):
# #                 nn.init.constant_(final_linear.weight, 0.0)
# #                 if final_linear.bias is not None:
# #                     nn.init.constant_(final_linear.bias, 0.0)

# #         # zero-out output layer bias a bit (optional)
# #         nn.init.constant_(self.output_layer.bias, 0.0)

# #         # pos_embed small init
# #         if self.pos_embed is not None:
# #             nn.init.trunc_normal_(self.pos_embed, std=0.02)

# #     def forward(self, z, t, vis, act):
# #         """
# #         z:   (B, d_in)    noisy latent
# #         t:   (B,)         timestep scalars
# #         vis: (B, d_in)    visual condition
# #         act: (B, d_in)    action condition

# #         returns:
# #             (B, d_in)  or (B, d_in*2) if learn_sigma
# #         """
# #         B = z.shape[0]
# #         assert vis.shape == (B, self.input_dim) and act.shape == (B, self.input_dim) and z.shape == (B, self.input_dim)

# #         # --- project tokens (separate projections) ---
# #         v0 = self.vis_proj(vis)    # (B, hidden_size)  <- condition token
# #         a0 = self.act_proj(act)    # (B, hidden_size)  <- condition token
# #         n0 = self.noise_proj(z)    # (B, hidden_size)  <- main noisy token

# #         # tokens for transformer: concatenate [noise, vis, act]
# #         n = n0.unsqueeze(1)        # (B,1,hidden)
# #         v = v0.unsqueeze(1)        # (B,1,hidden)
# #         a = a0.unsqueeze(1)        # (B,1,hidden)
# #         x = torch.cat([n, v, a], dim=1)   # (B,3,hidden)

# #         if self.pos_embed is not None:
# #             x = x + self.pos_embed  # broadcast (1,3,D)

# #         # --- timestep embedding --- (used only for adaLN modulation)
# #         t_emb = self.t_embedder(t)             # (B, hidden_size)
# #         c = self.cond_proj(t_emb)              # (B, hidden_size)

# #         # --- transformer blocks: AdaLN modulation uses c ---
# #         for blk in self.blocks:
# #             x = blk(x, c)  # x shape stays (B,3,hidden_size)

# #         # readout the noisy token (index 0 since it's the first token)
# #         token_out = x[:, 0, :]                  # (B, hidden_size)
# #         out = self.output_layer(token_out)      # (B, out_dim)
# #         return out


# import torch
# import torch.nn as nn
# import math
# from timm.models.vision_transformer import Attention, Mlp

# # ---------- small util ----------
# def modulate(x, shift, scale):
#     # x: (B, T, D), shift, scale: (B, D)
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# # ---------- timestep embedder ----------
# class TimestepEmbedder(nn.Module):
#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True),
#         )
#         self.frequency_embedding_size = frequency_embedding_size

#     @staticmethod
#     def timestep_embedding(t, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
#         args = t[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding

#     def forward(self, t):
#         t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
#         t_emb = self.mlp(t_freq)
#         return t_emb  # (B, hidden_size)

# # ---------- DiT block (adaLN-Zero style) ----------
# class DiTBlock(nn.Module):
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

#         # AdaLN modulation: produces shift/scale/gate for MSA and MLP branches
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         # c: (B, hidden_size) for adaLN
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
#         # MSA
#         x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         # MLP
#         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         return x

# # ---------- Vector-version DiT with 3 parameterizations ----------
# class DiTFeaturePredictor(nn.Module):
#     """
#     Inputs:
#         z   : (B, D)  noisy latent x_t (注意：这里我们在 projector 空间里喂 x_t^proj)
#         t   : (B,)    timestep
#         vis : (B, D)  condition (projector 空间)
#         act : (B, D)  condition (projector 空间)

#     Output:
#         pred : (B, D) in the space defined by param_type ('x0' | 'eps' | 'v')
#                我们在本任务中设置 param_type = 'v'，输出 v 速度。
#     """
#     def __init__(
#         self,
#         input_dim,
#         hidden_size=256,
#         depth=6,
#         num_heads=8,
#         mlp_ratio=4.0,
#         timestep_embed_dim=256,
#         use_pos_embed=True,
#         param_type: str = "v",    # 默认速度预测
#     ):
#         super().__init__()
#         assert hidden_size % num_heads == 0
#         assert param_type in ("x0", "eps", "v")
#         self.input_dim = input_dim
#         self.hidden_size = hidden_size
#         self.param_type = param_type

#         # tokens: [noisy(x_t), vis, act]
#         self.num_tokens = 3
#         self.noise_proj = nn.Linear(input_dim, hidden_size)
#         self.vis_proj   = nn.Linear(input_dim, hidden_size)
#         self.act_proj   = nn.Linear(input_dim, hidden_size)

#         self.noise_norm = nn.LayerNorm(hidden_size)
#         self.vis_norm   = nn.LayerNorm(hidden_size)
#         self.act_norm   = nn.LayerNorm(hidden_size)

#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size)) if use_pos_embed else None

#         # timestep + (vis,act) summary for adaLN
#         self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=timestep_embed_dim)
#         self.cond_proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True)
#         )
#         self.ca_proj = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True)
#         )

#         self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
#         self.output_layer = nn.Linear(hidden_size, input_dim)

#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#         for blk in self.blocks:
#             last = blk.adaLN_modulation[-1]
#             if isinstance(last, nn.Linear):
#                 nn.init.constant_(last.weight, 0.0)
#                 if last.bias is not None:
#                     nn.init.constant_(last.bias, 0.0)
#         if self.pos_embed is not None:
#             nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.constant_(self.output_layer.bias, 0.0)

#     # ---------- 参数化工具 ----------
#     @staticmethod
#     def alpha_sigma_from_alphas_cumprod(t: torch.Tensor, alphas_cumprod: torch.Tensor):
#         alpha_bar = alphas_cumprod[t]                   # (B,)
#         alpha_t = torch.sqrt(alpha_bar).unsqueeze(-1)   # (B,1)
#         sigma_t = torch.sqrt(1.0 - alpha_bar).unsqueeze(-1)
#         return alpha_t, sigma_t

#     @staticmethod
#     def target_from_param(param_type: str, x0, eps, alpha_t, sigma_t, x_t):
#         if param_type == "x0":
#             return x0
#         elif param_type == "eps":
#             return eps
#         elif param_type == "v":
#             return alpha_t * eps - sigma_t * x0
#         else:
#             raise ValueError(f"Unknown param_type {param_type}")

#     @staticmethod
#     def pred_to_x0(param_type: str, pred, alpha_t, sigma_t, x_t):
#         if param_type == "x0":
#             return pred
#         elif param_type == "eps":
#             return (x_t - sigma_t * pred) / alpha_t
#         elif param_type == "v":
#             return alpha_t * x_t - sigma_t * pred
#         else:
#             raise ValueError(f"Unknown param_type {param_type}")

#     # ---------- 前向 ----------
#     def forward(self, z, t, vis, act):
#         """
#         所有输入都在同一向量空间（这里就是 projector 输出的维度）。
#         z=x_t^proj, vis=online_projector(start_feat), act=mem_start(同维).
#         """
#         B = z.shape[0]
#         # assert z.shape[1] == self.input_dim and vis.shape == (B, self.input_dim) and act.shape == (B, self.input_dim)

#         n0 = self.noise_norm(self.noise_proj(z))
#         v0 = self.vis_norm(self.vis_proj(vis))
#         a0 = self.act_norm(self.act_proj(act))

#         x = torch.stack([n0, v0, a0], dim=1)           # (B,3,H)
#         if self.pos_embed is not None:
#             x = x + self.pos_embed

#         t_emb = self.t_embedder(t)                     # (B,H)
#         c = self.cond_proj(t_emb) + self.ca_proj(torch.cat([v0, a0], dim=-1))  # (B,H)

#         for blk in self.blocks:
#             x = blk(x, c)

#         token_out = x[:, 0, :]                         # (B,H)
#         pred = self.output_layer(token_out)            # (B,D), in param space
#         return pred

# # ----------------- small unit test / usage example -----------------
# if __name__ == "__main__":
#     B = 4
#     d = 64
#     hidden = 128
#     depth = 4
#     num_heads = 8

#     model = DiTFeaturePredictor(input_dim=d, hidden_size=hidden, depth=depth, num_heads=num_heads, learn_sigma=False)
#     z = torch.randn(B, d)
#     t = torch.randint(0, 1000, (B,))
#     vis = torch.randn(B, d)
#     act = torch.randn(B, d)

#     out = model(z, t, vis, act)
#     print("out.shape:", out.shape)   # expect (B, d)

#     # If learn_sigma=True:
#     model2 = DiTFeaturePredictor(input_dim=d, hidden_size=hidden, depth=depth, num_heads=num_heads, learn_sigma=True)
#     out2 = model2(z, t, vis, act)
#     print("out2.shape (mean+logvar):", out2.shape)  # expect (B, d*2)

