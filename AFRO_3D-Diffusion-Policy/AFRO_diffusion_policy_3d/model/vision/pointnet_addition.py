import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
######################new###########################
# -------------------------
# pointdiff
# -------------------------

class VarianceSchedule(nn.Module):
    
    def __init__(self, generator_config):
        super().__init__()
        
        self.num_steps = generator_config.time_schedule.num_steps
        self.beta_start = generator_config.time_schedule.beta_start
        self.beta_end = generator_config.time_schedule.beta_end
        self.mode = generator_config.time_schedule.mode
        
        if self.mode == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, steps=self.num_steps)
            
        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        alphas = 1 - betas
        
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def recurrent_uniform_sampling(self, batch_size, interval_nums):
        interval_size = self.num_steps / interval_nums
        sampled_intervals = []
        for i in range(interval_nums):
            start = int(i * interval_size) + 1
            end = int((i + 1) * interval_size)
            sampled_interval = np.random.choice(np.arange(start, end + 1), batch_size)
            sampled_intervals.append(sampled_interval)
        ts = np.vstack(sampled_intervals)
        ts = torch.tensor(ts)
        ts = torch.stack([ts[:, i][torch.randperm(interval_nums)] for i in range(batch_size)], dim=1)
        return ts

# Point Condition Network 
class PCNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super(PCNet, self).__init__()
        self.fea_layer = nn.Linear(dim_in, dim_out)
        self.cond_bias = nn.Linear(dim_cond, dim_out, bias=False)
        self.cond_gate = nn.Linear(dim_cond, dim_out)

    def forward(self, fea, cond):
        gate = torch.sigmoid(self.cond_gate(cond))
        bias = self.cond_bias(cond)
        out = self.fea_layer(fea) * gate + bias
        return out

# Point Denoising Network
class DenoisingNet(nn.Module):
    
    def __init__(self, point_dim, cond_dims, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            PCNet(3, 128, cond_dims+3),
            PCNet(128, 256, cond_dims+3),
            PCNet(256, 512, cond_dims+3),
            PCNet(512, 256, cond_dims+3),
            PCNet(256, 128, cond_dims+3),
            PCNet(128, 3, cond_dims+3)
        ])

    def forward(self, coords, beta, cond):
        """
        Args:
            coords:   Noise point clouds at timestep t, (B, N, 3).
            beta:     Time. (B, ).
            cond:     Condition. (B, F).
        """

        batch_size = coords.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        cond = cond.view(batch_size, 1, -1)         # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        cond_emb = torch.cat([time_emb, cond], dim=-1)    # (B, 1, F+3)
        
        out = coords
        for i, layer in enumerate(self.layers):
            out = layer(fea=out, cond=cond_emb)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return coords + out
        else:
            return out

# from SoftPool import soft_pool2d, SoftPool2d
# class CANet(nn.Module): 
#     def __init__(self, encoder_dims, cond_dims):
#         super().__init__()
#         self.encoder_dims = encoder_dims
#         self.cond_dims = cond_dims

#         self.mlp1 = nn.Sequential(
#             nn.Linear(self.encoder_dims, 512, bias=True),
#             nn.ReLU(True),
#             nn.Linear(512, 512, bias=True),
#             nn.ReLU(True),
#         )

#         self.linear1 = nn.Linear(512, 512)

#         self.mlp2 = nn.Sequential(
#             nn.Linear(1024, 512, bias=True),
#             nn.ReLU(True),
#             nn.Linear(512, self.cond_dims, bias=True),
#             nn.ReLU(True),
#         )

#         self.linear2 = nn.Linear(self.cond_dims, self.cond_dims)

#     def forward(self, patch_fea):
#         '''
#             patch_feature : B G 384
#             -----------------
#             point_condition : B 384
#         '''
        
#         patch_fea = self.mlp1(patch_fea)          # B 512
#         # soft_pool2d
#         global_fea = self.linear1(patch_fea)  # B 512
#         combined_fea = torch.cat([patch_fea, global_fea], dim=-1)                 # B 1024
#         combined_fea = self.mlp2(combined_fea)                                       # B F
#         condition_fea = self.linear2(combined_fea)  # B F
#         return condition_fea
from SoftPool import soft_pool2d
class CANet(nn.Module): 
    def __init__(self, encoder_dims, cond_dims):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.cond_dims = cond_dims

        self.mlp1 = nn.Sequential(
            nn.Conv2d(self.encoder_dims, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, self.cond_dims, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, patch_fea):
        '''
            patch_feature : B G 384
            -----------------
            point_condition : B 384
        '''
        
        patch_fea = patch_fea.transpose(1, 2)     # B 384 G
        patch_fea = patch_fea.unsqueeze(-1)       # B 384 G 1
        patch_fea = self.mlp1(patch_fea)          # B 512 G 1
        # soft_pool2d
        global_fea = soft_pool2d(patch_fea, kernel_size=[patch_fea.size(2), 1])  # B 512 1 1
        global_fea = global_fea.expand(-1, -1, patch_fea.size(2), -1)            # B 512 G 1
        combined_fea = torch.cat([patch_fea, global_fea], dim=1)                 # B 1024 G 1
        combined_fea = self.mlp2(combined_fea)                                       # B F G 1
        condition_fea = soft_pool2d(combined_fea, kernel_size=[combined_fea.size(2), 1])  # B F 1 1
        condition_fea = condition_fea.squeeze(-1).squeeze(-1)                          #  B F
        return condition_fea


class CPDM(nn.Module):
    def __init__(self, generator_config, **kwargs):
        super().__init__()
        # self.config = config
        self.cond_dims = generator_config.cond_dims
        # self.encoder_dims = generator_config.encoder_dims 
        self.net = DenoisingNet(point_dim=3, cond_dims=self.cond_dims, residual=False)
        # self.net = DenoisingNet_Transformer(cond_dims=self.cond_dims, residual=True)
        self.var_sched = VarianceSchedule(generator_config)
        self.interval_nums = generator_config.interval_nums

    def forward(self, coords, cond, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
        """
        batch_size, _, point_dim = coords.size()

        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0

        for i in range(self.interval_nums):
            t = ts[i].tolist()
            
            alphas_cumprod = self.var_sched.alphas_cumprod[t]
            beta = self.var_sched.betas[t]
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       # (B, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)   # (B, 1, 1)
            
            noise = torch.randn_like(coords)  # (B, N, d)
            pred_noise = self.net(sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise, beta=beta, cond=cond)
            loss = F.mse_loss(noise.view(-1, point_dim), pred_noise.view(-1, point_dim), reduction='mean')
            total_loss += (loss * (1.0 / self.interval_nums))

        return total_loss

# from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
# class Group(nn.Module):  # FPS + KNN
#     def __init__(self, num_group, group_size):
#         super().__init__()
#         self.num_group = num_group
#         self.group_size = group_size
#         self.knn = KNN(k=self.group_size, transpose_mode=True)

#     def forward(self, xyz):
#         '''
#             input: B N 3
#             ---------------------------
#             output: B G M 3
#             center : B G 3
#         '''
#         batch_size, num_points, _ = xyz.shape
#         # fps the centers out
#         center = self.fps(xyz, self.num_group) # B G 3
#         # knn to get the neighborhood
#         _, idx = self.knn(xyz, center) # B G M
#         assert idx.size(1) == self.num_group
#         assert idx.size(2) == self.group_size
#         idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
#         idx = idx + idx_base
#         idx = idx.view(-1)
#         neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
#         neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
#         # normalize
#         neighborhood = neighborhood - center.unsqueeze(2)
#         return neighborhood, center
    
#     def fps(data, number):
#         '''
#             data B N 3
#             number int
#         '''
#         fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
#         fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
#         return fps_data


# ==========Pointmaed相关导入==========
from timm.layers import DropPath
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x       
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x  
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x
    
