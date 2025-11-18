import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import VisEncoder, DiTFDMDncoder


class AFRO(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # ===== param =====
            ema_momentum_start=0.996,
            ema_target_epoch=300,
            lambda_long_term=1.0,
            lambda_reverse=1.0,
            fdm_d_model=256,
            # ===== VICReg 目标权重（最终稳定到的数值） =====
            vicreg_inv_weight=25.0,
            vicreg_var_weight=25.0,
            vicreg_cov_weight=1.0,
            vicreg_eps=1e-4,
            # ===== VICReg 预热配置 =====
            vicreg_warmup_start=0,        
            vicreg_warmup_epochs=0,   
            **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: 
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # ===== Student / Teacher encoders (Teacher = EMA of Student) =====
        self.vis_encoder = VisEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        self.ema_vis_encoder = VisEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # self.ema_vis_encoder = copy.deepcopy(self.vis_encoder)
        for p in self.ema_vis_encoder.parameters():
            p.requires_grad = False

        self.obs_feature_dim = self.vis_encoder.output_shape()  # D
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[AFRO] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[AFRO] pointnet_type: {self.pointnet_type}", "yellow")

        # ======= Projector（带 EMA）=======
        D = self.obs_feature_dim

        # bookkeeping
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.kwargs = kwargs

        # ===== IDM: Δf -> 16 dim =====
        self.latent_action_dim = 16
        self.idm_mlp = nn.Sequential(
            nn.Linear(D, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_action_dim)
        )

        # ===== FDM =====
        self.fdm_vis_decoder = DiTFDMDncoder(input_dim=D, hidden=fdm_d_model, depth=4, num_heads=4)
        self.noise_scheduler = noise_scheduler

        # training details
        self.lambda_long_term = lambda_long_term
        self.lambda_reverse = lambda_reverse

        # EMA 
        self.ema_momentum_start = float(ema_momentum_start)
        self.ema_target_epoch = int(ema_target_epoch)

        # VICReg 目标权重（最终稳定值）
        self.vicreg_inv_weight = float(vicreg_inv_weight)
        self.vicreg_var_weight = float(vicreg_var_weight)
        self.vicreg_cov_weight = float(vicreg_cov_weight)
        self.vicreg_eps = vicreg_eps

        # VICReg 预热配置
        self.vicreg_warmup_start = int(vicreg_warmup_start)
        self.vicreg_warmup_epochs = int(vicreg_warmup_epochs)

        print_params(self)

    # ========= VICReg（直接在特征空间）=========
    def _vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                    inv_w: float, var_w: float, cov_w: float):

        inv_loss = F.mse_loss(z1, z2)

        z1c = z1 - z1.mean(dim=0, keepdim=True)
        z2c = z2 - z2.mean(dim=0, keepdim=True)

        eps = getattr(self, "vicreg_eps", 1e-4)
        std1 = torch.sqrt(z1c.var(dim=0, unbiased=False) + eps)
        std2 = torch.sqrt(z2c.var(dim=0, unbiased=False) + eps)
        var_loss = 0.5 * (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean())

        def _covariance_term(zc: torch.Tensor) -> torch.Tensor:
            N, D = zc.shape
            if N <= 1:
                return zc.new_zeros(())
            cov = (zc.T @ zc) / (N - 1.0)         # [D, D]
            off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
            return off / D

        cov_loss = _covariance_term(z1c) + _covariance_term(z2c)

        vicreg = inv_w * inv_loss + var_w * var_loss + cov_w * cov_loss
        return vicreg, inv_loss, var_loss, cov_loss


    # ========= VICReg 预热系数 =========
    def _warmup_scale(self, epoch: int) -> float:
        start = self.vicreg_warmup_start
        warm = self.vicreg_warmup_epochs
        if warm <= 0:
            return 1.0
        if epoch < start:
            return 0.0
        if epoch >= start + warm:
            return 1.0

        e = epoch - start
        return 0.5 * (1.0 - math.cos(math.pi * float(e) / float(warm)))

    # ========= EMA =========
    def _ema_momentum(self, epoch: int) -> float:
        e = max(0, min(epoch, self.ema_target_epoch))
        frac = e / float(self.ema_target_epoch) if self.ema_target_epoch > 0 else 1.0
        return self.ema_momentum_start + (1.0 - self.ema_momentum_start) * frac

    @torch.no_grad()
    def update_teacher(self, epoch: int):
        m = self._ema_momentum(epoch)

        def _ema_update_module(tgt: nn.Module, src: nn.Module):
            for p_t, p_s in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))
            for b_t, b_s in zip(tgt.buffers(), src.buffers()):
                if b_t.dtype.is_floating_point:
                    b_t.data.mul_(m).add_(b_s.data, alpha=(1.0 - m))
                else:
                    b_t.data.copy_(b_s.data)

        _ema_update_module(self.ema_vis_encoder, self.vis_encoder)

    # ========= Utils =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ========= Loss =========
    def compute_loss(self, batch, epoch):
        warm_scale = self._warmup_scale(epoch)
        var_w_now = self.vicreg_var_weight * warm_scale
        cov_w_now = self.vicreg_cov_weight * warm_scale
        inv_w_now = self.vicreg_inv_weight   

        nobs = self.normalizer.normalize(batch['obs'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        B, T, _, _ = nobs['point_cloud'].shape
        device = nobs['point_cloud'].device

        interval = 2
        sampled_times = list(range(0, T, interval))
        M = len(sampled_times)

        nobs_sel = {k: v[:, sampled_times, ...] for k, v in nobs.items()}
        nobs_sel_flat = dict_apply(nobs_sel, lambda x: x.reshape(-1, *x.shape[2:]))

        # Student 前向
        feats_stu_flat = self.vis_encoder(nobs_sel_flat)  # [B*M, D]
        D = feats_stu_flat.shape[-1]
        feats_stu = feats_stu_flat.view(B, M, D)

        # Teacher 编码（EMA，无梯度）
        with torch.no_grad():
            feats_tea_flat = self.ema_vis_encoder(nobs_sel_flat)  # [B*M, D]
            feats_tea = feats_tea_flat.view(B, M, D)

        # Student
        f_t_stu   = feats_stu[:, :-1, :]               # [B, M-1, D]
        f_tp1_stu = feats_stu[:,  1:, :]
        # Teacher
        f_t_tea   = feats_tea[:, :-1, :]
        f_tp1_tea = feats_tea[:,  1:, :]

        # ===== 正向：差分 -> 16维动作 =====
        delta_stu = f_tp1_stu - f_t_stu                 # [B, M-1, D]
        mem = self.idm_mlp(delta_stu)                   # [B, M-1, 16]

        BM = B * (M - 1)
        f_t_cond     = f_t_stu.reshape(BM, D)
        mem_flat     = mem.reshape(BM, self.latent_action_dim)  # 注意：mem 是 16 维
        target_flat  = f_tp1_tea.reshape(BM, D)

        t_flat = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (BM,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(target_flat)
        noisy_target = self.noise_scheduler.add_noise(target_flat, noise, t_flat)
        pred_flat = self.fdm_vis_decoder(noisy_target, t_flat, f_t_cond, mem_flat)  # [BM, D]

        long_term_vicreg_loss, inv_loss, var_loss, cov_loss = self._vicreg_loss(
            pred_flat, target_flat,
            inv_w=inv_w_now, var_w=var_w_now, cov_w=cov_w_now
        )
        long_term_loss = long_term_vicreg_loss

        # ===== 逆向：预测 f_t，用教师目标做监督 =====
        delta_rev = f_t_stu - f_tp1_stu                  # [B, M-1, D]
        mem_rev = self.idm_mlp(delta_rev)                # [B, M-1, 16]

        f_tp1_cond = f_tp1_stu.reshape(BM, D)
        mem_rev_flat = mem_rev.reshape(BM, self.latent_action_dim)
        target_rev_tea_flat = f_t_tea.reshape(BM, D).detach()
        t_rev_tea = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (BM,), device=device, dtype=torch.long
        )
        noise_rev_tea = torch.randn_like(target_rev_tea_flat)
        noisy_target_rev_tea = self.noise_scheduler.add_noise(target_rev_tea_flat, noise_rev_tea, t_rev_tea)
        pred_rev_tea_flat = self.fdm_vis_decoder(noisy_target_rev_tea, t_rev_tea, f_tp1_cond, mem_rev_flat)
        
        reverse_vicreg, reverse_inv_loss, reverse_var_loss, reverse_cov_loss = self._vicreg_loss(
            pred_rev_tea_flat, target_rev_tea_flat,
            inv_w=inv_w_now, var_w=var_w_now, cov_w=cov_w_now
        )

        # ===== 总损失 =====
        loss = self.lambda_long_term * long_term_loss + self.lambda_reverse * reverse_vicreg
        
        self.update_teacher(epoch)

        return loss, {
            'loss_total': float(loss.item()),
            'inv_loss': float(inv_loss.item()),
            'var_loss': float(var_loss.item()),
            'cov_loss': float(cov_loss.item()),
            'reverse_mse': float(reverse_inv_loss.item()),
            'reverse_var_loss': float(reverse_var_loss.item()),
            'reverse_cov_loss': float(reverse_cov_loss.item()),
            # 记录当下生效的 VICReg 权重（含预热）
            'vicreg_inv_w_now': float(inv_w_now),
            'vicreg_var_w_now': float(var_w_now),
            'vicreg_cov_w_now': float(cov_w_now),
            'vicreg_warmup_scale': float(warm_scale),
        }
