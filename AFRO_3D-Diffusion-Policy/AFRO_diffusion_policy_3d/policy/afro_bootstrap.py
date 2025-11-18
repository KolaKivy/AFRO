import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import VisEncoder, TransformerIDMEncoder, FDMDecoder, DiTFDMDncoder


class AFRO(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_obs_steps,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            idm_cfg=None,
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

        # visual encoder
        self.vis_encoder = VisEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # EMA visual encoder
        self.ema_vis_encoder = VisEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        self.obs_feature_dim = self.vis_encoder.output_shape()
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # bookkeeping
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs
        self.action_feat_dim = 32
        self.encoder_output_dim = encoder_output_dim

        vis_in_dim = encoder_output_dim
        d_model = kwargs.get('fdm_d_model', 256)
        
        # IDM encoder 
        self.vte = TransformerIDMEncoder(cfg=idm_cfg, in_dim=vis_in_dim)

        # FDM
        self.fdm_vis_action_mem_in = nn.Linear(self.action_feat_dim, vis_in_dim)
        self.fdm_vis_decoder = DiTFDMDncoder(input_dim=vis_in_dim, hidden=d_model, depth=4, num_heads=4)
        self.noise_scheduler = noise_scheduler  

        # EMA encoder
        for p in self.ema_vis_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = kwargs.get('ema_decay', 0.996)

        self.feat_dropout_p = 0.0

        # ====== projector（online / target）======
        self.proj_out_dim = kwargs.get('proj_out_dim', vis_in_dim)
        def make_projector(in_dim, out_dim):
            hid = 2 * out_dim
            return nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, out_dim)
            )

        self.online_projector = make_projector(vis_in_dim, self.proj_out_dim)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # ====== VICReg weights ======
        self.vicreg_inv_weight = kwargs.get('vicreg_inv_weight', 25.0)  # invariance (MSE between projected pairs)
        self.vicreg_var_weight = kwargs.get('vicreg_var_weight', 25.0)
        self.vicreg_cov_weight = kwargs.get('vicreg_cov_weight', 1.0)
        self.vicreg_eps = kwargs.get('vicreg_eps', 1e-4)

        self.fixed_interval_K = kwargs.get('pair_interval', 4)

        print_params(self)

    # ====== EMA ======
    def get_ema_decay(self, step, max_steps=300, final_decay=0.9999, initial_decay=0.996):
        return initial_decay + (final_decay - initial_decay) * float(step) / float(max_steps)
    
    def _update_ema_modules(self, epoch):
        decay = self.get_ema_decay(epoch, max_steps=300, final_decay=0.9999, initial_decay=self.ema_decay)
        # encoder
        for p_online, p_ema in zip(self.vis_encoder.parameters(), self.ema_vis_encoder.parameters()):
            p_ema.data.mul_(decay).add_(p_online.data * (1.0 - decay))
        # projector
        for p_online, p_ema in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            p_ema.data.mul_(decay).add_(p_online.data * (1.0 - decay))

    # ====== VICReg ======
    def _vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # 1) Invariance（pairwise MSE）
        inv_loss = F.mse_loss(z1, z2)
        # 2) Variance
        def _variance_term(z):
            std = torch.sqrt(z.var(dim=0) + self.vicreg_eps)
            return torch.mean(F.relu(1.0 - std))
        var_loss = 0.5 * (_variance_term(z1) + _variance_term(z2))

        # 3) Covariance
        def _covariance_term(z):
            Nz, Dz = z.size()
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (Nz - 1.0)  # [D, D]
            diag = torch.diagonal(cov)
            cov_loss = (cov.pow(2).sum() - diag.pow(2).sum()) / Dz
            return cov_loss
        cov_loss = 0.5 * (_covariance_term(z1) + _covariance_term(z2))

        loss = (self.vicreg_inv_weight * inv_loss
                + self.vicreg_var_weight * var_loss
                + self.vicreg_cov_weight * cov_loss)
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _sample_student_pointcloud(self, pc_bt):
        """
        pc_bt: [B, M, N, C] -> [B, M, N2, C] (N2 < N) 
        """
        B, M, N, C = pc_bt.shape
        device = pc_bt.device
        keep = float(torch.empty(()).uniform_(0.2, 0.5))  
        target_num = int(N * max(0.01, keep))
        rand = torch.rand(B, M, N, device=device)            # [B, M, N]
        perm = rand.argsort(dim=-1)                          # [B, M, N]  
        idx = perm[..., :target_num]                         # [B, M, N2]
        pc_flat = pc_bt.reshape(B*M, N, C)
        idx_flat = idx.reshape(B*M, target_num)
        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C)
        out_flat = torch.gather(pc_flat, dim=1, index=idx_expanded)  # [B*M, N2, C]
        return out_flat.reshape(B, M, target_num, C)

    @torch.no_grad()
    def _gather_pairs_indices(self, T_eff: int, K: int, device):
        if T_eff - K <= 0:
            return None, None  # 无可用配对
        idx_t = torch.arange(0, T_eff - K, device=device)
        idx_tp = idx_t + K
        return idx_t, idx_tp

    def compute_loss(self, batch, epoch):
        nobs = self.normalizer.normalize(batch['obs'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        B, T, Da, _ = nobs['point_cloud'].shape
        device = nobs['point_cloud'].device

        self._update_ema_modules(epoch)

        interval = self.fixed_interval_K  
        sampled_times = list(range(0, T, interval))
        compressed_T = len(sampled_times)

        this_nobs_sel = {k: v[:, sampled_times, ...] for k, v in nobs.items()}  # [B, M, ...], M=compressed_T

        # ----- 教师（EMA）分支 -----
        this_nobs_sel_flat_teacher = dict_apply(this_nobs_sel, lambda x: x.reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            ema_enc_out = self.ema_vis_encoder(this_nobs_sel_flat_teacher)  # [B*M, D]
        feat_dim = ema_enc_out.shape[-1]
        teacher_feats = ema_enc_out.view(B, compressed_T, feat_dim)         # [B, M, D]

        # ----- 学生分支（点云下采样一点增强）-----
        pc_sel = this_nobs_sel['point_cloud']                  # [B, M, N, C]
        pc_sel_sampled = self._sample_student_pointcloud(pc_sel)
        this_nobs_sel_student = {}
        for k, v in this_nobs_sel.items():
            if k == 'point_cloud':
                this_nobs_sel_student[k] = pc_sel_sampled
            else:
                this_nobs_sel_student[k] = v
        this_nobs_sel_student_flat = dict_apply(this_nobs_sel_student, lambda x: x.reshape(-1, *x.shape[2:]))
        enc_out_student_flat = self.vis_encoder(this_nobs_sel_student_flat)  # [B*M, D]
        student_feats = enc_out_student_flat.view(B, compressed_T, feat_dim)  # [B, M, D]

        # ====== 生成 (t, t+K) 批次配对 ======
        K = self.fixed_interval_K
        idx_t, idx_tp = self._gather_pairs_indices(compressed_T, K, device)

        start_feat_student = student_feats[:, idx_t, :]   # [B, P, D]
        fut_feat_student   = student_feats[:, idx_tp, :]  # [B, P, D]
        target_ema         = teacher_feats[:, idx_tp, :].detach()  # [B, P, D]

        # dropout
        start_feat_student = F.dropout(start_feat_student, p=self.feat_dropout_p, training=self.training)
        fut_feat_student   = F.dropout(fut_feat_student,   p=self.feat_dropout_p, training=self.training)

        B_, P_, D_ = start_feat_student.shape
        start_flat = start_feat_student.reshape(B_ * P_, D_)
        fut_flat   = fut_feat_student.reshape(B_ * P_, D_)
        start_proj_for_cond = self.online_projector(start_flat)
        fut_flat_for_cond = self.online_projector(fut_flat)
        
        v_in_student = torch.stack([start_proj_for_cond, fut_flat_for_cond], dim=1)  # [B*P, 2, D]
        ts_pairs = None
        action_feat_student = self.vte(v_in_student, timesteps=ts_pairs)  # [B*P, action_feat_dim]
        mem_start = self.fdm_vis_action_mem_in(action_feat_student)       # [B*P, D]

        target_flat = target_ema.reshape(B_ * P_, D_)
        noise = torch.randn_like(target_flat)
        t_b = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B_ * P_,), device=device).long()
        noisy_feature = self.noise_scheduler.add_noise(target_flat, noise, t_b)

        pred_next = self.fdm_vis_decoder(noisy_feature, t_b, start_proj_for_cond, mem_start)  # [B*P, D]

        pred_proj   = self.online_projector(pred_next)              # [B*P, st_in_dim]
        with torch.no_grad():
            tgt_proj = self.target_projector(target_flat).detach()  # [B*P, st_in_dim]

        vicreg_loss = self._vicreg_loss(pred_proj, tgt_proj)
        loss = vicreg_loss  

        metrics = {
            'loss_total': float(loss.item()),
            'vicreg_loss': float(vicreg_loss.item()),
            'pairs': int(P_)
        }

        return loss, metrics
