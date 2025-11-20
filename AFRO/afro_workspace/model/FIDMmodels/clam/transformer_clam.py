# transformer_idm.py
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig

# 假设项目中已有的 get_pos_encoding（可能返回 nn.Embedding 或 Tensor）
from afro_workspace.model.FIDMmodels.utils.transformer_utils import get_pos_encoding


class TransformerIDM(nn.Module):
    """
    TransformerIDM:
      - 输入：observations [B, T, D_in] （优先期望 T==2，若 T>2 默认取最后两个时间步）
      - 使用 CLS token 与 pos-embedding，把 CLS 的输出映射为 latent action z [B, la_dim]
      - 使用 encoder (ACTEncoder) 的 pre-norm 变体
    """

    def __init__(self, cfg: DictConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        # embedding dims
        self.input_embed_dim = getattr(cfg, "input_embed_dim", None)
        assert self.input_embed_dim is not None, "cfg.input_embed_dim required"
        # encoder expected model dim
        self.model_dim = cfg.net.dim_model

        # if input_embed_dim != model_dim, keep a projection to match the encoder
        if self.input_embed_dim != self.model_dim:
            self.input_proj = nn.Linear(input_dim, self.input_embed_dim)
            self.align_proj = nn.Linear(self.input_embed_dim, self.model_dim)
        else:
            self.input_proj = nn.Linear(input_dim, self.model_dim)
            self.align_proj = nn.Identity()

        self.activation = nn.LeakyReLU(0.2)

        # encoder and mapping to latent action
        from diffusion_policy_3d.model.FIDMmodels.act.models import ACTEncoder  # import local module
        self.encoder = ACTEncoder(cfg.net)  # encoder expects dim_model == cfg.net.dim_model

        # CLS token to aggregate pair-level info
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02)

        # latent action head: map encoder dim -> la_dim
        self.la_dim = cfg.la_dim
        self.latent_action = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.la_dim),
        )

        # pos encoding (could be nn.Embedding or precomputed Tensor)
        self.pos_embed_src = get_pos_encoding(
            cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

    def _build_pos_embed(self, timesteps: Optional[torch.Tensor], seq_len: int, device: torch.device):
        """
        Return pos_embed tensor of shape [B, seq_len, model_dim] or None.
        Handles both learned (nn.Embedding) and fixed (Tensor) pos encodings.
        timesteps: LongTensor [B, seq_len] or [B, ...] ; if None, uses arange for positions.
        """
        posobj = self.pos_embed_src
        if posobj is None:
            return None

        if isinstance(posobj, nn.Embedding):
            # learned: needs indices [B, seq_len]
            if timesteps is None:
                idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(1, seq_len)  # [1, seq_len]
                idx = idx.long()
            else:
                idx = timesteps.long()
                # ensure shape [B, seq_len]
                if idx.dim() == 1:
                    idx = idx.unsqueeze(0)
            # if idx shape [B, seq_len], embedding will return [B, seq_len, D]
            return posobj(idx)
        else:
            # assume posobj is a Tensor of shape [max_len, D] or a callable that returns such
            if callable(posobj):
                # some implementations return a tensor when called with seq_len
                pe = posobj(seq_len)
            else:
                pe = posobj
            # pe: [max_len, D]
            if timesteps is None:
                # take first seq_len entries and expand to batch
                pe_sel = pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, D]
                return pe_sel  # broadcast by batch in usage
            else:
                # timesteps: [B, seq_len] of indices
                idx = timesteps.long()
                # gather along 0
                # pe[idx] not directly work if idx is 2D, so use indexing
                # result shape [B, seq_len, D]
                return pe[idx]

    def forward(self, observations: torch.Tensor, timesteps: Optional[torch.Tensor] = None, causal: bool = False, **kwargs):
        """
        observations: [B, T, D_in]
        timesteps: optional positional indices [B, T] (LongTensor)
        returns:
            la: [B, la_dim]      # latent action per provided pair (CLS aggregation)
            cls_feat: [B, model_dim] # (optional) raw CLS embedding before LA head
            obs_feat_pair: [B, 2, model_dim] # encoder features for the two frames
        """
        B, T, D_in = observations.shape
        device = observations.device

        obs_pair = observations  # [B,2,D_in]
        timesteps_pair = timesteps

        # embed each observation -> model_dim
        x = self.input_proj(obs_pair)  # [B,2,input_embed_dim or model_dim]
        x = self.activation(x)
        x = self.align_proj(x)  # now [B,2,model_dim]

        # build sequence [CLS, obs_t, obs_t+1] with corresponding pos embeddings
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,model_dim]
        seq = torch.cat([cls, x], dim=1)  # [B, 3, model_dim]
        seq_len = seq.size(1)

        # build pos embeddings for sequence:
        if timesteps_pair is None:
            pos_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, seq_len)
        else:
            obs_pos = timesteps_pair.long()  # [B,2]
            # build [cls_idx, obs_pos[:,0]+1, obs_pos[:,1]+1]
            pos_idx = torch.zeros(B, seq_len, device=device, dtype=torch.long)
            pos_idx[:, 0] = 0  # cls
            pos_idx[:, 1] = obs_pos[:, 0] + 1
            pos_idx[:, 2] = obs_pos[:, 1] + 1

        pos_embed = self._build_pos_embed(pos_idx, seq_len=seq_len, device=device)
        # Note: pos_embed might be [1, seq_len, D] or [B, seq_len, D], encoder handles both

        # pass through encoder
        # encoder signature supports pos_embed [B, seq_len, D] or [seq_len, D]
        encoded = self.encoder(seq, pos_embed=pos_embed)  # [B, seq_len, model_dim]

        # take CLS output (index 0) as aggregated pair-level representation
        cls_out = encoded[:, 0, :]  # [B, model_dim]

        la = self.latent_action(cls_out)  # [B, la_dim]

        # also return the two frame features (for potential downstream losses/analysis)
        obs_feat_pair = encoded[:, 1:, :]  # [B, 2, model_dim]

        # return la, cls_out, obs_feat_pair
        return la

