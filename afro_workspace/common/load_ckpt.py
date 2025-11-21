import torch
from termcolor import cprint
def load_ckpt(self, cfg):
    checkpoint = torch.load(cfg.policy.checkpoint, map_location='cpu')
    vis_encoder_weights = {}
    # dp3 encoder.mlp
    # for key, value in checkpoint['state_dicts']['model'].items():
    #     # if 'extractor' in key: 
    #     #     new_key = key.replace('obs_encoder.extractor.', '')
    #     #     vis_encoder_weights[new_key] = value
    #     if key.startswith('ema_vis_encoder.extractor.mlp.'):
    #             # 移除 'mlp.' 前缀
    #         new_key = key.replace('ema_vis_encoder.extractor.mlp.', '')
    #         vis_encoder_weights[new_key] = value

    # afro vis encoder
    # for key, value in checkpoint['state_dicts']['model'].items():
    #     if 'vis_encoder' in key: 
    #         if 'ema_vis_encoder' not in key and 'target_vis_encoder' not in key:
    #             new_key = key.replace('vis_encoder.extractor.', '')
    #             vis_encoder_weights[new_key] = value
    # self.model.obs_encoder.extractor.load_state_dict(vis_encoder_weights)

    # dp3 encoder
    for key, value in checkpoint['state_dicts']['model'].items():
        if 'obs_encoder' in key: 
            if 'extractor' not in key:
                new_key = key.replace('vis_encoder.extractor.', '')
                vis_encoder_weights[new_key] = value
    self.model.obs_encoder.extractor.load_state_dict(vis_encoder_weights)

    # dp3 policy checkpoint
    # self.model.load_state_dict(checkpoint['state_dicts']['model'])
    
    # for param in self.model.mlp.parameters():
    #     param.requires_grad = False
    cprint(f"load obs_encoder.extractor successful", "yellow")
