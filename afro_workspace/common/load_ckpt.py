import torch
from termcolor import cprint
def load_ckpt(self, cfg):
    checkpoint = torch.load(cfg.policy.checkpoint, map_location='cpu')
    vis_encoder_weights = {}
    for key, value in checkpoint['state_dicts']['model'].items():
        if 'obs_encoder' in key: 
            if 'extractor' not in key:
                new_key = key.replace('vis_encoder.extractor.', '')
                vis_encoder_weights[new_key] = value
    self.model.obs_encoder.extractor.load_state_dict(vis_encoder_weights)
    cprint(f"load obs_encoder.extractor successful", "yellow")
