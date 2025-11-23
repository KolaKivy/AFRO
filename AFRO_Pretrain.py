# if __name__ == "__main__":
#     import sys
#     import os
#     import pathlib

#     ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
#     sys.path.append(ROOT_DIR)
#     os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from afro_workspace.policy.afro import AFRO
from afro_workspace.dataset.base_dataset import BaseDataset
from afro_workspace.common.checkpoint_util import TopKCheckpointManager
from afro_workspace.common.pytorch_util import dict_apply, optimizer_to
from afro_workspace.model.common.lr_scheduler import get_scheduler
import matplotlib.pyplot as plt

from afro_workspace.common.addition import plot_history, calculate_average_metrics

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: AFRO = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        print(f"Dataset: {cfg.task.dataset}")
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        cprint(f"Validation dataset_val size: {cfg.val_dataloader.batch_size}", "yellow")
        cprint(f"Validation dataset size: {cfg.dataloader.batch_size}", "yellow")

        self.model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        train_history = list()

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            plot_loss = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch, local_epoch_idx)

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    step_log.update(loss_dict)

                    plot_loss.append(loss_dict)

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # print("loss_total: {:.5f}, L_idm_action: {:.5f}, vis_diff_loss: {:.5f}".format(
                #     loss_dict['loss_total'], loss_dict['L_idm_action'], loss_dict['vis_diff_loss']))
                # print("loss_total: {:.5f}, L_idm_action: {:.5f}, vis_diff_loss: {:.5f}, sta_diff_loss: {:.5f}, consis_loss: {:.5f}, ema_consis_loss: {:.5f}".format(
                #     loss_dict['loss_total'], loss_dict['L_idm_action'], loss_dict['vis_diff_loss'], loss_dict['sta_diff_loss'], loss_dict['consis_loss'], loss_dict['ema_consis_loss']))
                # print("loss_total: {:.5f}, L_idm_action: {:.5f}, vis_diff_loss: {:.5f}, sta_diff_loss: {:.5f}, consis_loss: {:.5f}".format(
                #     loss_dict['loss_total'], loss_dict['L_idm_action'], loss_dict['vis_diff_loss'], loss_dict['sta_diff_loss'], loss_dict['consis_loss']))
                # print("consis_weak_loss: {:.5f}, vis_diff_loss: {:.5f}, consis_feat_loss: {:.5f}".format(
                #     loss_dict['consis_weak_loss'], loss_dict['vis_diff_loss'], loss_dict['consis_feat_loss']))
                
                # print("vicreg_inv_mse: {:.5f}, vicreg_var: {:.5f}, vicreg_cov: {:.5f}".format(
                #     loss_dict['vicreg_inv_mse'], loss_dict['vicreg_var'], loss_dict['vicreg_cov']))
                # print("vicreg_loss: {:.5f}, cinfo_loss: {:.5f}".format(
                #     loss_dict['vicreg_loss'], loss_dict['cinfo_loss']))
                # print("v_mse: {:.5f}".format(
                #     loss_dict['v_mse']))
                
                # print("consis_weak_loss: {:.5f}, vis_diff_loss: {:.5f}, consis_feat_loss: {:.5f}, vicreg_loss: {:.5f}".format(
                #     loss_dict['consis_weak_loss'], loss_dict['vis_diff_loss'], loss_dict['consis_feat_loss'], loss_dict['vicreg_loss']))
                # print("consis_weak_loss: {:.5f}, vis_diff_loss: {:.5f}, consis_feat_loss: {:.5f}".format(
                #     loss_dict['consis_weak_loss'], loss_dict['vis_diff_loss'], loss_dict['consis_feat_loss']))

                # print("consis_weak_loss: {:.5f}, vis_diff_loss: {:.5f}, consis_feat_loss: {:.5f}, long_term_loss: {:.5f}".format(
                #     loss_dict['consis_weak_loss'], loss_dict['vis_diff_loss'], loss_dict['consis_feat_loss'], loss_dict['long_term_loss']))

                # print("long_term_feat_loss: {:.5f}, long_term_contrast_loss: {:.5f}, long_term_vicreg_loss: {:.5f}".format(
                #     loss_dict['long_term_feat_loss'], loss_dict['long_term_contrast_loss'], loss_dict['long_term_vicreg_loss']))
                
                # print("consis_weak_loss: {:.5f}, vis_diff_loss: {:.5f}, consis_feat_loss: {:.5f}, long_term_contrast_loss: {:.5f}, long_term_feat_loss: {:.5f}".format(
                #     loss_dict['consis_weak_loss'], loss_dict['vis_diff_loss'], loss_dict['consis_feat_loss'], loss_dict['long_term_contrast_loss'], loss_dict['long_term_feat_loss']))

                # print(
                #     "loss_total: {:.5f} | fwd_vicreg: {:.5f} (inv:{:.5f}, var:{:.5f}, cov:{:.5f}) | "
                #     "rev_vicreg: {:.5f} (inv:{:.5f}, var:{:.5f}, cov:{:.5f}) | symmetry: {:.5f} | pairs: {}"
                #     .format(
                #         loss_dict['loss_total'],
                #         loss_dict['vicreg_fwd_total'], loss_dict['vicreg_fwd_inv_mse'], loss_dict['vicreg_fwd_var'], loss_dict['vicreg_fwd_cov'],
                #         loss_dict['vicreg_rev_total'], loss_dict['vicreg_rev_inv_mse'], loss_dict['vicreg_rev_var'], loss_dict['vicreg_rev_cov'],
                #         loss_dict['symmetry_loss'],
                #         loss_dict['pairs'],
                #     )
                # )
                
            # print(
            #     "loss_total: {:.5f} | fwd_vicreg: {:.5f} (inv:{:.5f}, var:{:.5f}, cov:{:.5f}) | pairs: {}"
            #     .format(
            #         loss_dict['loss_total'],
            #         loss_dict['vicreg_fwd_total'], loss_dict['vicreg_fwd_inv_mse'], loss_dict['vicreg_fwd_var'], loss_dict['vicreg_fwd_cov'],
            #         loss_dict['pairs'],
            #     )
            # )

            # print(
            #     "loss_total: {:.5f} | ce: {:.5f} | cov_penalty: {:.5f} | var_penalty: {:.5f} | ema_decay: {:.6f} | pairs: {}"
            #     .format(
            #         loss_dict['loss_total'],
            #         loss_dict['loss_ce'],
            #         loss_dict['cov_penalty'],
            #         loss_dict['var_penalty'],
            #         loss_dict['ema_decay'],
            #         loss_dict['pairs'],
            #     )
            # )

            # print(
            #     "loss_total: {:.5f} | "
            #     "fwd_vicreg: {:.5f} | fwd_inv: {:.5f} | fwd_cov: {:.5f} | fwd_var: {:.5f} | "
            #     "rev_vicreg(x{:.2f}): {:.5f} | rev_inv: {:.5f} | rev_cov: {:.5f} | rev_var: {:.5f} | "
            #     "la_norm: {:.5f} | pairs: {}"
            #     .format(
            #         loss_dict['loss_total'],
            #         loss_dict['vicreg_fwd_total'],
            #         loss_dict['vicreg_fwd_inv_mse'],
            #         loss_dict['vicreg_fwd_cov'],
            #         loss_dict['vicreg_fwd_var'],
            #         # float(self.lambda_bidir),
            #         loss_dict['vicreg_rev_total'],
            #         loss_dict['vicreg_rev_inv_mse'],
            #         loss_dict['vicreg_rev_cov'],
            #         loss_dict['vicreg_rev_var'],
            #         loss_dict['la_norm_mean'],
            #         loss_dict['pairs'],
            #     )
            # )
               
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss 
                
            plot_loss_dict = calculate_average_metrics(plot_loss)
            train_history.append(plot_loss_dict)

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(tag='pretrain_latest')
                plot_history(train_history, self.epoch, self.output_dir, self.cfg.training.seed)

            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

        plot_history(train_history, self.epoch-1, self.output_dir, self.cfg.training.seed)
        self.save_checkpoint(tag='pretrain_latest')
        print(f'Saved plots to {self.output_dir}')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")         
            
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'afro_workspace', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
