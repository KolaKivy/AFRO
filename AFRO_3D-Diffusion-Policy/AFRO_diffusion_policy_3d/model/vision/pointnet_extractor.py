import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

import numpy as np

from diffusion_policy_3d.model.transformer.STtransformer import SpatioTransformer,TemporalTransformer
from diffusion_policy_3d.model.FIDMmodels.clam.transformer_clam import TransformerIDM
from diffusion_policy_3d.model.transformer.DiT1D import DiTFeaturePredictor
from diffusion_policy_3d.model.transformer.DiT1D_Simple import DiTFeaturePredictor_simple

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class PointNetEncoderXYZ_AVG(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):

        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
    def forward(self, x):
        x = self.mlp(x)
        # max_x = torch.max(x, 1)[0]
        mean_x_mid = torch.mean(x, 1)
        mean_x = self.final_projection(mean_x_mid)
        return mean_x, mean_x_mid

class PointNetEncoderXYZ_SoftmaxPool(nn.Module):
    """Encoder for Pointcloud with softmax-pooling"""

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):

        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x):
        x = self.mlp(x)  # (B, N, C)
        w = F.softmax(x, dim=1)        # (B, N, C)
        softmax_x = torch.sum(x * w, dim=1)  # (B, C)
        softmax_x = self.final_projection(softmax_x)
        return softmax_x


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            cprint(f"[DP3Encoder] use pointnet", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointnet_avg":
            cprint(f"[DP3Encoder] use pointnet", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ_AVG(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointtransformer":
            cprint(f"use pointtransformer", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
            else:
                pointcloud_encoder_cfg.in_channels = 3
            
            from .pointtransformer_v1.pointTransformer_v1_noKNN import PurePointTransformerExtractor
            self.extractor = PurePointTransformerExtractor(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels

class V2S_MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = None,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(64, in_dim * 2) 
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Action_Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = None,
        act_fn=nn.ReLU,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(64, in_dim * 2) 
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class VisEncoder(nn.Module): 
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 encoder_config=None,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
  
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[VisEncoder] pointnet_type is {pointnet_type}","red")
        if pointnet_type == "pointnet":
            cprint(f"[DP3Encoder] use pointnet", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        if pointnet_type == "pointnet_avg":
            cprint(f"[DP3Encoder] use pointnet_avg", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ_AVG(**pointcloud_encoder_cfg)
        if pointnet_type == "pointnet_soft":
            cprint(f"[DP3Encoder] use pointnet_soft", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ_SoftmaxPool(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointnet++_pretrained":
            # from .PointNet_master.log.sem_seg.pointnet2_sem_seg.pointnet2_sem_seg import pointnet2_sem_seg
            # from .PointNet_master.log.part_seg.pointnet2_part_seg_msg.pointnet2_part_seg_msg import pointnet2_part_seg_msg
            cprint(f"[DP3Encoder] use pointnet++_pretrained", "red")
            # Initialize the model with appropriate number of classes
            # The model expects (B, C, N) format input
            
            if use_pc_color:
                from .PointNet_master.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import pointnet2_cls_ssg_wo_normals
                checkpoint_path = "/mnt/data/zhoudingjie/kivy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/PointNet_master/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
                pointcloud_encoder_cfg.in_channels = 6
                # For semantic segmentation, we need to specify num_classes
                # Using a reasonable default, adjust if needed
                self.extractor = pointnet2_cls_ssg_wo_normals(num_class=40)
            else:
                raise NotImplementedError(f"pointnet++_pretrained with use_pc_color: {use_pc_color}")

            # Load pretrained weights
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                self.extractor.load_state_dict(state_dict, strict=False)
                cprint(f"[DP3Encoder] Successfully loaded pretrained weights from {checkpoint_path}", "green")
            except Exception as e:
                cprint(f"[DP3Encoder] Warning: Could not load pretrained weights: {e}", "yellow")
            
            # Freeze the model parameters to prevent training
            for param in self.extractor.parameters():
                param.requires_grad = False
            cprint(f"[DP3Encoder] Froze pointnet2_sem_seg parameters", "green")
            self.final_projection = nn.Sequential(
                nn.Linear(1024, out_channel),
                nn.LayerNorm(out_channel)
            )
            self.n_output_channels = out_channel
        elif pointnet_type == "pointtransformer":
            cprint(f"use pointtransformer", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
            else:
                pointcloud_encoder_cfg.in_channels = 3
            
            from .pointtransformer_v1.pointTransformer_v1_noKNN import PurePointTransformerExtractor
            self.extractor = PurePointTransformerExtractor(**pointcloud_encoder_cfg)
        elif pointnet_type == "pointempty":
            cprint(f"[DP3Encoder] use pointempty", "red")
            self.n_output_channels = 1024
        elif pointnet_type == "pointcnn":
            from .pointcnn.pointcnn import PointCNN
            cprint(f"[DP3Encoder] use pointcnn", "red")
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointCNN(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointCNN(**pointcloud_encoder_cfg)
        elif pointnet_type == "clip":
            cprint(f"use clip", "red")
            import clip
            clip_model = "ViT-B/32"
            self.clip, self.preprocess = clip.load(clip_model, jit=False)
            cprint(f"clip loaded is {clip_model}", "red")
            self.clip = self.clip.eval().requires_grad_(False)
            self.clip = self.clip.to(self.device)
            cprint(f"[CLIP] Model loaded and moved to {self.device}", "green")
            clip_feature_dim = self.clip.visual.output_dim
            cprint(f"[CLIP] Feature dimension: {clip_feature_dim}", "green")
            self.img_projection = nn.Sequential(
                nn.Linear(clip_feature_dim, out_channel),
                nn.LayerNorm(out_channel)
            )
            self.n_output_channels = out_channel
        elif pointnet_type == "dinov2":
            cprint(f"use dinov2", "red")
            # Use timm for better Python 3.8 compatibility
            try:
                import timm
                dinov2_model = 'vit_small_patch14_dinov2.lvd142m'
                self.dinov2 = timm.create_model(dinov2_model, pretrained=True)
                cprint(f"dinov2 loaded is {dinov2_model}", "red")
                dinov2_feature_dim = 384  # DINOv2 ViT-S/14 output dimension
            except ImportError:
                raise ImportError("timm not available. Please install: pip install timm")
            
            self.dinov2 = self.dinov2.eval().requires_grad_(False)
            self.dinov2 = self.dinov2.to(self.device)
            cprint(f"[DINOv2] Model loaded and moved to {self.device}", "green")
            cprint(f"[DINOv2] Feature dimension: {dinov2_feature_dim}", "green")
            
            self.img_projection = nn.Sequential(
                nn.Linear(dinov2_feature_dim, out_channel),
                nn.LayerNorm(out_channel)
            )
            self.n_output_channels = out_channel

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        points = observations[self.point_cloud_key]
        points = points.to(device)
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        if self.pointnet_type == "pointnet" or self.pointnet_type == "pointnet_avg" or self.pointnet_type == "pointnet_soft" :
            pn_feat = self.extractor(points)    # B * out_channel
            return pn_feat
        if self.pointnet_type == "pointnet_dynamicflow":
            pn_feat = self.extractor(points)    # B * out_channel
            return pn_feat
        elif self.pointnet_type == "pointtransformer":
            pn_feat = self.extractor(points)
            return pn_feat
        elif self.pointnet_type == "pointcnn":
            pn_feat = self.extractor(points)    # B * out_channel
            return pn_feat
        elif self.pointnet_type == "pointempty":
            return torch.zeros((points.shape[0], self.n_output_channels)).to(device)
        elif self.pointnet_type == "pointnet++_pretrained":
            # Convert from (B, N, C) to (B, C, N) format for pointnet2_sem_seg
            points_transposed = points.transpose(1, 2)  # B * C * N
            with torch.no_grad():  # Ensure no gradients since model is frozen
                _, l4_points = self.extractor(points_transposed)  # Extract features from last layer
            # l4_points should be (B, feature_dim, num_points_l4)
            # Apply global max pooling to get (B, feature_dim)
            pn_feat = torch.max(l4_points, dim=2)[0]  # B * feature_dim
            # cprint(f"pn_feat shape: {pn_feat.shape}", "red")
            pn_feat = self.final_projection(pn_feat)
            return pn_feat
        elif self.pointnet_type == "clip":
            image = observations[self.rgb_image_key]
            # print(f"[CLIP DEBUG] Original image shape: {image.shape}")
            # print(f"[CLIP DEBUG] Image dtype: {image.dtype}, min: {image.min():.3f}, max: {image.max():.3f}")
            
            # Optimized GPU-only preprocessing - no PIL conversions or CPU transfers
            B = image.shape[0]
            
            # Handle different input formats
            if image.dim() == 4:
                if image.shape[-1] == 3:  # [B, H, W, C] format
                    image = image.permute(0, 3, 1, 2)  # -> [B, C, H, W]
                # else already [B, C, H, W]
            
            # Convert dtype if needed (uint8 -> float32)
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            
            # Ensure values are in [0, 1] range
            if image.max() > 1.0:
                image = image / 255.0
            
            # Resize to 224x224 if needed (GPU operation)
            if image.shape[-2:] != (224, 224):
                image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Apply CLIP normalization (ImageNet stats) - all GPU operations
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device).view(1, 3, 1, 1)
            processed_batch = (image - mean) / std
            
            with torch.no_grad():
                image_feat = self.clip.encode_image(processed_batch)
            
            image_feat = image_feat.float()
            # print(f"[CLIP DEBUG] CLIP encoded features shape: {image_feat.shape}")
            # print(f"[CLIP DEBUG] CLIP features dtype: {image_feat.dtype}")
            image_feat = self.img_projection(image_feat)
            # print(f"[CLIP DEBUG] After projection shape: {image_feat.shape}")
            return image_feat
        elif self.pointnet_type == "dinov2":
            image = observations[self.rgb_image_key]
            # print(f"[DINOv2 DEBUG] Original image shape: {image.shape}")
            # print(f"[DINOv2 DEBUG] Image dtype: {image.dtype}, min: {image.min():.3f}, max: {image.max():.3f}")
            
            # Optimized GPU-only preprocessing - no PIL conversions or CPU transfers
            B = image.shape[0]
            
            # Handle different input formats
            if image.dim() == 4:
                if image.shape[-1] == 3:  # [B, H, W, C] format
                    image = image.permute(0, 3, 1, 2)  # -> [B, C, H, W]
                # else already [B, C, H, W]
            
            # Convert dtype if needed (uint8 -> float32)
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            
            # Ensure values are in [0, 1] range
            if image.max() > 1.0:
                image = image / 255.0
            
            # Resize to 518x518 for DINOv2 model (GPU operation)
            if image.shape[-2:] != (518, 518):
                image = F.interpolate(image, size=(518, 518), mode='bilinear', align_corners=False)
            
            # Apply DINOv2 normalization (ImageNet stats) - all GPU operations
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
            processed_batch = (image - mean) / std
            
            with torch.no_grad():
                image_feat = self.dinov2(processed_batch)
            
            image_feat = image_feat.float()
            # print(f"[DINOv2 DEBUG] DINOv2 encoded features shape: {image_feat.shape}")
            # print(f"[DINOv2 DEBUG] DINOv2 features dtype: {image_feat.dtype}")
            image_feat = self.img_projection(image_feat)
            # print(f"[DINOv2 DEBUG] After projection shape: {image_feat.shape}")
            return image_feat
        elif self.pointnet_type == "pointdif":
            B,_,_ = points.shape
            neighborhood, center = self.group_divider(points)
            encoder_token, mask = self.mask_encoder(neighborhood, center) 
            return encoder_token, mask, center           

    def output_shape(self):
        return self.n_output_channels

class StaEncoder(nn.Module): 
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: Dict) -> torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        state = observations[self.state_key]
        state = state.to(device)
        state_feat = self.state_mlp(state)  # B * 64
        return state_feat

    def output_shape(self):
        return self.n_output_channels

class IDMEncoder(nn.Module): 
    def __init__(self, 
                 in_dim=64,
                 action_feat_dim=256,
                 transf_d_model=256,
                 transf_nhead=8,
                 transf_layers=2,
                 dropout = 0.0,
                 ): 
        super().__init__()
        
        self.TT = TemporalTransformer(
            in_dim=in_dim,
            model_dim=transf_d_model, 
            out_dim=action_feat_dim, 
            num_blocks=transf_layers, 
            num_heads=transf_nhead, 
            dropout=dropout
        )

    def forward(self, v_in):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        latent_action = self.TT(v_in)  
        return latent_action

class FDMDecoder(nn.Module): 
    def __init__(self, 
                 in_dim=32,
                 state_feat_dim=256,
                 transf_d_model=256,
                 transf_nhead=8,
                 transf_layers=2,
                 dropout = 0.0,
                 ): 
        super().__init__()
        
        self.TT = TemporalTransformer(
            in_dim=in_dim,
            model_dim=transf_d_model, 
            out_dim=state_feat_dim, 
            num_blocks=transf_layers, 
            num_heads=transf_nhead, 
            dropout=dropout
        )

    def forward(self, v_in):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        latent_action = self.TT(v_in)  
        return latent_action

class TransformerIDMEncoder(nn.Module): 
    def __init__(self, 
                 cfg = None,
                 in_dim=64,
                 ): 
        super().__init__()

        self.idm_encoder = TransformerIDM(cfg, input_dim=in_dim)

    def forward(self, v_in, timesteps=None):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        latent_action = self.idm_encoder(v_in, timesteps)  
        return latent_action
    

class DiTFDMDncoder(nn.Module):
    def __init__(self, 
                 input_dim = 64,        
                 hidden = 256,
                 depth = 6,
                 num_heads = 8,
                 ):
        super().__init__()
        self.fdm_decoder = DiTFeaturePredictor(
            input_dim=input_dim,
            hidden_size=hidden,
            depth=depth,
            num_heads=num_heads,
            timestep_embed_dim=256,
            use_pos_embed=True,
            param_type='x0'       
        )
        
    def forward(self, z, t, vis, act):
        v_pred = self.fdm_decoder(z, t, vis, act)  # 直接输出 v 速度
        return v_pred

class DiTFDMDecoder(nn.Module):
    def __init__(self, 
                 input_dim = 64,        
                 hidden = 128,
                 depth = 2,
                 num_heads = 4,
                 ):
        super().__init__()
        self.fdm_decoder = DiTFeaturePredictor_simple(
            input_dim=input_dim,
            hidden_size=hidden,
            depth=depth,
            num_heads=num_heads,
            timestep_embed_dim=128,
            use_pos_embed=True,
            param_type='x0'       
        )
        
    def forward(self, z, t, vis, act):
        X0 = self.fdm_decoder(z, t, vis, act) 
        return X0

class IDMActionDecoder(nn.Module):
    """Decode latent action (256) -> actual action vector (action_dim) for IDM action supervision."""
    def __init__(self, in_dim=256, action_dim=10):
        super().__init__()
        self.net = create_mlp(in_dim, action_dim, net_arch=[256, 128], activation_fn=nn.ReLU)

    def forward(self, x):
        return self.net(x)

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


    
######################new###########################
# -------------------------
# Time & Condition Embedding
# -------------------------
def sinusoidal_timestep_embedding(timesteps, dim):
    timesteps = timesteps.view(-1).float()
    half = dim // 2
    exponents = torch.arange(half, device=timesteps.device).float() / float(half)
    freqs = 1.0 / (10000 ** exponents)
    args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)

class TimeCondEmbed(nn.Module):
    def __init__(self, time_dim=64, cond_in_dim=0, out_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.proj = nn.Sequential(
            nn.Linear(time_dim + cond_in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, betas, cond):
        t_emb = sinusoidal_timestep_embedding(betas, self.time_dim).to(betas.device)
        if cond is None:
            x = t_emb
        else:
            x = torch.cat([t_emb, cond], dim=-1)
        return self.proj(x)  # (B, out_dim)

# -------------------------
# FiLM residual block (per-point)
# -------------------------
class FiLMResBlock(nn.Module):
    def __init__(self, dim, film_dim, hidden=None, dropout=0.0):
        super().__init__()
        h = hidden or dim
        self.fc1 = nn.Linear(dim, h)
        self.norm1 = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        # film projection: produce scale and bias for this block
        self.film_proj = nn.Linear(film_dim, dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, film_emb):
        """
        x: (B, N, dim)
        film_emb: (B, film_dim)
        """
        B, N, _ = x.shape
        # compute film params
        film = self.film_proj(film_emb)  # (B, 2*dim)
        s, b = film.chunk(2, dim=-1)     # each (B, dim)
        s = s.unsqueeze(1)               # (B,1,dim)
        b = b.unsqueeze(1)

        y = self.fc1(x)                   # (B,N,h)
        y = self.norm1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)                   # (B,N,dim)
        y = self.norm2(y)

        # FiLM apply (1 + scale) * y + bias
        y = y * (1.0 + s) + b
        out = self.act(y + x)             # residual
        return out

# -------------------------
# Induced Set Attention Block (ISAB) - lightweight
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, q, k, v):
        # q: (B, Q, dim), k,v: (B, K, dim)
        B, Q, _ = q.shape
        _, K, _ = k.shape
        q = self.q_proj(q).view(B, Q, self.num_heads, self.head_dim).transpose(1,2)  # (B,heads,Q,hd)
        k = self.k_proj(k).view(B, K, self.num_heads, self.head_dim).transpose(1,2)  # (B,heads,K,hd)
        v = self.v_proj(v).view(B, K, self.num_heads, self.head_dim).transpose(1,2)  # (B,heads,K,hd)

        scores = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)  # (B,heads,Q,K)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,heads,Q,hd)
        out = out.transpose(1,2).contiguous().view(B, Q, self.dim)  # (B,Q,dim)
        return self.out(out)

class ISAB(nn.Module):
    def __init__(self, dim, m=32, num_heads=4):
        """
        Induced Set Attention Block:
        - learnable inducing points I (m)
        - H = Attention(I, X)
        - Out = Attention(X, H)
        Complexity O(N*m)
        """
        super().__init__()
        self.m = m
        self.dim = dim
        self.I = nn.Parameter(torch.randn(1, m, dim))  # learnable inducing points
        self.att1 = MultiHeadAttention(dim, num_heads=num_heads)
        self.att2 = MultiHeadAttention(dim, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.ln3 = nn.LayerNorm(dim)
    def forward(self, x):
        # x: (B,N,dim)
        B, N, D = x.shape
        I = self.I.expand(B, -1, -1)           # (B, m, dim)
        H = self.att1(I, x, x)                 # (B, m, dim)
        H = self.ln1(H + I)
        out = self.att2(x, H, H)               # (B, N, dim)
        out = self.ln2(out + x)
        out = self.ln3(out + self.ff(out))
        return out

# -------------------------
# Lightweight Denoiser
# -------------------------
class LightweightDenoiser(nn.Module):
    def __init__(self, point_dim=3, cond_dims=0, film_dim=128, hidden_dim=64,
                 num_resblocks=3, isab_m=32, isab_heads=4, residual=False, dropout=0.0):
        super().__init__()
        self.point_dim = point_dim
        self.cond_dims = cond_dims
        self.film_dim = film_dim
        self.hidden_dim = hidden_dim
        self.residual = residual

        # initial per-point embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # time+cond embedder
        self.tc_embed = TimeCondEmbed(time_dim=64, cond_in_dim=cond_dims, out_dim=film_dim)

        # stack of FiLM residual blocks
        self.resblocks = nn.ModuleList([
            FiLMResBlock(dim=hidden_dim, film_dim=film_dim, hidden=hidden_dim*2, dropout=dropout)
            for _ in range(num_resblocks)
        ])

        # light ISAB to model cross-point relations with complexity O(N*m)
        self.isab = ISAB(dim=hidden_dim, m=isab_m, num_heads=isab_heads)

        # decoder head
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, point_dim)
        )

    def forward(self, coords, betas, cond):
        """
        coords: (B,N,3)
        betas: (B,) or (B,1)
        cond: (B, F)
        returns: if residual True -> coords + delta, else delta
        """
        B, N, _ = coords.shape
        device = coords.device
        if betas.dim() == 2 and betas.shape[1] == 1:
            betas = betas.view(-1)

        film_emb = self.tc_embed(betas, cond)  # (B, film_dim)
        feat = self.input_mlp(coords)          # (B,N,hidden)
        for blk in self.resblocks:
            feat = blk(feat, film_emb)        # (B,N,hidden)

        isab_out = self.isab(feat)            # (B,N,hidden)
        # fuse local & global: concat feat and isab_out
        fused = torch.cat([feat, isab_out], dim=-1)  # (B,N,2*hidden)
        delta = self.dec(fused)               # (B,N,3)

        if self.residual:
            return coords + delta
        else:
            return delta

# -------------------------
# CPDM lightweight wrapper (uses your VarianceSchedule)
# -------------------------
class CPDM_Light(nn.Module):
    def __init__(self, generator_config, **kwargs):
        super().__init__()
        self.cond_dims = generator_config.cond_dims
        # network hyperparams can be supplied via generator_config or defaults are used
        self.net = LightweightDenoiser(
            point_dim=3,
            cond_dims=self.cond_dims,
            film_dim=getattr(generator_config, "film_dim", 128),
            hidden_dim=getattr(generator_config, "hidden_dim", 64),
            num_resblocks=getattr(generator_config, "num_resblocks", 3),
            isab_m=getattr(generator_config, "isab_m", 32),
            isab_heads=getattr(generator_config, "isab_heads", 4),
            residual=False,
            dropout=getattr(generator_config, "dropout", 0.0)
        )
        self.var_sched = VarianceSchedule(generator_config)
        self.interval_nums = generator_config.interval_nums

    def forward(self, coords, cond, ts=None):
        batch_size, _, point_dim = coords.size()
        if ts is None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0.0
        for i in range(self.interval_nums):
            t = ts[i].tolist()

            alphas_cumprod = self.var_sched.alphas_cumprod[t]
            beta = self.var_sched.betas[t]
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1).to(coords.device)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1).to(coords.device)

            noise = torch.randn_like(coords)
            noisy = sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise

            beta_tensor = torch.tensor(beta, device=coords.device, dtype=torch.float32).view(-1)
            pred = self.net(noisy, betas=beta_tensor, cond=cond)  # returns coords+delta
            # if net returns coords+delta, predicted_noise = pred - coords
            # pred_noise = pred - noisy  # predicted noise on the noisy input
            # but ground-truth noise used to create noisy is `noise`
            loss = F.mse_loss(noise.view(-1, point_dim), pred.view(-1, point_dim), reduction='mean')
            total_loss += loss * (1.0 / self.interval_nums)
        return total_loss

from typing import Optional, Tuple

class RankLoss(nn.Module):
    """
    改进版 RankLoss:
    - 默认使用 CrossEntropyLoss（互斥单标签问题更合适）
    - 在投影后使用 LayerNorm / 可选 L2-norm
    - 支持 batch-shared permutation（shared_perm）与可复现 seed
    - forward 返回 (loss, logits, perms, token_acc)
    """
    def __init__(
        self,
        encoder_output_dim: int,
        N: int = 9,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        loss_type: str = "ce",         # 默认用 CrossEntropy（更稳）
        shared_perm: bool = False,     # batch 内是否共享一个 permutation（便于调试）
        layernorm_after_proj: bool = True,
        l2_normalize_tokens: bool = False,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        assert loss_type in ("bce", "ce"), "loss_type must be 'bce' or 'ce'"
        self.N = N
        self.encoder_output_dim = encoder_output_dim
        self.d_model = d_model
        self.loss_type = loss_type
        self.shared_perm = shared_perm
        self.layernorm_after_proj = layernorm_after_proj
        self.l2_normalize_tokens = l2_normalize_tokens
        self.seed = seed
        self.device = device

        # 投影 encoder 输出到 transformer 的 d_model
        self.input_proj = nn.Linear(encoder_output_dim, d_model)
        if self.layernorm_after_proj:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()

        # 可学习的位置 embedding
        self.pos_embed = nn.Parameter(torch.zeros(N, d_model))

        # Transformer Encoder（保持原结构）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头：d_model -> N (每个 token 对 N 类进行打分)
        self.classifier = nn.Linear(d_model, N)

        # 损失选择
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')

        # 初始化 pos_embed
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _sample_permutations(self, B: int, device: torch.device):
        if self.seed is None:
            if self.shared_perm:
                perm = torch.randperm(self.N, device=device)
                perms = perm.unsqueeze(0).expand(B, -1).contiguous()
            else:
                perms = torch.stack([torch.randperm(self.N, device=device) for _ in range(B)], dim=0)
        else:
            # reproducible using Generator; ensure different perms across batch even with same generator
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            if self.shared_perm:
                perm = torch.randperm(self.N, generator=g, device=device)
                perms = perm.unsqueeze(0).expand(B, -1).contiguous()
            else:
                perms = []
                for i in range(B):
                    # advance seed slightly for each sample to avoid identical perms
                    g.manual_seed(self.seed + i)
                    perms.append(torch.randperm(self.N, generator=g, device=device))
                perms = torch.stack(perms, dim=0)
        return perms.long()

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        """
        x: (B, N, encoder_output_dim)
        return_all True 时返回 (loss, logits, perms, token_acc)
        否则只返回 loss
        """
        assert x.dim() == 3 and x.size(1) == self.N and x.size(2) == self.encoder_output_dim, \
            f"Expected input shape (B, {self.N}, {self.encoder_output_dim}), got {tuple(x.shape)}"

        B = x.size(0)
        device = x.device if self.device is None else self.device

        # 1) 生成 permutation
        perms = self._sample_permutations(B, device=device)  # (B, N)

        # 2) 根据 permutation 取出对应 token
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        shuffled_x = x[batch_idx, perms]  # (B, N, C)

        # 3) 投影到 d_model
        proj = self.input_proj(shuffled_x)  # (B, N, d_model)
        proj = self.norm(proj)
        if self.l2_normalize_tokens:
            proj = F.normalize(proj, p=2, dim=-1)

        # add pos embed (position in shuffled sequence)
        proj = proj + self.pos_embed.unsqueeze(0)

        # 4) Transformer (N, B, E)
        proj_t = proj.permute(1, 0, 2).contiguous()
        trans_out_t = self.transformer(proj_t)
        trans_out = trans_out_t.permute(1, 0, 2).contiguous()

        # 5) 分类头 -> logits (B, N, N)
        logits = self.classifier(trans_out)  # (B, N, N)

        # 6) loss
        if self.loss_type == "bce":
            targets_onehot = F.one_hot(perms, num_classes=self.N).float()
            loss = self.criterion(logits, targets_onehot)
            # preds for accuracy (treat as argmax)
            preds = torch.sigmoid(logits).argmax(dim=-1)
        else:
            # CrossEntropy expects (B*N, N) logits and (B*N,) labels
            loss = self.criterion(logits.view(-1, self.N), perms.view(-1))
            preds = logits.argmax(dim=-1)  # (B, N)

        token_acc = (preds == perms).float().mean().item()

        if return_all:
            return loss, logits, perms, token_acc
        else:
            return loss