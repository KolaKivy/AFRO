
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from easydict import EasyDict
import sys
import os
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_lns = nn.ModuleList()  
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_lns.append(nn.LayerNorm(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            ln = self.mlp_lns[i]
            new_points = conv(new_points)  
            
            B, C, nsample, npoint = new_points.shape
            new_points = new_points.permute(0, 2, 3, 1)  
            new_points = ln(new_points)  
            new_points = new_points.permute(0, 3, 1, 2)  
            new_points = F.relu(new_points)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetPlusPlusEncoder(nn.Module):
    """PointNet++ Encoder for Point Cloud Classification"""
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.final_norm = final_norm
        self.use_projection = use_projection
        self.in_channels = in_channels
        
        cprint(f"[PointNetPlusPlusEncoder] in_channels: {in_channels}", 'cyan')
        cprint(f"[PointNetPlusPlusEncoder] use_layernorm: {use_layernorm}", 'cyan')
        cprint(f"[PointNetPlusPlusEncoder] final_norm: {final_norm}", 'cyan')
        
        # Set Abstraction layers - handle variable input channels
        # For PointNet++, we always use 3D coordinates for spatial operations
        # Additional features (like RGB) are handled separately
        coord_dim = 3  # Always use XYZ coordinates for spatial operations
        feature_dim = max(0, in_channels - 3)  # Additional features beyond XYZ
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                         in_channel=coord_dim + feature_dim, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                         in_channel=128 + coord_dim, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                         in_channel=256 + coord_dim, mlp=[256, 256, 512], group_all=True)
        
        # Final projection layer
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(512, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(512, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
            
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetPlusPlusEncoder] not use projection", "yellow")

    def forward(self, xyz):
        """
        Input:
            xyz: input points data, [B, N, C] where C can be 3 (XYZ) or 6 (XYZRGB)
        Return:
            features: global feature vector, [B, out_channels]
        """
        B, N, C = xyz.shape
        
        # For PointNet++, we need xyz coordinates and optional features
        # Always use first 3 channels as coordinates for spatial operations
        xyz_coords = xyz[:, :, :3].permute(0, 2, 1)  # [B, 3, N]
        
        if C > 3:
            # Additional features beyond XYZ (e.g., RGB)
            points = xyz[:, :, 3:].permute(0, 2, 1)  # [B, C-3, N]
        else:
            points = None
        
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz_coords, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global feature
        x = l3_points.view(B, 512)
        x = self.final_projection(x)
        
        return x




















# from pyexpat import model
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import copy
# from easydict import EasyDict
# import sys
# import os
# from .uni3d.uni3d import create_uni3d
# from typing import Optional, Dict, Tuple, Union, List, Type
# from termcolor import cprint

# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist

# def index_points(points, idx):
#     """
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points

# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, C]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest].view(B, 1, C)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids

# def query_ball_point(radius, nsample, xyz, new_xyz):
#     """
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
#     sqrdists = square_distance(new_xyz, xyz)
#     group_idx[sqrdists > radius ** 2] = N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#     return group_idx

# def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
#     """
#     Input:
#         npoint:
#         radius:
#         nsample:
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, nsample, 3]
#         new_points: sampled points data, [B, npoint, nsample, 3+D]
#     """
#     B, N, C = xyz.shape
#     S = npoint
    
#     fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
#     new_xyz = index_points(xyz, fps_idx)
#     idx = query_ball_point(radius, nsample, xyz, new_xyz)
#     grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
#     grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

#     if points is not None:
#         grouped_points = index_points(points, idx)
#         new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
#     else:
#         new_points = grouped_xyz_norm
#     if returnfps:
#         return new_xyz, new_points, grouped_xyz, fps_idx
#     else:
#         return new_xyz, new_points

# def sample_and_group_all(xyz, points):
#     """
#     Input:
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, 1, 3]
#         new_points: sampled points data, [B, 1, N, 3+D]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     new_xyz = torch.zeros(B, 1, C).to(device)
#     grouped_xyz = xyz.view(B, 1, N, C)
#     if points is not None:
#         new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
#     else:
#         new_points = grouped_xyz
#     return new_xyz, new_points

# class PointNetSetAbstraction(nn.Module):
#     def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
#         super(PointNetSetAbstraction, self).__init__()
#         self.npoint = npoint
#         self.radius = radius
#         self.nsample = nsample
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
#             last_channel = out_channel
#         self.group_all = group_all

#     def forward(self, xyz, points):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         """
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)

#         if self.group_all:
#             new_xyz, new_points = sample_and_group_all(xyz, points)
#         else:
#             new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
#         # new_xyz: sampled points position data, [B, npoint, C]
#         # new_points: sampled points data, [B, npoint, nsample, C+D]
#         new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points =  F.relu(bn(conv(new_points)))

#         new_points = torch.max(new_points, 2)[0]
#         new_xyz = new_xyz.permute(0, 2, 1)
#         return new_xyz, new_points

# class PointNetPlusPlusEncoder(nn.Module):
#     """PointNet++ Encoder for Point Cloud Classification"""
#     def __init__(self,
#                  in_channels: int = 3,
#                  out_channels: int = 1024,
#                  use_layernorm: bool = False,
#                  final_norm: str = 'none',
#                  use_projection: bool = True,
#                  **kwargs
#                  ):
#         super().__init__()
#         self.use_layernorm = use_layernorm
#         self.final_norm = final_norm
#         self.use_projection = use_projection
#         self.in_channels = in_channels
        
#         cprint(f"[PointNetPlusPlusEncoder] in_channels: {in_channels}", 'cyan')
#         cprint(f"[PointNetPlusPlusEncoder] use_layernorm: {use_layernorm}", 'cyan')
#         cprint(f"[PointNetPlusPlusEncoder] final_norm: {final_norm}", 'cyan')
        
#         # Set Abstraction layers - handle variable input channels
#         # For PointNet++, we always use 3D coordinates for spatial operations
#         # Additional features (like RGB) are handled separately
#         coord_dim = 3  # Always use XYZ coordinates for spatial operations
#         feature_dim = max(0, in_channels - 3)  # Additional features beyond XYZ
        
#         self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
#                                          in_channel=coord_dim + feature_dim, mlp=[64, 64, 128], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
#                                          in_channel=128 + coord_dim, mlp=[128, 128, 256], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
#                                          in_channel=256 + coord_dim, mlp=[256, 256, 512], group_all=True)
        
#         # Final projection layer
#         if final_norm == 'layernorm':
#             self.final_projection = nn.Sequential(
#                 nn.Linear(512, out_channels),
#                 nn.LayerNorm(out_channels)
#             )
#         elif final_norm == 'none':
#             self.final_projection = nn.Linear(512, out_channels)
#         else:
#             raise NotImplementedError(f"final_norm: {final_norm}")
            
#         if not use_projection:
#             self.final_projection = nn.Identity()
#             cprint("[PointNetPlusPlusEncoder] not use projection", "yellow")

#     def forward(self, xyz):
#         """
#         Input:
#             xyz: input points data, [B, N, C] where C can be 3 (XYZ) or 6 (XYZRGB)
#         Return:
#             features: global feature vector, [B, out_channels]
#         """
#         B, N, C = xyz.shape
        
#         # For PointNet++, we need xyz coordinates and optional features
#         # Always use first 3 channels as coordinates for spatial operations
#         xyz_coords = xyz[:, :, :3].permute(0, 2, 1)  # [B, 3, N]
        
#         if C > 3:
#             # Additional features beyond XYZ (e.g., RGB)
#             points = xyz[:, :, 3:].permute(0, 2, 1)  # [B, C-3, N]
#         else:
#             points = None
        
#         # Set Abstraction layers
#         l1_xyz, l1_points = self.sa1(xyz_coords, points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
#         # Global feature
#         x = l3_points.view(B, 512)
#         x = self.final_projection(x)
        
#         return x