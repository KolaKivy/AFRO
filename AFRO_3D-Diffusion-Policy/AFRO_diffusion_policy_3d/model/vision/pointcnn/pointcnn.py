import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint):
    """
    FPS sampling using pure PyTorch
    xyz: [B, N, 3]
    npoint: number of points to sample
    Returns: [B, npoint] indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def knn_search(support_xyz, query_xyz, k):
    """
    KNN search using pure PyTorch
    support_xyz: [B, N, 3]
    query_xyz: [B, M, 3]
    k: number of neighbors
    Returns: distances [B, M, k], indices [B, M, k]
    """
    B, N, _ = support_xyz.shape
    _, M, _ = query_xyz.shape
    
    # Compute pairwise distances
    support_xyz = support_xyz.unsqueeze(2)  # [B, N, 1, 3]
    query_xyz = query_xyz.unsqueeze(1)      # [B, 1, M, 3]
    
    dist = torch.sum((support_xyz - query_xyz) ** 2, dim=-1)  # [B, N, M]
    dist = dist.transpose(1, 2)  # [B, M, N]
    
    # Get k nearest neighbors
    distances, indices = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    
    return distances, indices

def gather_points(points, idx):
    """
    Gather points according to indices
    points: [B, N, C]
    idx: [B, M] or [B, M, K]
    Returns: [B, M, C] or [B, M, K, C]
    """
    B, N, C = points.shape
    if len(idx.shape) == 2:
        M = idx.shape[1]
        idx = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, M, C]
        return torch.gather(points, 1, idx)  # [B, M, C]
    else:
        M, K = idx.shape[1], idx.shape[2]
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, M, K, C]
        points = points.unsqueeze(2).expand(-1, -1, K, -1)  # [B, N, K, C]
        return torch.gather(points, 1, idx)  # [B, M, K, C]

class XConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=16, dilation=1):
        super(XConv, self).__init__()
        self.K = K  # 邻域点数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        # X 变换 MLP：输入 [K, 3]（相对坐标），输出 [K, K]
        self.x_transform = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, K * K)
        )

        # 卷积 MLP：输入 [K, in_channels]，输出 [K, out_channels]
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # LayerNorm 归一化
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, points, features, N_ratio ,idx=None):
        """
        points: [B, N, 3] - 点云坐标
        features: [B, N, in_channels] - 点云特征
        idx: [B, N', K] - 可选，KNN 索引
        Returns: [B, N', out_channels] - 输出特征
        """
        B, N, _ = points.shape
        N_out = N // N_ratio  # 下采样点数（FPS）

        # FPS 下采样
        sampled_idx = farthest_point_sample(points, N_out)  # [B, N']
        sampled_points = gather_points(points, sampled_idx)  # [B, N', 3]
        sampled_features = gather_points(features, sampled_idx)  # [B, N', in_channels]

        # KNN 分组
        if idx is None:
            _, idx = knn_search(points, sampled_points, self.K * self.dilation)  # [B, N', K]
            idx = idx[:, :, :self.K]  # 取前 K 个邻域点
        else:
            # 如果传入了idx，需要确保它与当前的sampled_points匹配
            # 这里我们重新计算KNN以确保一致性
            _, idx = knn_search(points, sampled_points, self.K * self.dilation)  # [B, N', K]
            idx = idx[:, :, :self.K]  # 取前 K 个邻域点

        # 提取邻域
        neighbor_points = gather_points(points, idx)  # [B, N', K, 3]
        neighbor_features = gather_points(features, idx)  # [B, N', K, in_channels]

        # 相对坐标
        neighbor_points = neighbor_points - sampled_points.unsqueeze(2)  # [B, N', K, 3]

        # X 变换
        B_curr, N_curr, K_curr, _ = neighbor_points.shape
        neighbor_points_flat = neighbor_points.view(-1, 3)  # [B*N'*K, 3]
        x_matrix_flat = self.x_transform(neighbor_points_flat)  # [B*N'*K, K*K]
        x_matrix = x_matrix_flat.view(B_curr, N_curr, K_curr, self.K * self.K)  # [B, N', K, K*K]
        
        # 对每个中心点，我们需要一个 K x K 的变换矩阵
        # 这里我们对 K 个邻域点的变换矩阵进行平均
        x_matrix = x_matrix.mean(dim=2)  # [B, N', K*K]
        x_matrix = x_matrix.view(B_curr, N_curr, self.K, self.K)  # [B, N', K, K]
        
        # 投影特征
        neighbor_features = neighbor_features.permute(0, 1, 3, 2)  # [B, N', in_channels, K]
        transformed_features = torch.matmul(neighbor_features, x_matrix)  # [B, N', in_channels, K]
        transformed_features = transformed_features.permute(0, 1, 3, 2)  # [B, N', K, in_channels]

        # 卷积
        output_features = self.conv(transformed_features)  # [B, N', K, out_channels]
        output_features = output_features.max(dim=2)[0]  # [B, N', out_channels]（max pooling）

        # LayerNorm
        output_features = self.ln(output_features)

        return sampled_points, output_features, idx

class PointCNN(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 128,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        super(PointCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        block_channel = [64, 128, 256, 512,1024]
        self.N_ratio = [2,2,2]


        # X-Conv 层
        self.xconv1 = XConv(in_channels=in_channels, out_channels=block_channel[0], K=16)
        self.xconv2 = XConv(in_channels=block_channel[0], out_channels=block_channel[1], K=16)
        self.xconv3 = XConv(in_channels=block_channel[1], out_channels=block_channel[2], K=8)

        # 全局池化后全连接层
        self.fc = nn.Sequential(
            nn.Linear(block_channel[2], block_channel[3]),
            nn.ReLU(),
            nn.LayerNorm(block_channel[3]),
            nn.Linear(block_channel[3], block_channel[4])
        )
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[4], out_channels),
                nn.LayerNorm(out_channels)
            )
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        """
        x: [B, N, C] - 输入点云（坐标+特征）
        Returns: [B, 1024] - 全局特征
        """
        points = x[:, :, :3]  # [B, N, 3]（坐标）
        features = x  # [B, N, C]（特征）

        # X-Conv 层次
        points, features, idx = self.xconv1(points, features ,self.N_ratio[0])  # [B, N/2, 64]
        points, features, idx = self.xconv2(points, features ,self.N_ratio[1])  # [B, N/4, 128]
        points, features, _ = self.xconv3(points, features ,self.N_ratio[2])  # [B, N/8, 256]

        # 全局 max pooling
        global_features = features.max(dim=1)[0]  # [B, 256]

        # 全连接层
        output = self.fc(global_features)  # [B, 1024]
        output = self.final_projection(output)

        return output
