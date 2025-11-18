import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x))) # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]# [B, 1024, 1]
        x = x.view(-1, 1024)                # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))   # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))   # [B, 256]
        x = self.fc3(x) # [B, 9]

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # [B, 9]
        x = x.view(-1, 3, 3) # [B, 3, 3]
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))   
        x = self.fc3(x) # [B, 64*64]

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k) # [B, 64, 64]
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Use LayerNorm instead of BatchNorm1d
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()

        # 1. Spatial transformer: 对输入的 xyz 坐标对齐
        trans = self.stn(x)     # x: [B, D, N] -> trans: [B, 3, 3]
        x = x.transpose(2, 1)   # [B, N, D]
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans) # [B, N, 3]
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)   # [B, D, N]

        # Apply Conv1d + LayerNorm + ReLU
        x = self.conv1(x)  # [B, 64, N]
        x = x.transpose(1, 2)  # [B, N, 64] for LayerNorm
        x = F.relu(self.ln1(x))  # LayerNorm + ReLU
        x = x.transpose(1, 2)  # [B, 64, N]

        if self.feature_transform:
            trans_feat = self.fstn(x)   # [B, 64, 64]
            x = x.transpose(2, 1)       # [B, N, 64]
            x = torch.bmm(x, trans_feat)# [B, N, 64]
            x = x.transpose(2, 1)       # [B, 64, N]
        else:
            trans_feat = None

        pointfeat = x # [B, 64, N]
        
        # Second layer
        x = self.conv2(x)  # [B, 128, N]
        x = x.transpose(1, 2)  # [B, N, 128] for LayerNorm
        x = F.relu(self.ln2(x))  # LayerNorm + ReLU
        x = x.transpose(1, 2)  # [B, 128, N]
        
        # Third layer
        x = self.conv3(x)  # [B, 1024, N]
        x = x.transpose(1, 2)  # [B, N, 1024] for LayerNorm
        x = self.ln3(x)  # LayerNorm (no ReLU here)
        x = x.transpose(1, 2)  # [B, 1024, N]
        
        x = torch.max(x, 2, keepdim=True)[0]# [B, 1024, 1]
        x = x.view(-1, 1024)                # [B, 1024]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # [B, 1024, N]
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1] # 64
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
