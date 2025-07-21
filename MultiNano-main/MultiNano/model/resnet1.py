import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnSum(nn.Module):
    """将二维特征按列求和转换为一维特征"""

    def forward(self, x):
        return x.sum(dim=2)


class Residual1D(nn.Module):
    """一维残差块"""

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        identity = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            identity = self.conv3(identity)

        y += identity
        return F.relu(y)


def resnet_block1d(in_channels, out_channels, num_blocks, first_block=False):
    """构建一维残差块序列"""
    blk = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            blk.append(Residual1D(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual1D(out_channels, out_channels))
    return nn.Sequential(*blk)


class ResNet1D(nn.Module):
    """改进的一维ResNet主干网络"""

    def __init__(self, in_channels=5, out_channels=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            resnet_block1d(32, 32, 2, first_block=True),
            nn.Dropout(0.3),
            resnet_block1d(32, 64, 2),
            resnet_block1d(64, 128, 2),
            nn.Dropout(0.5),
            resnet_block1d(128, out_channels, 2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ColumnSumResNet(nn.Module):
    """端到端的二维转一维特征提取网络"""

    def __init__(self, in_channels=5, feature_dim=256):
        super().__init__()
        self.column_sum = ColumnSum()
        self.resnet1d = ResNet1D(in_channels, feature_dim)

    def forward(self, x):
        # 输入形状: (B, C, H, W)
        x = self.column_sum(x)  # 转换为 (B, C, W)
        return self.resnet1d(x)  # 输出 (B, feature_dim)