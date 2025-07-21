import torch
from torch import nn
import torch.nn.functional as F
from MultiNano.model.util import FlattenLayer


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            center = kernel_size // 7
            self.conv.weight[:, :, :center + 1, :center + 1].normal_(0, 0.01)
            self.conv.weight[:, :, center:, center:].zero_()
            self.conv.bias.zero_()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * x


class EnhancedSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(torch.cat([self.avg_pool(x).view(b, c),
                                      self.max_pool(x).view(b, c)], dim=1)).view(b, c, 1, 1)


class EnhancedResidual2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.se = EnhancedSEBlock(out_channels)
        self.sa = SpatialAttention(5)
        self.pos_conv = nn.Conv2d(out_channels, out_channels, 3,
                                  padding=1, groups=out_channels)
        nn.init.constant_(self.pos_conv.weight[:, :, 2:, 2:], 0)
        nn.init.constant_(self.pos_conv.bias, 0)

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.se(x) + self.sa(x)  # 并行注意力机制
        return F.relu(self.pos_conv(x) + identity)


class OptimizedResNet2D(nn.Module):
    def __init__(self, in_channels=5, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            # 初始特征提取
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),

            # 第一阶段（保持分辨率）
            #self._make_layer(32, 32, num_blocks=2, stride=1),

            # 下采样阶段
            self._make_layer(32, 64, num_blocks=1, stride=2),
            self._make_layer(64, 128, num_blocks=2, stride=2),
            self._make_layer(128, out_channels, num_blocks=2, stride=2),  # 最终输出通道256

            # 自适应特征处理
            nn.AdaptiveAvgPool2d(1),
            FlattenLayer()
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride,dropout=0.3):
        layers = [EnhancedResidual2D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(EnhancedResidual2D(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

