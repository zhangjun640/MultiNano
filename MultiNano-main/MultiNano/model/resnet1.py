import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnSum(nn.Module):
    """Sums a 2D feature map along its height (dim=2) to produce a 1D sequence."""

    def forward(self, x):
        # x shape: (B, C, H, W) -> returns (B, C, W)
        return x.sum(dim=2)


class Residual1D(nn.Module):
    """A standard 1D residual block."""

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
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
    """Creates a sequence of 1D residual blocks."""
    blk = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            blk.append(
                Residual1D(in_channels, out_channels, use_1x1conv=True, stride=2)
            )
        else:
            # Use in_channels for the first block in a sequence
            current_in = in_channels if i == 0 else out_channels
            blk.append(Residual1D(current_in, out_channels))
    return nn.Sequential(*blk)


class ResNet1D(nn.Module):
    """An improved 1D ResNet backbone for feature extraction."""

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
            nn.Flatten(),
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ColumnSumResNet(nn.Module):
    """An end-to-end network for 2D to 1D feature extraction."""

    def __init__(self, in_channels=5, feature_dim=256):
        super().__init__()
        self.column_sum = ColumnSum()
        self.resnet1d = ResNet1D(in_channels, feature_dim)

    def forward(self, x):
        # Input shape: (B, C, H, W)
        x = self.column_sum(x)  # Convert to (B, C, W)
        return self.resnet1d(x)  # Output shape: (B, feature_dim)