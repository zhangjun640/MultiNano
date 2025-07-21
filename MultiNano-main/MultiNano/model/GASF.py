import torch
import torch.nn as nn


class BaseTime2Img(nn.Module):
    def __init__(self, output_size=65):
        super().__init__()
        self.output_size = output_size
        self.eps = 1e-6

    def get_valid_region(self, x):
        """双指针法定位有效数据区域（兼容低版本PyTorch）"""
        N, C, L = x.shape
        valid_mask = torch.zeros(N, C, L, dtype=torch.bool, device=x.device)

        for n in range(N):
            for c in range(C):
                # 寻找首尾非零索引
                non_zero = torch.nonzero(x[n, c]).view(-1)
                if len(non_zero) == 0:
                    continue
                start = non_zero[0].item()
                end = non_zero[-1].item()
                valid_mask[n, c, start:end + 1] = True
        return valid_mask

    def resize_matrix(self, matrix):
        """统一尺寸调整方法"""
        N, C, H, W = matrix.shape
        output = torch.zeros(N, C, self.output_size, self.output_size,
                             device=matrix.device)
        size = min(H, self.output_size)
        output[:, :, :size, :size] = matrix[:, :, :size, :size]
        return output


class GASF(BaseTime2Img):
    def forward(self, x):
        valid_mask = self.get_valid_region(x)
        N, C, L = x.shape

        # 兼容低版本的极值计算
        x_masked = x.masked_fill(~valid_mask, float('inf'))
        x_min = x_masked.min(dim=2, keepdim=True)[0].clamp(max=0)  # 防止inf干扰
        x_masked = x.masked_fill(~valid_mask, -float('inf'))
        x_max = x_masked.max(dim=2, keepdim=True)[0].clamp(min=0)

        x_range = (x_max - x_min).clamp(min=self.eps)
        x_norm = 2 * (x - x_min) / x_range - 1
        x_norm = torch.where(valid_mask, x_norm, 0.0)

        # 相位计算
        phi = torch.acos(x_norm.clamp(-1 + self.eps, 1 - self.eps))
        gasf = torch.cos(phi.unsqueeze(3) + phi.unsqueeze(2))
        return self.resize_matrix(gasf)


class GADF(BaseTime2Img):
    def forward(self, x):
        valid_mask = self.get_valid_region(x)
        N, C, L = x.shape

        # 极值计算兼容方案
        x_masked = x.masked_fill(~valid_mask, float('inf'))
        x_min = x_masked.min(dim=2, keepdim=True)[0].clamp(max=0)
        x_masked = x.masked_fill(~valid_mask, -float('inf'))
        x_max = x_masked.max(dim=2, keepdim=True)[0].clamp(min=0)

        x_range = (x_max - x_min).clamp(min=self.eps)
        x_norm = 2 * (x - x_min) / x_range - 1
        x_norm = torch.where(valid_mask, x_norm, 0.0)

        # 相位差计算
        phi = torch.acos(x_norm.clamp(-1 + self.eps, 1 - self.eps))
        gadf = torch.sin(phi.unsqueeze(3) - phi.unsqueeze(2))
        return self.resize_matrix(gadf)


class MTF(BaseTime2Img):
    def __init__(self, output_size=65, n_bins=65):
        super().__init__(output_size)
        self.n_bins = n_bins

    def forward(self, x):
        valid_mask = self.get_valid_region(x)
        N, C, L = x.shape

        # 分箱计算 (兼容多通道)
        quantiles = []
        for n in range(N):
            channel_quantiles = []
            for c in range(C):
                valid_data = x[n, c][valid_mask[n, c]]
                if len(valid_data) == 0:
                    q = torch.zeros(self.n_bins + 1, device=x.device)
                else:
                    q = torch.quantile(
                        valid_data,
                        torch.linspace(0, 1, self.n_bins + 1, device=x.device)
                    )
                channel_quantiles.append(q)
            quantiles.append(torch.stack(channel_quantiles))  # [C, n_bins+1]
        quantiles = torch.stack(quantiles)  # [N, C, n_bins+1]

        # 构建多通道转移矩阵
        mtf = torch.zeros(N, C, self.n_bins, self.n_bins, device=x.device)

        # 生成所有时间步的索引
        for t in range(L - 1):
            current = x[:, :, t]  # [N, C]
            next_val = x[:, :, t + 1]  # [N, C]

            # 计算bin索引 (保持通道独立性)
            current_bin = torch.sum(current.unsqueeze(-1) >= quantiles[:, :, :-1], dim=3)  # [N, C]
            next_bin = torch.sum(next_val.unsqueeze(-1) >= quantiles[:, :, :-1], dim=3)  # [N, C]

            # 转换为合法索引 (0 <= index < n_bins)
            current_bin = torch.clamp(current_bin, 0, self.n_bins - 1).long()  # [N, C]
            next_bin = torch.clamp(next_bin, 0, self.n_bins - 1).long()  # [N, C]

            # 生成批量索引
            batch_idx = torch.arange(N, device=x.device)[:, None].expand(-1, C)  # [N, C]
            channel_idx = torch.arange(C, device=x.device)[None, :].expand(N, -1)  # [N, C]

            # 更新转移矩阵 (向量化操作)
            mtf[batch_idx, channel_idx, current_bin, next_bin] += 1

        mtf = mtf / (L - 1)  # 归一化
        return self.resize_matrix(mtf)

class RP(BaseTime2Img):
    def __init__(self, output_size=65, threshold=0.2):
        super().__init__(output_size)
        self.threshold = threshold

    def forward(self, x):
        valid_mask = self.get_valid_region(x)
        N, C, L = x.shape  # 输入x的形状为 [batch_size, channels, seq_len]

        # 标准化处理（保持通道独立性）
        x_masked = x * valid_mask.float()
        mean = x_masked.sum(dim=2) / valid_mask.sum(dim=2).clamp(min=1)  # [N, C]
        std = torch.sqrt(
            ((x_masked - mean.unsqueeze(2)) ** 2 * valid_mask.float()).sum(dim=2) /
            valid_mask.sum(dim=2).clamp(min=1)
        )  # [N, C]
        x_norm = (x - mean.unsqueeze(2)) / std.unsqueeze(2).clamp(min=self.eps)  # [N, C, L]
        x_norm = torch.where(valid_mask, x_norm, 0.0)

        # 为每个通道独立计算递归图
        rp_list = []
        for c in range(C):
            # 取当前通道的数据: [N, L] -> [N, L, 1]
            x_channel = x_norm[:, c, :].unsqueeze(-1)  # [N, L, 1]

            # 计算递归图（时间步间的距离）
            distance_matrix = torch.cdist(x_channel, x_channel)  # [N, L, L]
            rp = (distance_matrix < self.threshold).float()  # [N, L, L]

            rp_list.append(rp.unsqueeze(1))  # 添加通道维度 -> [N, 1, L, L]

        # 合并通道: [N, C, L, L]
        rp = torch.cat(rp_list, dim=1)

        return self.resize_matrix(rp)