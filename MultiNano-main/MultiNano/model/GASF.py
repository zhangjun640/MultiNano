import torch
import torch.nn as nn


class BaseTime2Img(nn.Module):
    def __init__(self, output_size=65):
        super().__init__()
        self.output_size = output_size
        self.eps = 1e-6

    def get_valid_region(self, x):
        """Locate valid data region using two-pointer method (compatible with older PyTorch versions)"""
        N, C, L = x.shape
        valid_mask = torch.zeros(N, C, L, dtype=torch.bool, device=x.device)

        for n in range(N):
            for c in range(C):
                # Find the first and last non-zero index
                non_zero = torch.nonzero(x[n, c]).view(-1)
                if len(non_zero) == 0:
                    continue
                start = non_zero[0].item()
                end = non_zero[-1].item()
                valid_mask[n, c, start:end + 1] = True
        return valid_mask

    def resize_matrix(self, matrix):
        """Unified resizing method"""
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

        # Extremum calculation compatible with old versions
        x_masked = x.masked_fill(~valid_mask, float('inf'))
        x_min = x_masked.min(dim=2, keepdim=True)[0].clamp(max=0)  # Avoid interference from inf
        x_masked = x.masked_fill(~valid_mask, -float('inf'))
        x_max = x_masked.max(dim=2, keepdim=True)[0].clamp(min=0)

        x_range = (x_max - x_min).clamp(min=self.eps)
        x_norm = 2 * (x - x_min) / x_range - 1
        x_norm = torch.where(valid_mask, x_norm, 0.0)

        # Phase calculation
        phi = torch.acos(x_norm.clamp(-1 + self.eps, 1 - self.eps))
        gasf = torch.cos(phi.unsqueeze(3) + phi.unsqueeze(2))
        return self.resize_matrix(gasf)


class GADF(BaseTime2Img):
    def forward(self, x):
        valid_mask = self.get_valid_region(x)
        N, C, L = x.shape

        # Extremum calculation compatibility
        x_masked = x.masked_fill(~valid_mask, float('inf'))
        x_min = x_masked.min(dim=2, keepdim=True)[0].clamp(max=0)
        x_masked = x.masked_fill(~valid_mask, -float('inf'))
        x_max = x_masked.max(dim=2, keepdim=True)[0].clamp(min=0)

        x_range = (x_max - x_min).clamp(min=self.eps)
        x_norm = 2 * (x - x_min) / x_range - 1
        x_norm = torch.where(valid_mask, x_norm, 0.0)

        # Phase difference calculation
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

        # Binning (compatible with multi-channel)
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

        # Construct multi-channel transition matrix
        mtf = torch.zeros(N, C, self.n_bins, self.n_bins, device=x.device)

        # Generate index for all time steps
        for t in range(L - 1):
            current = x[:, :, t]  # [N, C]
            next_val = x[:, :, t + 1]  # [N, C]

            # Compute bin index (preserve channel independence)
            current_bin = torch.sum(current.unsqueeze(-1) >= quantiles[:, :, :-1], dim=3)  # [N, C]
            next_bin = torch.sum(next_val.unsqueeze(-1) >= quantiles[:, :, :-1], dim=3)  # [N, C]

            # Convert to valid index (0 <= index < n_bins)
            current_bin = torch.clamp(current_bin, 0, self.n_bins - 1).long()  # [N, C]
            next_bin = torch.clamp(next_bin, 0, self.n_bins - 1).long()  # [N, C]

            # Generate batch indices
            batch_idx = torch.arange(N, device=x.device)[:, None].expand(-1, C)  # [N, C]
            channel_idx = torch.arange(C, device=x.device)[None, :].expand(N, -1)  # [N, C]

            # Update transition matrix (vectorized operation)
            mtf[batch_idx, channel_idx, current_bin, next_bin] += 1

        mtf = mtf / (L - 1)  # Normalize
        return self.resize_matrix(mtf)


class RP(BaseTime2Img):
    def __init__(self, output_size=65, threshold=0.2):
        super().__init__(output_size)
        self.threshold = threshold

    def forward(self, x):
        valid_mask = self.get_valid_region(x)
        N, C, L = x.shape  # Input x shape: [batch_size, channels, seq_len]

        # Standardization (preserve channel independence)
        x_masked = x * valid_mask.float()
        mean = x_masked.sum(dim=2) / valid_mask.sum(dim=2).clamp(min=1)  # [N, C]
        std = torch.sqrt(
            ((x_masked - mean.unsqueeze(2)) ** 2 * valid_mask.float()).sum(dim=2) /
            valid_mask.sum(dim=2).clamp(min=1)
        )  # [N, C]
        x_norm = (x - mean.unsqueeze(2)) / std.unsqueeze(2).clamp(min=self.eps)  # [N, C, L]
        x_norm = torch.where(valid_mask, x_norm, 0.0)

        # Compute recurrence plots independently for each channel
        rp_list = []
        for c in range(C):
            # Take current channel data: [N, L] -> [N, L, 1]
            x_channel = x_norm[:, c, :].unsqueeze(-1)  # [N, L, 1]

            # Compute recurrence plot (distance between time steps)
            distance_matrix = torch.cdist(x_channel, x_channel)  # [N, L, L]
            rp = (distance_matrix < self.threshold).float()  # [N, L, L]

            rp_list.append(rp.unsqueeze(1))  # Add channel dimension -> [N, 1, L, L]

        # Concatenate channels: [N, C, L, L]
        rp = torch.cat(rp_list, dim=1)

        return self.resize_matrix(rp)
