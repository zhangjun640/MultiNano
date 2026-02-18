import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTime2Img(nn.Module):
    def __init__(self, output_size=65):
        super().__init__()
        self.output_size = output_size
        self.eps = 1e-6

    def get_valid_mask(self, x):
        """
        Vectorized version of locating valid data region.
        Returns a boolean mask of shape [N, C, L] where valid regions are True.
        """
        N, C, L = x.shape
        # x shape: [N, C, L]

        # 1. Identify non-zero elements
        non_zero_mask = (x != 0)  # [N, C, L]

        # 2. Find start index (first non-zero)
        # argmax returns the first index of the maximum value (True).
        # If all are False (zero row), it returns 0.
        starts = torch.argmax(non_zero_mask.int(), dim=2)  # [N, C]

        # 3. Find end index (last non-zero)
        # Flip to find the first non-zero from the back, then subtract from length
        ends = L - 1 - torch.argmax(non_zero_mask.flip(dims=[2]).int(), dim=2)  # [N, C]

        # 4. Handle cases where the whole row is zero (start=0, end=L-1 logic above might fail edge case)
        # Check if there is at least one non-zero
        has_data = non_zero_mask.any(dim=2)  # [N, C]

        # 5. Create range tensor to compare with starts and ends
        # range_tensor: [1, 1, L] -> broadcastable to [N, C, L]
        range_tensor = torch.arange(L, device=x.device).view(1, 1, -1)

        # mask: index >= start AND index <= end AND has_data
        valid_mask = (range_tensor >= starts.unsqueeze(2)) & \
                     (range_tensor <= ends.unsqueeze(2)) & \
                     has_data.unsqueeze(2)

        return valid_mask

    def resize_matrix(self, matrix):
        """
        Optimized resize using interpolation for better GPU utilization
        or slicing if strictly cropping logic is desired.
        Original logic was cropping top-left. Keeping it for consistency.
        """
        # matrix: [N, C, H, W]
        if matrix.shape[-1] == self.output_size and matrix.shape[-2] == self.output_size:
            return matrix

        # Using slicing as per original logic (top-left crop)
        # Note: If memory is still tight, consider F.interpolate instead of calculating full matrix then cropping.
        return matrix[..., :self.output_size, :self.output_size]


class GASF(BaseTime2Img):
    def forward(self, x):
        # x: [N, C, L]
        valid_mask = self.get_valid_mask(x)  # [N, C, L]

        # Vectorized Min/Max calculation
        # Replace invalid values with inf/-inf to ignore them in min/max
        x_masked_min = x.masked_fill(~valid_mask, float('inf'))
        x_min = x_masked_min.min(dim=2, keepdim=True)[0]  # [N, C, 1]
        x_min = x_min.masked_fill(x_min == float('inf'), 0.0).clamp(max=0)  # Handle all-zero rows

        x_masked_max = x.masked_fill(~valid_mask, -float('inf'))
        x_max = x_masked_max.max(dim=2, keepdim=True)[0]  # [N, C, 1]
        x_max = x_max.masked_fill(x_max == -float('inf'), 0.0).clamp(min=0)

        x_range = (x_max - x_min).clamp(min=self.eps)

        # Normalize
        x_norm = 2 * (x - x_min) / x_range - 1

        # Zero out invalid regions (padding)
        x_norm = torch.where(valid_mask, x_norm, torch.tensor(0.0, device=x.device))

        # Phase calculation
        # clamp to avoid numerical errors slightly outside [-1, 1]
        phi = torch.acos(x_norm.clamp(-1 + self.eps, 1 - self.eps))

        # GASF: cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
        # Using addition theorem is often more memory efficient than creating the sum matrix phi_i + phi_j
        # BUT standard formula is just broadcasting sum.
        # Shape: [N, C, L, 1] + [N, C, 1, L] -> [N, C, L, L]
        gasf = torch.cos(phi.unsqueeze(-1) + phi.unsqueeze(-2))

        return self.resize_matrix(gasf)


class GADF(BaseTime2Img):
    def forward(self, x):
        valid_mask = self.get_valid_mask(x)

        # Same normalization logic as GASF
        x_masked_min = x.masked_fill(~valid_mask, float('inf'))
        x_min = x_masked_min.min(dim=2, keepdim=True)[0].masked_fill(
            x_masked_min.min(dim=2, keepdim=True)[0] == float('inf'), 0.0).clamp(max=0)

        x_masked_max = x.masked_fill(~valid_mask, -float('inf'))
        x_max = x_masked_max.max(dim=2, keepdim=True)[0].masked_fill(
            x_masked_max.max(dim=2, keepdim=True)[0] == -float('inf'), 0.0).clamp(min=0)

        x_range = (x_max - x_min).clamp(min=self.eps)
        x_norm = 2 * (x - x_min) / x_range - 1
        x_norm = torch.where(valid_mask, x_norm, torch.tensor(0.0, device=x.device))

        phi = torch.acos(x_norm.clamp(-1 + self.eps, 1 - self.eps))

        # GADF: sin(phi_i - phi_j)
        gadf = torch.sin(phi.unsqueeze(-1) - phi.unsqueeze(-2))

        return self.resize_matrix(gadf)


class MTF(BaseTime2Img):
    def __init__(self, output_size=65, n_bins=65):
        super().__init__(output_size)
        self.n_bins = n_bins

    def forward(self, x):
        """
        Highly optimized MTF calculation without loops.
        Uses rank-based binning (equivalent to quantile binning) and scatter_add.
        """
        valid_mask = self.get_valid_mask(x)  # [N, C, L]
        N, C, L = x.shape
        device = x.device

        # --- 1. Vectorized Quantile Binning ---
        # Instead of calculating quantile values and then binning,
        # we can use argsort to get ranks, which is equivalent to equal-depth binning.

        # We need to handle padding (invalid data).
        # Strategy: Set invalid data to infinity so they sort to the end, then clamp bins.
        x_for_sort = x.clone()
        x_for_sort[~valid_mask] = float('inf')

        # Get ranks: [N, C, L]
        # argsort twice gives the rank of each element
        ranks = torch.argsort(torch.argsort(x_for_sort, dim=-1), dim=-1)

        # Calculate valid lengths per channel
        valid_lengths = valid_mask.sum(dim=-1, keepdim=True)  # [N, C, 1]
        valid_lengths = valid_lengths.clamp(min=1)  # Avoid division by zero

        # Convert ranks to bins: floor(rank / length * n_bins)
        # Elements in invalid region will have high ranks, resulting in bins >= n_bins
        bins = (ranks.float() / valid_lengths.float() * self.n_bins).long()

        # Clamp bins to range [0, n_bins-1] and mask out invalid ones
        bins = bins.clamp(0, self.n_bins - 1)
        bins = bins * valid_mask.long()  # Invalid regions become bin 0 (doesn't matter, we won't count them)

        # --- 2. Vectorized Transition Matrix Construction ---

        # Get Current and Next bins
        current_bins = bins[:, :, :-1]  # [N, C, L-1]
        next_bins = bins[:, :, 1:]  # [N, C, L-1]

        # Mask for valid transitions (both t and t+1 must be valid)
        valid_trans = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]

        # Flatten batches and channels for counting
        # We want to count transitions (i -> j) for each (n, c) pair independently.
        # We can create a flattened index:
        # global_idx = batch_channel_offset + current_bin * n_bins + next_bin

        # Create batch-channel offset
        # Each sample (N, C) has a matrix of size n_bins * n_bins
        # Total matrices: N * C
        total_matrices = N * C
        matrix_size = self.n_bins * self.n_bins

        # [N, C, L-1] -> [N*C*(L-1)]
        flat_curr = current_bins.reshape(-1)
        flat_next = next_bins.reshape(-1)
        flat_valid = valid_trans.reshape(-1)

        # Prepare offset: [0, 1, ..., NC-1] repeated L-1 times
        # Shape [N, C] -> [N*C] -> [N*C, 1]
        batch_idx = torch.arange(total_matrices, device=device).unsqueeze(1)
        # Shape [N*C, L-1] -> flatten
        batch_idx = batch_idx.expand(-1, L - 1).reshape(-1)

        # Calculate flat indices where we need to add 1
        # Index in the huge flattened output tensor of shape [N*C, n_bins, n_bins] -> [N*C*n_bins*n_bins]
        # But easier: [N*C, n_bins*n_bins]
        flat_indices = batch_idx * matrix_size + flat_curr * self.n_bins + flat_next

        # Filter only valid transitions
        valid_indices = flat_indices[flat_valid]

        # Use bincount to sum up transitions
        # This counts occurrences of each index
        counts = torch.bincount(valid_indices, minlength=total_matrices * matrix_size)

        # Reshape back to [N, C, n_bins, n_bins]
        mtf = counts.reshape(N, C, self.n_bins, self.n_bins).float()

        # Normalize
        # Divide by number of valid transitions (L-1 valid)
        # Note: valid_lengths is L. Transitions is L-1.
        normalization = (valid_lengths - 1).clamp(min=1).unsqueeze(-1)
        mtf = mtf / normalization

        return self.resize_matrix(mtf)


class RP(BaseTime2Img):
    def __init__(self, output_size=65, threshold=0.2):
        super().__init__(output_size)
        self.threshold = threshold

    def forward(self, x):
        valid_mask = self.get_valid_mask(x)
        N, C, L = x.shape

        # 1. Standardization
        x_masked = x * valid_mask.float()
        valid_counts = valid_mask.sum(dim=2, keepdim=True).clamp(min=1)

        mean = x_masked.sum(dim=2, keepdim=True) / valid_counts

        # Standard deviation
        var = ((x_masked - mean) ** 2 * valid_mask.float()).sum(dim=2, keepdim=True) / valid_counts
        std = torch.sqrt(var).clamp(min=self.eps)

        x_norm = (x - mean) / std
        x_norm = torch.where(valid_mask, x_norm, torch.tensor(0.0, device=x.device))

        # 2. Vectorized Distance Matrix (RP)
        # We want to compute cdist for each (N, C) separately.
        # Reshape [N, C, L] -> [N*C, L, 1] to treat channels as part of the batch for cdist
        x_flat = x_norm.view(-1, L, 1)  # [N*C, L, 1]

        # cdist computes distance between each pair in the batch
        dist_matrix = torch.cdist(x_flat, x_flat, p=2)  # [N*C, L, L]

        # Threshold
        rp = (dist_matrix < self.threshold).float()

        # Reshape back: [N, C, L, L]
        rp = rp.view(N, C, L, L)

        # Mask out invalid regions (optional, but good for correctness)
        # valid_mask_sq: [N, C, L, L]
        valid_mask_sq = valid_mask.unsqueeze(3) & valid_mask.unsqueeze(2)
        rp = rp * valid_mask_sq.float()

        return self.resize_matrix(rp)
