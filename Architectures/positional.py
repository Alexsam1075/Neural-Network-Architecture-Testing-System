import math

import torch
import torch.nn as nn


class DynamicSinusoidalPositionEncoding(nn.Module):
    """Sinusoidal positions generated for the requested length, not a fixed table."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, batch_size: int, length: int, device, dtype=None) -> torch.Tensor:
        dtype = dtype or torch.float32
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        even_dims = torch.arange(0, self.dim, 2, device=device, dtype=dtype)
        div_term = torch.exp(even_dims * (-math.log(10000.0) / max(1, self.dim)))

        pe = torch.zeros(length, self.dim, device=device, dtype=dtype)
        angles = positions * div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(angles)
        if self.dim > 1:
            pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0).expand(batch_size, -1, -1)
