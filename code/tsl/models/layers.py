"""Shared layers: RevIN (reversible instance normalisation) and series decomposition."""
from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Kim et al., ICLR 2022. Per-instance, per-channel normalisation (no affine by default).

    Normalises over the time axis of the *input* window only, so it is causal.
    """

    def __init__(self, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1)) if affine else None
        self.bias = nn.Parameter(torch.zeros(1)) if affine else None

    def norm(self, x: torch.Tensor):
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x, (mean, std)

    def denorm(self, y: torch.Tensor, stats):
        mean, std = stats
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps)
        return y * std + mean


class MovingAvg(nn.Module):
    """Moving average with replicate padding (as in Autoformer/DLinear), implemented as ONE matmul with a
    cached (L x L) averaging matrix: identical result (to float rounding), several times faster on CPU
    for single-window online steps."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size
        self._cache: dict = {}

    def _matrix(self, L: int, device, dtype):
        key = (L, device, dtype)
        if key not in self._cache:
            front, back = (self.k - 1) // 2, self.k // 2
            A = torch.zeros(L, L, dtype=torch.float64)
            for t in range(L):
                for j in range(t - front, t + back + 1):
                    A[t, min(max(j, 0), L - 1)] += 1.0 / self.k
            self._cache[key] = A.to(device=device, dtype=dtype)
        return self._cache[key]

    def forward(self, x):  # (B, L, C)
        return torch.matmul(self._matrix(x.shape[1], x.device, x.dtype), x)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.ma = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.ma(x)
        return x - trend, trend
