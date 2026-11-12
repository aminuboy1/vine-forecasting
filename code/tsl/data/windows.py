"""Sliding windows (zero-copy via numpy stride tricks) for offline training."""
from __future__ import annotations

import numpy as np
import torch


def make_windows(z: np.ndarray, start: int, end: int, L: int, H: int, stride: int = 1):
    """All (x, y) windows fully inside rows [start, end): x=z[s-L+1:s+1], y=z[s+1:s+H+1].

    Returns (X, Y) as strided *views* of shape (N, L, C) and (N, H, C) plus origin indices s (N,).
    drop_last is never applied: every complete window is included.
    """
    seg = z[start:end]
    n = seg.shape[0] - L - H + 1
    if n <= 0:
        C = z.shape[1]
        return (np.zeros((0, L, C), np.float32), np.zeros((0, H, C), np.float32), np.zeros(0, np.int64))
    w = np.lib.stride_tricks.sliding_window_view(seg, L + H, axis=0)  # (n, C, L+H)
    w = w[::stride]
    X = np.moveaxis(w[:, :, :L], 1, 2)
    Y = np.moveaxis(w[:, :, L:], 1, 2)
    origins = start + L - 1 + np.arange(0, n, stride)
    return X, Y, origins


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, z: np.ndarray, start: int, end: int, L: int, H: int, stride: int = 1):
        self.X, self.Y, self.origins = make_windows(z, start, end, L, H, stride)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(self.X[i])), torch.from_numpy(np.ascontiguousarray(self.Y[i]))


def iterate_batches(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator | None,
                    device: torch.device | str = "cpu"):
    """Yield float32 tensor batches from (possibly strided) arrays; drop_last=False."""
    n = len(X)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        xb = torch.from_numpy(np.ascontiguousarray(X[j], dtype=np.float32)).to(device)
        yb = torch.from_numpy(np.ascontiguousarray(Y[j], dtype=np.float32)).to(device)
        yield xb, yb
