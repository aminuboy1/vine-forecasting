"""Provisional-versus-settled stress test (paper 2, methodology Sec. 7.6).

Creates real-time inputs x_t = y_t + eta_t, where eta is an AR(1) process per channel with coefficient
`phi` and stationary standard deviation `sd_frac` * std(y_c) (std over the rows given in `ref_rows`,
normally the training rows). Labels y stay settled and are released at s+d. Every method runs unchanged;
only the input channel differs. Methods that silently use inputs as targets will degrade.
"""
from __future__ import annotations

import numpy as np


def make_provisional(y: np.ndarray, phi: float = 0.9, sd_frac: float = 0.3, seed: int = 12345,
                     ref_rows: slice | None = None) -> np.ndarray:
    y = np.asarray(y, np.float32)
    T, C = y.shape
    ref = y[ref_rows] if ref_rows is not None else y
    sd_y = ref.std(0)
    sd_eps = sd_frac * sd_y * np.sqrt(1.0 - phi ** 2)        # innovation sd giving stationary sd = sd_frac*sd_y
    rng = np.random.default_rng(seed)                        # generator seed separate from training seeds
    eps = rng.standard_normal((T, C)) * sd_eps
    eta = np.empty((T, C), np.float64)
    eta[0] = rng.standard_normal(C) * sd_frac * sd_y
    for t in range(1, T):
        eta[t] = phi * eta[t - 1] + eps[t]
    return (y + eta).astype(np.float32)
