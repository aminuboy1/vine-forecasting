"""Synthetic data-generating process for paper 2 (ablation A11): a frozen backbone with a KNOWN, regime-dependent
under-reaction, so the theory's quantities (predictable revisions, delay penalty, recovery time) are controlled.

Series (per channel): y_t = level_t + u_t, u_t AR(1).
  * "jump" regimes: the level is a random walk with occasional jumps (shifts persist);
  * "revert" regimes: the level is pinned and u_t is strongly mean-reverting.
Regimes alternate every `regime_len` steps (abrupt switches), so the optimal correction of a fixed backbone flips.

Backbone (frozen, torch): yhat_{t,h} = m_t + (1 - kappa_h) (x_t - m_t), with m_t the lookback mean and
kappa_h = kappa_max * h / H: it shrinks long horizons towards the window mean (under-reacts to level shifts,
over-reacts in reverting regimes). Its revisions are predictable, with regime-dependent sign.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def revision_dgp(T: int = 6000, C: int = 3, regime_len: int = 600, jump_prob: float = 0.02, jump_sd: float = 1.0,
                 rw_sd: float = 0.05, ar_phi_jump: float = 0.5, ar_phi_revert: float = 0.3, noise_sd: float = 0.3,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns y (T, C) float32 and the regime label per step (T,) (0 = jump, 1 = revert)."""
    rng = np.random.default_rng(seed)
    regime = (np.arange(T) // regime_len) % 2
    y = np.zeros((T, C))
    level = np.zeros(C)
    u = np.zeros(C)
    for t in range(T):
        if regime[t] == 0:
            level = level + rw_sd * rng.standard_normal(C) + (rng.random(C) < jump_prob) * jump_sd * rng.standard_normal(C)
            u = ar_phi_jump * u + noise_sd * rng.standard_normal(C)
        else:
            u = ar_phi_revert * u + noise_sd * rng.standard_normal(C)
        y[t] = level + u
    y = (y - y[: T // 5].mean(0)) / (y[: T // 5].std(0) + 1e-8)
    return y.astype(np.float32), regime


class UnderReactingBackbone(nn.Module):
    """yhat_{t,h} = m + (1 - kappa_h)(x_t - m), m = mean of the lookback window. Parameter-free, frozen."""

    def __init__(self, L: int, H: int, kappa_max: float = 0.8):
        super().__init__()
        self.L, self.H = L, H
        self.register_buffer("kappa", torch.linspace(kappa_max / H, kappa_max, H))

    def forward(self, x, **_):                    # x (B, L, C) -> (B, H, C)
        m = x.mean(1, keepdim=True)
        last = x[:, -1:, :]
        return m + (1.0 - self.kappa[None, :, None]) * (last - m)
