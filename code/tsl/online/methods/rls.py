"""Recursive-least-squares filter bank on backbone residuals.

One independent RLS filter per (horizon step h, channel c) with exponential forgetting factor lambda.
For a forecast issued at origin s, the filter for (h, c) predicts the frozen backbone's residual
r_s[h, c] = y_s[h, c] - base_s[h, c] from the feature vector

    phi_s[h, c] = [1, base_s[h, c] (optional), r_{s-d}[h, c], r_{s-d-1}[h, c], ..., r_{s-d-p+1}[h, c]]

i.e. a bias, a gain term and p lags of *matured* residuals (origin s-d is the most recent origin whose
label is released at time s). The corrected forecast is base_s + theta^T phi_s. When the label of
origin s' matures, every filter does one RLS step on phi_{s'} (recomputed from the retained residual
history, identical to the one used at predict time). Fully vectorised over (H, C):
cost O(H C K^2) per step with K = 1 + use_base + p.
"""
from __future__ import annotations

import numpy as np

from ..base import OnlineMethod
from ..protocol import RevealedInfo


class RLSBank(OnlineMethod):
    name = "rls"

    def __init__(self, n_lags: int = 2, use_base: bool = True, use_bias: bool = True, lam: float = 0.999,
                 p0: float = 1.0, clip: float | None = 5.0, dtype: str = "float64", **kw):
        super().__init__(**kw)
        self.p, self.use_base, self.use_bias, self.lam = n_lags, use_base, use_bias, lam
        self.p0, self.clip, self.dtype = p0, clip, np.dtype(dtype)

    def setup(self, ctx):
        super().setup(ctx)
        H, C = ctx.H, ctx.C
        self.d = ctx.delay
        self.K = int(self.use_bias) + int(self.use_base) + self.p
        if self.K == 0:
            raise ValueError("RLS needs at least one feature")
        self.theta = np.zeros((H, C, self.K), self.dtype)
        self.P = np.broadcast_to(np.eye(self.K, dtype=self.dtype) * self.p0, (H, C, self.K, self.K)).copy()
        # residual history keyed by origin; only the last d+p matured origins are needed (see _features)
        self.rhist: dict[int, np.ndarray] = {}
        self._phi = np.zeros((H, C, self.K), self.dtype)
        self.n_updates = 0

    def _features(self, base_pred, origin):
        """phi for a forecast issued at `origin`; lags = residuals of origins origin-d, ..., origin-d-p+1."""
        phi = self._phi                                 # preallocated (H, C, K) work buffer
        j = 0
        if self.use_bias:
            phi[..., 0] = 1.0
            j = 1
        if self.use_base:
            phi[..., j] = base_pred
            j += 1
        for k in range(self.p):
            r = self.rhist.get(origin - self.d - k)
            if r is None:
                phi[..., j + k] = 0.0
            else:
                phi[..., j + k] = r
        return phi

    def predict(self, x, base_pred, t):
        phi = self._features(base_pred, t)
        corr = np.einsum("hck,hck->hc", self.theta, phi)
        if self.clip is not None:
            np.clip(corr, -self.clip, self.clip, out=corr)
        return (base_pred + corr).astype(np.float32)

    def update(self, info: RevealedInfo):
        for m in info.matured:
            r = (m.y - m.base_pred).astype(self.dtype)              # (H, C) residual of the frozen backbone
            phi = self._features(m.base_pred, m.origin)              # same features as at predict time
            lam = self.lam
            # einsum (no path optimisation) is several times faster than batched matmul for tiny K x K blocks
            Pphi = np.einsum("hcij,hcj->hci", self.P, phi)            # (H, C, K)
            denom = lam + np.einsum("hck,hck->hc", phi, Pphi)         # (H, C)
            gain = Pphi / denom[..., None]
            err = r - np.einsum("hck,hck->hc", self.theta, phi)
            self.theta += gain * err[..., None]
            self.P -= np.einsum("hci,hcj->hcij", gain, Pphi)
            self.P *= 1.0 / lam
            self.n_updates += 1
            if self.n_updates % 1000 == 0:                           # keep P symmetric
                self.P = 0.5 * (self.P + np.swapaxes(self.P, -1, -2))
            if self.p > 0:
                self.rhist[m.origin] = r.astype(np.float32)
                # origins needed later: >= (next update origin) - d - p + 1 = m.origin + 1 - d - p + 1
                for o in [o for o in self.rhist if o < m.origin - self.d - self.p + 1]:
                    del self.rhist[o]

    def state_breakdown(self):
        return {"theta": self.theta, "P": self.P, "pending_residual_history": list(self.rhist.values())}

    def extra_metrics(self):
        return {"n_updates": self.n_updates}
