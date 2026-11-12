"""ELF-style closed-form Fourier linear corrector blended with the frozen backbone.

REIMPLEMENTATION (not the authors' code) of ELF, Lee et al., "Lightweight Online Adaption of Time Series
Foundation Model Forecasts", ICML 2025 (PMLR v267, lee25ag; arXiv:2502.12920). Following the paper:

ELF-Forecaster (EF)
  * channel-independent linear model with ONE complex weight matrix shared by all channels;
  * input: mean-removed lookback, real DFT, keep the lowest alpha fraction of bins (alpha = 0.9);
  * output: lowest alpha fraction of the H-step spectrum, zero-padded and inverse-real-FFT'd; the mean of
    the lookback is added back; targets are low-pass filtered the same way when fitting;
  * closed-form ridge W = (X^H X + lambda I)^{-1} X^H Y (lambda = 20) from accumulated sufficient
    statistics, rows scaled by a running per-channel std (Welford); refit every M = 200 matured pairs;
  * before the first fit the EF falls back to a seasonal-naive forecast; EF enters the blend only after
    `warmup_updates` (= 5) update steps.
ELF-Weighter
  * y = w * y_backbone + (1 - w) * y_EF;
  * slow weighter: exponential weights (eta = 0.5) on the cumulative per-update-step losses;
  * fast weighter: same, over the last B = 5 update steps;
  * merge weighter: exponential weights choosing between the fast- and slow-weighted combinations;
    w = beta * w_fast + (1 - beta) * w_slow.

Deviations / choices to report: losses are MSE (or MAE) on standardised data instead of MASE; in this
harness all fitting uses matured labels only (truthful delay d), which the original paper did not study;
the shared-W + Woodbury-inverse storage is replaced by storing X^H X (K_in x K_in) and solving at refit
(identical solution); the seasonal-naive fallback uses the period `season` (default: steps per day).
"""
from __future__ import annotations

import math

import numpy as np

from ..base import OnlineMethod
from ..protocol import RevealedInfo


def _softmax_neg(losses: np.ndarray, eta: float) -> np.ndarray:
    a = -eta * losses
    a = a - a.max(0, keepdims=True)
    e = np.exp(a)
    return e / e.sum(0, keepdims=True)


class ELFStyle(OnlineMethod):
    name = "elf"

    def __init__(self, alpha: float = 0.9, ridge: float = 20.0, update_every: int = 200, eta: float = 0.5,
                 fast_window: int = 5, warmup_updates: int = 5, loss: str = "mse", season: int | None = None,
                 per_channel_weights: bool = False, scale_rows: bool = True, warm_start_rows: int = 0, **kw):
        super().__init__(**kw)
        self.alpha, self.lam, self.M, self.eta = alpha, ridge, update_every, eta
        self.B, self.warmup, self.loss_kind, self.season = fast_window, warmup_updates, loss, season
        self.per_channel, self.scale_rows, self.warm_start_rows = per_channel_weights, scale_rows, warm_start_rows

    # ------------------------------------------------------------------ setup
    def setup(self, ctx):
        super().setup(ctx)
        L, H, C = ctx.L, ctx.H, ctx.C
        self.kin = max(1, math.ceil(self.alpha * (L // 2 + 1)))
        self.kout = max(1, math.ceil(self.alpha * (H // 2 + 1)))
        self.A = np.zeros((self.kin, self.kin), np.complex128)        # sum X^H X
        self.Bm = np.zeros((self.kin, self.kout), np.complex128)      # sum X^H Y
        self.W: np.ndarray | None = None
        self.n_rows = 0
        # Welford running per-channel statistics of the observed series (for row scaling)
        self.w_n, self.w_mean, self.w_m2 = 0, np.zeros(C), np.zeros(C)
        self.block: list = []                 # matured pairs since the last refit
        self.ef_pending: dict[int, np.ndarray] = {}   # EF forecasts awaiting their labels
        nw = C if self.per_channel else 1
        self.loss_cum = np.zeros((2, nw))      # cumulative per-update-step losses [backbone, EF]
        self.loss_recent: list = []             # last B per-update-step losses (fast weighter)
        self.merge_cum = np.zeros((2, nw))     # cumulative losses of [fast-combo, slow-combo]
        self.w = np.ones(nw)                    # weight on backbone
        self.w_fast = np.ones(nw)
        self.w_slow = np.ones(nw)
        self.n_updates = 0
        self.season = self.season or int((ctx.pretrain_info or {}).get("steps_per_day", 24))
        if self.warm_start_rows:
            self._warm_start()

    def _warm_start(self):
        h = self.ctx.history
        end = h.observed_until() + 1
        z = h.get(max(0, end - self.warm_start_rows), end)
        L, H = self.ctx.L, self.ctx.H
        from ...data.windows import make_windows
        X, Y, _ = make_windows(z, 0, len(z), L, H)
        for x, y in zip(X, Y):
            self._welford(x[-1])
        if len(X):
            self._accumulate(np.ascontiguousarray(X), np.ascontiguousarray(Y))
            self._refit()

    # ------------------------------------------------------------------ EF core
    def _welford(self, v):
        self.w_n += 1
        dlt = v - self.w_mean
        self.w_mean += dlt / self.w_n
        self.w_m2 += dlt * (v - self.w_mean)

    def _std(self):
        if self.w_n < 2:
            return np.ones_like(self.w_mean)
        s = np.sqrt(self.w_m2 / (self.w_n - 1))
        return np.where(s < 1e-6, 1.0, s)

    def _feat(self, X):
        """X: (N, L, C) -> complex features (N*C, kin) and means (N, C)."""
        mu = X.mean(1)
        F = np.fft.rfft(X - mu[:, None, :], axis=1)[:, : self.kin, :]     # (N, kin, C)
        return np.transpose(F, (0, 2, 1)).reshape(-1, self.kin), mu

    def _accumulate(self, X, Y):
        Fx, mu = self._feat(X)
        Fy = np.fft.rfft(Y - mu[:, None, :], axis=1)[:, : self.kout, :]
        Fy = np.transpose(Fy, (0, 2, 1)).reshape(-1, self.kout)
        if self.scale_rows:
            s = np.tile(self._std(), X.shape[0])[:, None]
            Fx, Fy = Fx / s, Fy / s
        self.A += Fx.conj().T @ Fx
        self.Bm += Fx.conj().T @ Fy
        self.n_rows += Fx.shape[0]

    def _refit(self):
        self.W = np.linalg.solve(self.A + self.lam * np.eye(self.kin), self.Bm)

    def _ef_forecast(self, x):
        L, H = self.ctx.L, self.ctx.H
        if self.W is None:  # seasonal naive fallback
            p = min(self.season, L)
            idx = (np.arange(H) % p) + (L - p)
            return x[idx]
        Fx, mu = self._feat(x[None])
        Fo = Fx @ self.W                                                 # (C, kout)
        full = np.zeros((Fo.shape[0], H // 2 + 1), np.complex128)
        full[:, : self.kout] = Fo
        y = np.fft.irfft(full, n=H, axis=1).T + mu                       # (H, C)
        return y

    # ------------------------------------------------------------------ interface
    def predict(self, x, base_pred, t):
        ef = self._ef_forecast(x.astype(np.float64)).astype(np.float32)
        self.ef_pending[t] = ef
        if self.n_updates < self.warmup:
            return base_pred
        w = self.w if self.per_channel else self.w[0]
        return (w * base_pred + (1.0 - w) * ef).astype(np.float32)

    def _loss(self, pred, y):
        e = pred - y
        v = (e ** 2) if self.loss_kind == "mse" else np.abs(e)
        return v.mean(0) if self.per_channel else np.array([v.mean()])

    def update(self, info: RevealedInfo):
        for m in info.matured:
            ef = self.ef_pending.pop(m.origin, None)
            self._welford(m.x[-1].astype(np.float64))
            self.block.append((m.x, m.y, m.base_pred, ef))
        if len(self.block) < self.M:
            return
        blk, self.block = self.block, []
        X = np.stack([b[0] for b in blk]).astype(np.float64)
        Y = np.stack([b[1] for b in blk]).astype(np.float64)
        # ---- weighter losses on this block (forecasts as issued)
        lb = np.mean([self._loss(b[2], b[1]) for b in blk], 0)
        le = np.mean([self._loss(b[3], b[1]) for b in blk if b[3] is not None], 0) if blk[0][3] is not None else lb
        lf = np.mean([self._loss(self._combo(self.w_fast, b[2], b[3]), b[1]) for b in blk], 0)
        ls = np.mean([self._loss(self._combo(self.w_slow, b[2], b[3]), b[1]) for b in blk], 0)
        step = np.stack([lb, le])
        self.loss_cum += step
        self.loss_recent = (self.loss_recent + [step])[-self.B:]
        self.merge_cum += np.stack([lf, ls])
        self.w_slow = _softmax_neg(self.loss_cum, self.eta)[0]
        self.w_fast = _softmax_neg(np.sum(self.loss_recent, 0), self.eta)[0]
        beta = _softmax_neg(self.merge_cum, self.eta)[0]
        self.w = beta * self.w_fast + (1 - beta) * self.w_slow
        # ---- EF refit with the new block
        self._accumulate(X, Y)
        self._refit()
        self.n_updates += 1

    def _combo(self, w, base, ef):
        if ef is None:
            return base
        w = w if self.per_channel else w[0]
        return w * base + (1 - w) * ef

    def state_breakdown(self):
        return {"XtX": self.A, "XtY": self.Bm, "W": self.W, "welford": [self.w_mean, self.w_m2],
                "weighter_losses": [self.loss_cum, self.merge_cum, self.loss_recent], "raw_block": [list(b[:3]) for b in self.block],
                "pending_ef_forecasts": list(self.ef_pending.values())}

    def extra_metrics(self):
        return {"n_updates": self.n_updates, "final_w_backbone": float(np.mean(self.w)), "kin": self.kin,
                "kout": self.kout}
