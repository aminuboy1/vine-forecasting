"""Faithful port of the delay-aware audit's RLS bank ("RLS-Bank", median of passthrough + 6 RLS experts).

SOURCE
  Paper : Fentazi, Ameur & Ksentini, "Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for
          Time-Series Forecasting", arXiv:2610.12232v1 (8 Oct 2026), Section 4 "The bank, and why a median".
  Code  : https://github.com/Xodios/TRAINING-ON-THE-FUTURE-A-DELAY-AWARE-AUDIT-OF-TEST-TIME-ADAPTATION-FOR-TIME-SERIES-FORECASTING
          commit 4940d02b398cfd5ab503837edaa794411c7b5aa3 (2026-10-08),
          code_supplement/notebooks/rls_delay_tta_full_run.ipynb (the paper's harness): `Config` (cell 3),
          `gen_forecasts` + `build_feats` (cells 12), `RLS` / `RLSBankMed` (cell 14), `replay` (cell 16),
          `method_zoo` entry "rls_bank_med" (cell 18). The 70/10/20 notebook has byte-identical adapter code.
  Licence: MIT (c) 2026 Mohamed Readh Fentazi, Mazene Ameur & Adlen Ksentini. Reimplemented (not vendored); the
          update equations below follow their code line by line.

ALGORITHM (as in their `RLSBankMed` with featset "feat"; everything per (channel c, horizon step h))
  Features (F = 4), computed at origin t from the real-time INPUT channel only (P = steps per day):
      phi = [ yhat_t[h, c],  1,  x_t[c] - m_t[c],  snv_t[h, c] - m_t[c] ]
      m_t   = mean(x[t-P+1 .. t])          (trailing daily mean; theirs: swP[o-P].mean(-1), o = t + 1)
      snv_t = x[t+1+h-P]                   (seasonal naive; theirs: swH[o-P][h])
  Experts: k = 0 frozen passthrough w = [1, 0, 0, 0] (never updated); k = 1..6 RLS with forgetting
      lambda in (1.0, 0.9999, 0.9995, 0.999, 0.995, 0.99) (sorted descending, float32), all initialised at
      passthrough w = [1, 0, 0, 0] and P = p0 I with p0 = 10. Targets are the LABELS y (not residuals).
  Combination: per-(c, h) median of the 7 expert predictions w_k . phi (np.median over k).
  Update (when the label block of origin s' is released, i.e. at s' + d under P-strict), for every RLS expert:
      e = y - w.phi;  Pphi = P phi;  k = Pphi / (lam + phi.Pphi);  w += k e;
      P = (P - k Pphi^T) / lam;  P = (P + P^T)/2;
      trace clip: P *= min(1, rho / max(tr(P)/F, 1e-12)), rho = 10;
      reset: any (k, c, h) with non-finite w or tr(P) -> w = [1, 0, 0, 0], P = rho I.
  Solve schedule: none (RLS rank-1 update every matured label; O(K C H F^2) per step). No output clipping.
  Timing: their `replay` does update(origin i - d) then predict(origin i); our evaluator calls update(info_t)
      (label of origin t - d) before predict(x_t): identical order. Their "d >= H" runs are exactly P-strict.
  Precision: float32 throughout (as theirs). `dtype="float64"` is available but is not their setting.

DEVIATIONS (and why)
  1. Two channels: features are built from the input channel x, targets from the label channel y. In their code
     both are the same array z, so on public benchmarks (x == y) this is identical.
  2. Horizon longer than a day (H > P): their `snv = swH[o-P]` reads z[o-P .. o-P+H-1], i.e. FUTURE values when
     H > P (a leak; it never triggered in the paper, where H = 24 <= P). We use the causal seasonal naive
     snv[h] = x[t+1+h-P*ceil((h+1)/P)] (period repetition), which equals theirs whenever H <= P.
  3. When P > L (e.g. Weather P = 144 with L = 96) the trailing day is read through the guarded
     `ctx.history.inputs` (allowed: inputs up to now). Their harness had the full array; values are identical.
     If fewer than P rows exist before the first origin, the available rows are used (theirs would wrap).
  4. Features of a matured origin are recomputed at update time from the stored input window (deterministic,
     bit-identical to the predict-time features) instead of being read from a precomputed [T, C, H, F] tensor.
  5. `skip_nonfinite` (default True): (c, h) units whose released label or features are non-finite are left
     untouched instead of being updated and then reset by their non-finite rule. This is a no-op on finite
     streams (all public benchmarks after the usual preprocessing); `skip_nonfinite=False` restores their
     behaviour exactly.
  6. P-prefix: their bank never uses partial labels, so `uses_prefix = False` (prefix labels are ignored in
     P-prefix mode, as in their paper).
  The single-filter ablations of the paper are available as options: `lams=(0.999,), passthrough=False` is their
  "rls_feat"; `featset="calib"` (features [yhat, 1]) is their "rls_calib"; `trace_clip=1e12` is "*_noclip".
"""
from __future__ import annotations

import numpy as np

from ..base import OnlineMethod
from ..protocol import RevealedInfo

AUDIT_LAMS = (1.0, 0.9999, 0.9995, 0.999, 0.995, 0.99)


class AuditRLSBank(OnlineMethod):
    name = "rls_audit"

    def __init__(self, lams=AUDIT_LAMS, passthrough: bool = True, featset: str = "feat", p0: float = 10.0,
                 trace_clip: float = 10.0, season: int | None = None, dtype: str = "float32",
                 skip_nonfinite: bool = True, **kw):
        super().__init__(**kw)
        if featset not in ("feat", "calib"):
            raise ValueError("featset must be 'feat' (F=4, the bank) or 'calib' (F=2)")
        self.lams_cfg, self.passthrough, self.featset = tuple(lams), bool(passthrough), featset
        self.p0, self.clip, self.season = float(p0), float(trace_clip), season
        self.dtype, self.skip_nonfinite = np.dtype(dtype), bool(skip_nonfinite)

    # ------------------------------------------------------------------ setup
    def setup(self, ctx):
        super().setup(ctx)
        H, C, dt = ctx.H, ctx.C, self.dtype
        self.H, self.C = H, C
        self.P_season = int(self.season or (ctx.pretrain_info or {}).get("steps_per_day", 24))
        self.F = 4 if self.featset == "feat" else 2
        F = self.F
        self.lams = np.array(sorted(self.lams_cfg, reverse=True), dt)       # theirs: desc, float32
        self.n_rls = len(self.lams)
        K = self.n_rls + int(self.passthrough)
        self.w = np.zeros((K, C, H, F), dt)
        self.w[..., 0] = 1.0                                               # all experts start at passthrough
        self.P = np.broadcast_to(np.eye(F, dtype=dt) * dt.type(self.p0), (self.n_rls, C, H, F, F)).copy()
        self._lam3 = self.lams[:, None, None]
        self._lam5 = self.lams[:, None, None, None, None]
        self._snv_idx = np.arange(H) % self.P_season                       # causal seasonal-naive rows
        self.n_updates = 0
        self.n_resets = 0
        self.n_skipped = 0

    # ------------------------------------------------------------------ features (C, H, F)
    def _day(self, window: np.ndarray, origin: int) -> np.ndarray:
        P = self.P_season
        if P <= window.shape[0]:
            return window[-P:]
        return self.ctx.history.inputs(origin - P + 1, origin + 1)          # guarded: inputs up to now only

    def _features(self, window: np.ndarray, base: np.ndarray, origin: int) -> np.ndarray:
        dt = self.dtype
        base_ch = np.ascontiguousarray(base.T, dtype=dt)                   # (C, H)
        if self.featset == "calib":
            return np.stack([base_ch, np.ones_like(base_ch)], -1)
        day = self._day(window, origin).astype(dt, copy=False)            # (P, C)
        mday = day.mean(0).astype(dt)                                      # (C,)
        last = window[-1].astype(dt, copy=False)
        snv = day[self._snv_idx].T                                         # (C, H): x[t+1+h-P] for h < P
        X = np.empty((self.C, self.H, 4), dt)
        X[..., 0] = base_ch
        X[..., 1] = 1.0
        X[..., 2] = (last - mday)[:, None]
        X[..., 3] = snv - mday[:, None]
        return X

    # ------------------------------------------------------------------ interface
    def predict(self, x, base_pred, t):
        X = self._features(x, base_pred, t)
        pk = np.einsum("kchf,chf->kch", self.w, X)
        out = np.median(pk, axis=0)                                        # per-coordinate median
        return out.T.astype(np.float32)

    def update(self, info: RevealedInfo):
        for m in info.matured:
            X = self._features(m.x, m.base_pred, m.origin)
            y = np.ascontiguousarray(m.y.T, dtype=self.dtype)              # (C, H) label block
            mask = None
            if self.skip_nonfinite:
                ok = np.isfinite(y) & np.isfinite(X).all(-1)
                if not ok.all():
                    mask = ok
                    self.n_skipped += int((~ok).sum())
                    X = np.where(ok[..., None], X, 0).astype(self.dtype)
                    y = np.where(ok, y, 0).astype(self.dtype)
                    w_old, P_old = self.w.copy(), self.P.copy()
            self._rls_step(X, y)
            if mask is not None:
                bad = ~mask
                self.w[:, bad] = w_old[:, bad]
                self.P[:, bad] = P_old[:, bad]
            self.n_updates += 1

    def _rls_step(self, x, y):
        """Verbatim (vectorised) `RLSBankMed.update` of the audit notebook."""
        wr = self.w[1:] if self.passthrough else self.w
        e = y[None] - np.einsum("kchf,chf->kch", wr, x)
        lam = self._lam3
        Px = np.einsum("kchij,chj->kchi", self.P, x)
        kg = Px / (lam + np.einsum("kchf,chf->kch", Px, x))[..., None]
        wr += kg * e[..., None]
        self.P = (self.P - kg[..., :, None] * Px[..., None, :]) / self._lam5
        self.P = 0.5 * (self.P + self.P.transpose(0, 1, 2, 4, 3))
        F = self.F
        tr = np.einsum("kchff->kch", self.P) / F
        self.P *= np.minimum(1.0, self.clip / np.maximum(tr, 1e-12))[..., None, None]
        bad = ~(np.isfinite(wr).all(-1) & np.isfinite(tr))
        if bad.any():
            self.n_resets += int(bad.sum())
            wr[bad] = 0.0
            wr[..., 0][bad] = 1.0
            self.P[bad] = np.eye(F, dtype=self.dtype) * self.clip

    def state_breakdown(self):
        return {"w": self.w, "P": self.P, "lams": self.lams}

    def extra_metrics(self):
        return {"n_updates": self.n_updates, "n_resets": self.n_resets, "n_skipped_units": self.n_skipped,
                "season": self.P_season, "n_experts": int(self.w.shape[0])}
