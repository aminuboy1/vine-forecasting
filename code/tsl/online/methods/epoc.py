"""Faithful port of EPOC (Endpoint-Preserving Online Correction) with its global blending gate.

SOURCE
  Paper : T. Fujimoto & H. Nishi, "EPOC: Endpoint-Preserving Online Correction With Compressed Residual State for
          Multi-Horizon Time Series Forecasting", arXiv:2609.30929v2 (5 Oct 2026), Eqs. 4 and 10.
  Code  : https://github.com/keiotakmin/endpoint-preserving-residual-correction,
          commit f31dcfe58785b32ec5d03fe134821a87208cb274 (2026-10-16): src/endpoint_review_validation.py
          (`Config` "proposed", `PackedRidge`, `Tracker`, `evaluate`), src/channel_control.py (`RollingControl`,
          scope "global", kind "blend"), src/basis_probe.py (`columns`, the DCT-II basis); replay.py runs
          `evaluate(base, target, Config())`. The archived `boundary_context.BoundaryTracker('boundary4')` is the
          same method (their own test asserts equality to 1e-11).
  Licence: MIT (c) 2026 Takumi Fujimoto, Hiroaki Nishi. Reimplemented following their code line by line.

ALGORITHM (their defaults: K = 4, ridge 1, half-life 128 blocks, gate window 32, summary = endpoint)
  Blocks: origins spaced H apart, non-overlapping target blocks; the correction for block t uses completed blocks
  only. U = orthonormal DCT-II columns 0..K-1 (H x K). For each component k and channel c an independent ridge
  regression (p = 3 slopes + intercept) on
      v[k, c] = [ zprev[k, c],  (U^T B)[k, c],  sqrt(H) * rprev[c] ]
  where zprev = U^T e_prev (previous completed residual block, K coefficients), rprev = e_prev[-1] (its endpoint,
  shared by the K regressions of a channel), B the current frozen forecast; target z = U^T e, e = Y - B.
  Initial state zprev = 0, rprev = 0.
  Ridge (PackedRidge): exponentially weighted, decay rho = 2^(-1/128); penalty `ridge` on slopes AND intercept;
  sufficient statistics s (weight sum), cross (sum v), packed Gram (+ridge I, kept constant via (1-rho) ridge),
  rhs; prediction by the Schur-complement solve of the (p+1) x (p+1) augmented system.
  Correction: c = U pred (H x C); raw = B + c; c = raw - B (their operation order); issued = B + alpha c.
  Gate (RollingControl global blend, W = 32): alpha = clip(sum_b / sum_a, 0, 1) over the last W completed blocks,
  a = mean_c mean_h c^2, b = mean_c mean_h c e; alpha = 0 while sum_a <= 0.
  Update after block t's target is complete: gate.update(c, e); ridge.update(v, U^T e); zprev, rprev <- block t.
  Solve schedule: theirs solves every issue; we cache (slope, intercept) and re-solve only after an update
  (bit-identical). No guard, no output clipping, no physical-bound clipping (their `bounded` default is False).

DEVIATIONS (and why)
  1. Per-origin evaluation (our harness scores every origin; EPOC is defined on a stride-H block grid). Default
     (`update_stride=None` -> H): the ridge, the gate and the "previous block" summaries are those of the grid
     origins first_origin + k H, updated exactly as in their `evaluate`; forecasts at grid origins equal theirs.
     At an off-grid origin t the current ridge/gate are applied to t's own features with the previous block
     taken at the same phase (`offgrid="phase"`: the residual block of origin t - lag, lag = stride*ceil(d/stride);
     for d = H this is the block y[t-H+1..t] that ends at t, matching the feature's meaning). `offgrid="latest"`
     instead reuses the most recent grid block (their stateful `last_z/last_r` literally; measured to be harmful
     off-grid on a periodic toy, +96% MSE vs frozen, so not recommended).
     CAVEAT (measured): a single grid learner trains on one phase of the day when H equals the season, so its
     off-grid forecasts are weaker (periodic toy: -96% MSE on grid origins, -32% over all origins). Hence:
       * `phases="all"`: H independent copies of the published learner, copy p trained and used only on the grid
         first_origin + p + k H; each copy IS their algorithm on its grid (bit-identical to their `evaluate` on
         that grid); state is H times theirs. Recommended per-origin version.
       * `update_stride=1` (NOT their method): one learner updated on every matured origin, half-life and gate
         window rescaled to time units (the ridge penalty becomes relatively H times weaker). Ablation only.
  2. Delay d > H (not studied in the paper): updates happen at s + d (P-strict); with the default grid the
     previous-block features of grid origin s are those of grid origin s - H ceil(d/H), i.e. the most recent
     completed block, which is what their stateful tracker would hold under that delay.
  3. Missing values: their code has no guard and asserts finite losses. `skip_nonfinite` (default True; no-op on
     finite data) skips (component, channel) units whose target/features are non-finite and skips the gate
     update for a block containing non-finite values; `skip_nonfinite=False` keeps their arithmetic exactly.
     (The "missing-value guard" in our methodology is DRiC's scale guard, see dric.py; EPOC has none.)
  4. P-prefix: partial labels are not used (as in the paper): `uses_prefix = False`.
  State accounting follows their `state_bytes` (ids, basis, ridge statistics, zprev, rprev, gate) under "learned";
  issue-time tokens awaiting labels and the per-origin summaries kept for off-grid phases are "pending_*".
"""
from __future__ import annotations

import copy

import numpy as np

from ..base import OnlineMethod
from ..protocol import RevealedInfo


def dct_columns(h, ids):
    """Their `basis_probe.columns` (orthonormal DCT-II columns `ids`)."""
    ids = np.asarray(ids)
    u = np.cos(np.pi * (np.arange(h)[:, None] + .5) * ids / h) * np.sqrt(2 / h)
    u[:, ids == 0] /= np.sqrt(2)
    return u


class _PackedRidge:
    """Their `endpoint_review_validation.PackedRidge`, with the solution cached between updates."""

    def __init__(self, k, c, p, ridge, decay):
        self.p, self.ridge, self.decay = p, ridge, decay
        self.s = np.zeros(1)
        self.cross = np.zeros((k, c, p))
        self.gram = np.zeros((k, c, p * (p + 1) // 2))
        self.rhs = np.zeros((k, c, p + 1))
        self.i, self.j = np.triu_indices(p)
        self.gram[..., self.i == self.j] = ridge
        self._sol = None

    def _solve(self):
        i, j = self.i, self.j
        mat = np.empty((*self.cross.shape[:-1], self.p, self.p))
        mat[..., i, j] = self.gram
        mat[..., j, i] = self.gram
        d = self.ridge + self.s[0]
        mat -= self.cross[..., :, None] * self.cross[..., None, :] / d
        rhs = self.rhs[..., 1:] - self.cross * self.rhs[..., 0, None] / d
        slope = np.linalg.solve(mat, rhs[..., None])[..., 0]
        intercept = (self.rhs[..., 0] - np.sum(self.cross * slope, axis=-1)) / d
        return slope, intercept

    def predict(self, v):
        if self._sol is None:
            self._sol = self._solve()
        slope, intercept = self._sol
        return intercept + np.sum(v * slope, axis=-1)

    def update(self, v, y, keep=None):
        if keep is not None:                       # skip_nonfinite path: restore skipped units afterwards
            old = [a.copy() for a in (self.cross, self.gram, self.rhs)]
            s_old = self.s.copy()
        self.s *= self.decay
        self.s += 1
        self.cross *= self.decay
        self.cross += v
        self.gram *= self.decay
        i, j = self.i, self.j
        self.gram += v[..., i] * v[..., j]
        self.gram[..., i == j] += (1 - self.decay) * self.ridge
        self.rhs *= self.decay
        self.rhs[..., 0] += y
        self.rhs[..., 1:] += v * y[..., None]
        if keep is not None:
            # s is shared by all units (their "deterministic intercept weight sum"); a unit skipped here keeps
            # its own statistics, so it would see a different s. We therefore skip s's update only if no unit is
            # kept; partially skipped blocks are rare (non-finite data) and documented as a deviation.
            bad = ~keep
            self.cross[bad], self.gram[bad], self.rhs[bad] = old[0][bad], old[1][bad], old[2][bad]
            if not keep.any():
                self.s[:] = s_old
        self._sol = None


class _RollingControl:
    """Their `channel_control.RollingControl(c, 'global', 'blend', window)`."""

    def __init__(self, window):
        self.window = window
        self.history = np.zeros((window, 2, 1))
        self.sums = np.zeros((2, 1))
        self.t = 0

    def alpha(self):
        a, b = self.sums
        return np.clip(np.divide(b, a, out=np.zeros_like(b), where=a > 0), 0, 1)

    def update(self, correction, residual):
        a = np.mean(correction ** 2, axis=0)
        b = np.mean(correction * residual, axis=0)
        a = np.array([a.mean()])
        b = np.array([b.mean()])
        value = np.stack([a, b])
        slot = self.t % self.window
        self.sums += value - self.history[slot]
        self.history[slot] = value
        self.t += 1


class EPOC(OnlineMethod):
    name = "epoc"

    def __init__(self, k: int = 4, inputs=("z", "a", "r"), ridge: float = 1., half_life: float | None = 128.,
                 window: int = 32, summary: str = "endpoint", update_stride: int | None = None,
                 offgrid: str = "phase", phases: str = "single", skip_nonfinite: bool = True, **kw):
        super().__init__(**kw)
        if phases not in ("single", "all"):
            raise ValueError("phases must be 'single' or 'all'")
        self.phases = phases
        if phases == "all":
            self.name = "epoc_phased"
        if offgrid not in ("phase", "latest"):
            raise ValueError("offgrid must be 'phase' or 'latest'")
        if summary not in ("endpoint", "first", "middle", "mean"):
            raise ValueError(summary)
        self.k, self.inputs, self.ridge, self.half_life = int(k), tuple(inputs), float(ridge), half_life
        self.window, self.summary, self.stride_cfg = int(window), summary, update_stride
        self.offgrid, self.skip_nonfinite = offgrid, bool(skip_nonfinite)

    def setup(self, ctx):
        super().setup(ctx)
        self.subs = None
        if self.phases == "all" and int(self.stride_cfg or ctx.H) > 1:
            # one independent copy of the published learner per grid phase (each = their algorithm exactly)
            self.subs = []
            for ph in range(int(self.stride_cfg or ctx.H)):
                sub = copy.copy(self)
                sub.phases = "single"
                sub.setup(ctx)
                sub.first = ctx.first_origin + ph
                self.subs.append(sub)
            self.first, self.stride = ctx.first_origin, len(self.subs)
            return
        H, C = ctx.H, ctx.C
        if self.k > H:
            raise ValueError("k must be <= H")
        self.H, self.C = H, C
        self.stride = int(self.stride_cfg or H)
        per_block = H / self.stride
        decay = 1. if self.half_life is None else 2. ** (-1. / (float(self.half_life) * per_block))
        self.ids = np.arange(self.k, dtype=np.int64)
        self.u = dct_columns(H, self.ids)
        self.model = _PackedRidge(self.k, C, len(self.inputs), self.ridge, decay)
        self.gate = _RollingControl(int(round(self.window * per_block)))
        self.last_z = np.zeros((self.k, C)) if "z" in self.inputs else None
        self.last_r = np.zeros(C) if "r" in self.inputs else None
        self.first = ctx.first_origin
        self.lag = self.stride * int(np.ceil(ctx.delay / self.stride))
        self.summ: dict[int, tuple] = {}      # origin -> (z (k, C), r (C,)) for the 'phase' lookup
        self.pending: dict[int, tuple] = {}   # grid origin -> (token v, correction)
        self.n_updates = self.n_skipped = self.n_gate_skips = 0

    # ------------------------------------------------------------------ helpers
    def _is_grid(self, t):
        return (t - self.first) % self.stride == 0

    def _prev(self, t):
        if self.offgrid == "latest":
            return self.last_z, self.last_r
        got = self.summ.get(t - self.lag)
        if got is None:                        # before the first completed block: their zero initial state
            return (None if self.last_z is None else np.zeros((self.k, self.C)),
                    None if self.last_r is None else np.zeros(self.C))
        return got

    def _summary_value(self, residual):
        if self.summary == "first":
            return residual[0]
        if self.summary == "middle":
            return residual[(self.H - 1) // 2]
        if self.summary == "mean":
            return residual.mean(axis=0)
        return residual[-1]

    # ------------------------------------------------------------------ interface
    def _sub(self, t):
        return self.subs[(t - self.first) % self.stride]

    def predict(self, x, base_pred, t):
        if self.subs is not None:
            return self._sub(t).predict(x, base_pred, t)
        b = np.asarray(base_pred, np.float64)
        lz, lr = self._prev(t)
        values = []
        for f in self.inputs:
            if f == "z":
                values.append(lz)
            elif f == "a":
                values.append(self.u.T @ b)
            elif f == "r":
                values.append(np.broadcast_to(np.sqrt(self.H) * lr, (self.k, b.shape[1])))
        v = np.stack(values, axis=-1)
        correction = self.u @ self.model.predict(v)
        raw = b + correction
        correction = raw - b                  # their floating-point operation order
        alpha = self.gate.alpha()[0]
        if self._is_grid(t):
            self.pending[t] = (v, correction)
        return (b + alpha * correction).astype(np.float32)

    def update(self, info: RevealedInfo):
        for m in info.matured:
            if self.subs is not None:
                self._sub(m.origin)._update_one(m, info.now)
            else:
                self._update_one(m, info.now)

    def _update_one(self, m, now):
        b = np.asarray(m.base_pred, np.float64)
        residual = np.asarray(m.y, np.float64) - b
        z = self.u.T @ residual
        if self.offgrid == "phase":
            self.summ[m.origin] = (z.copy(), self._summary_value(residual).copy())
            for o in [o for o in self.summ if o < now - self.lag]:
                del self.summ[o]
        item = self.pending.pop(m.origin, None)
        if item is None:
            return
        v, correction = item
        if self.skip_nonfinite and not (np.isfinite(correction).all() and np.isfinite(residual).all()):
            self.n_gate_skips += 1
        else:
            self.gate.update(correction, residual)
        keep = None
        if self.skip_nonfinite:
            ok = np.isfinite(z) & np.isfinite(v).all(-1)
            if not ok.all():
                keep = ok
                self.n_skipped += int((~ok).sum())
                v = np.where(ok[..., None], v, 0.)
                z = np.where(ok, z, 0.)
        self.model.update(v, z, keep)
        if self.last_z is not None:
            self.last_z = z
        if self.last_r is not None:
            self.last_r = self._summary_value(residual).copy()
        self.n_updates += 1

    # ------------------------------------------------------------------ reporting
    def state_breakdown(self):
        if self.subs is not None:
            st = {}
            for i, sub in enumerate(self.subs):
                for k, v in sub.state_breakdown().items():
                    if i and k in ("basis", "component_ids"):
                        continue                          # shared constants, counted once
                    st[(f"pending_phase{i}_{k[8:]}" if k.startswith("pending_") else f"phase{i}_{k}")] = v
            return st
        st = {"component_ids": self.ids, "basis": self.u,
              "ridge": [self.model.s, self.model.cross, self.model.gram, self.model.rhs],
              "gate": [self.gate.history, self.gate.sums]}
        if self.last_z is not None:
            st["last_z"] = self.last_z
        if self.last_r is not None:
            st["last_r"] = self.last_r
        st["pending_tokens"] = [list(p) for p in self.pending.values()]
        if self.offgrid == "phase":
            st["pending_phase_summaries"] = [list(p) for p in self.summ.values()]
        return st

    def their_state_bytes(self) -> int:
        """Their `evaluate(...)['state_bytes']` convention (4,528 B at C = 7, K = 4, H = 24); summed over phases."""
        if self.subs is not None:
            return sum(sub.their_state_bytes() for sub in self.subs)
        m = self.model
        s = self.ids.nbytes + self.u.nbytes + sum(a.nbytes for a in (m.s, m.cross, m.gram, m.rhs))
        s += (0 if self.last_z is None else self.last_z.nbytes) + (0 if self.last_r is None else self.last_r.nbytes)
        return s + self.gate.history.nbytes + self.gate.sums.nbytes

    def extra_metrics(self):
        if self.subs is not None:
            ms = [sub.extra_metrics() for sub in self.subs]
            return {"n_updates": sum(m["n_updates"] for m in ms), "skipped_units": sum(m["skipped_units"] for m in ms),
                    "gate_skips": sum(m["gate_skips"] for m in ms),
                    "alpha_mean": float(np.mean([m["alpha"] for m in ms])),
                    "their_state_bytes": self.their_state_bytes(), "phases": len(self.subs),
                    "feature_lag": ms[0]["feature_lag"]}
        return {"n_updates": self.n_updates, "skipped_units": self.n_skipped, "gate_skips": self.n_gate_skips,
                "alpha": float(self.gate.alpha()[0]), "their_state_bytes": self.their_state_bytes(),
                "stride": self.stride, "feature_lag": self.lag}
