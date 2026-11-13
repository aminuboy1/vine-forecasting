"""Faithful port of DRiC (DCT Ridge Corrector): local (DRiC-L4/L8) and global (DRiC-G) recursive ridge correctors.

SOURCE
  Paper : T. Fujimoto & H. Nishi, "Local and Global Recursive Ridge Regression for Online Residual Correction of
          Frozen Multivariate Forecasters under Label Delay", arXiv:2610.29604v1 (21 Oct 2026), Section 3.
  Code  : https://github.com/keiotakmin/dric, commit 7ce0a33fb6ffa506f253b7376b1ff9e773df5349 (2026-10-21),
          file ridge_corrector.py (`dct_basis`, `Ridge`, `Blend`, `Config`, `Corrector`, `replay`).
  Licence: MIT (c) 2026 Takumi Fujimoto, Hiroaki Nishi. Reimplemented following their code line by line.

ALGORITHM (their `Corrector`; defaults = the paper's settings)
  Blocks: forecasts are issued on a grid of origins spaced H apart (non-overlapping target blocks). For a block
  with frozen forecast B (H x C) and input window W (lookback x C, lookback a multiple of H, M = lookback / H
  sub-blocks, newest first S_1..S_M), with U = orthonormal DCT-II basis (H x J), J = GUARD_COMPONENTS = 8:
    a   = U^T B (J x C);   last = sqrt(H) * W[-1]
    DRiC-L(K): per channel  x_l = [1, a[:K], (U^T S_1)[:K], last]           (q = 2K + 2; local_blocks = 1)
               target (z - zg)[:K], z = U^T (Y - B)
    DRiC-G   : one system   x_g = [1, a, U^T S_1, ..., U^T S_M, last]        (q_g = 2 + J + M J = 42 at M = 4)
               target z (J coefficients), samples of all kept channels pooled
  Ridge: exponentially weighted, intercept penalised; G_0 = ridge I, R_0 = 0; on an admitted block
    G <- rho G + sum x x^T + (1 - rho) ridge I,  R <- rho R + sum x z^T,  rho = 2^(-1/128) (half-life 128 blocks);
    prediction beta = G^{-1} R (solved per channel for L), packed upper-triangular Gram as in their code.
  Guard (tau = 10): per-channel scale s of the residual power p = ||U^T e||^2 / J; s starts at the first block's
    p (NaN = "fresh"); a block is admitted iff p <= tau^2 s; admitted blocks update s <- rho_s s + (1-rho_s) p,
    rho_s = 2^(-1/128); skipped blocks change neither the ridges nor s. The issued coefficient vector (zero-padded
    to J) is rescaled to norm <= tau sqrt(J s) (no clip while s is fresh).
  Blend: one weight shared by all channels, clip(sum<g, e> / sum||g||^2, 0, 1) over the last 32 completed blocks
    (means over H x C), 0 while the denominator is 0. Issued forecast = B + w * U z (z clipped).
  Solve schedule: their code solves G beta = R at every issue; we cache beta and re-solve only after an admitted
    update (bit-identical: G, R only change in `update`).
  Update timing: their replay issues block t then updates on block t's target before issuing block t + 1, i.e.
    label delay d = H on the block grid. Under our P-strict evaluator with d = H the label of grid origin s is
    released at s + H, before the forecast at grid origin s + H: identical order.

DEVIATIONS (and why)
  1. Per-origin evaluation. Our harness scores a forecast at EVERY origin; DRiC is defined on a stride-H block
     grid. Default (`update_stride=None` -> H): the learner (ridges, guard, blend) is updated only with grid
     origins s = first_origin + k H, exactly as in their replay, so the forecasts at grid origins are those of
     their code; forecasts at off-grid origins apply the current (frozen-between-updates) corrector to that
     origin's own features (all DRiC features are origin-local, so this is well defined).
     CAVEAT (measured): a single grid learner only ever trains on one phase of the day when H equals the season
     (e.g. hourly data, H = 24), so its off-grid forecasts can be much worse than its grid forecasts (periodic
     toy: -96% MSE on the grid, -43% over all origins). Two per-origin deployments are therefore provided:
       * `phases="all"`: H independent copies of the published learner, copy p trained and used only on the
         grid first_origin + p + k H. Every copy IS their algorithm on its own block grid (bit-identical to their
         replay on that grid); state is H times theirs. Recommended per-origin version.
       * `update_stride=1` (NOT their method): one learner updated on every matured origin with all half-lives
         and the blend window rescaled to keep their memory in time units (half-life 128 H steps, window 32 H
         steps); the ridge penalty is then relatively H times weaker. Ablation only.
  2. Delay d > H (not studied in the paper): the update of grid origin s happens at s + d (P-strict); the
     corrector keeps issuing with the last admitted state. Pending issue-time internals (features, zg,
     correction) are kept for every grid origin awaiting its label (their replay keeps one).
  3. Lookback: their code requires lookback % H == 0. If our L is not a multiple of H, the newest floor(L/H) H
     rows are used; if L < H, `lookback=H` rows are read through `ctx.history.inputs` (allowed). Override with
     `lookback=`. H < J = 8 raises (their DCT basis would be rank-deficient).
  4. `skip_nonfinite` (default True; no-op on finite data): a channel whose residual or features contain
     NaN/inf is treated like a guard skip, and the blend skips a block with any non-finite value. Their code has
     no NaN handling (it relies on forward-filled data; a NaN in one channel would poison the shared blend).
     `skip_nonfinite=False` restores their arithmetic exactly. Their "missing-value guard" is the scale guard
     above (missing codes such as -9999 become huge residual blocks and are skipped); it is always on unless
     `guard=False`.
  5. P-prefix: partial labels are not used (as in the paper): `uses_prefix = False`.
  State accounting follows their `state_bytes` (ridges, basis, component ids, guard scale, blend) under
  "learned"; issue-time internals awaiting labels are reported as "pending_*" (they do not count those).
"""
from __future__ import annotations

import copy

import numpy as np

from ..base import OnlineMethod
from ..protocol import RevealedInfo

GUARD_COMPONENTS = 8   # J: residual DCT coefficients watched by the guard (and width of the correction vector)
BLEND_WINDOW = 32
VARIANTS = {"l8": dict(local=8, global_=False), "l4": dict(local=4, global_=False),
            "g": dict(local=0, global_=True), "both": dict(local=8, global_=True)}


def dct_basis(h, k):
    """Orthonormal DCT-II basis, h x k (verbatim from their code)."""
    ids = np.arange(k)
    u = np.cos(np.pi * (np.arange(h)[:, None] + .5) * ids / h) * np.sqrt(2 / h)
    u[:, ids == 0] /= np.sqrt(2)
    return u


class _Ridge:
    """Their `Ridge` (exponentially weighted, penalised intercept, packed Gram), with a cached solution."""

    def __init__(self, channels, width, outputs, shared, ridge, decay):
        self.shared, self.ridge, self.decay = shared, ridge, decay
        self.i, self.j = np.triu_indices(width)
        self.diag = self.i == self.j
        groups = 1 if shared else channels
        self.gram = np.zeros((groups, len(self.i)))
        self.gram[:, self.diag] = ridge
        self.rhs = np.zeros((groups, width, outputs))
        self.width = width
        self._beta = None

    def beta(self):
        if self._beta is None:
            g = np.empty((self.gram.shape[0], self.width, self.width))
            g[:, self.i, self.j] = self.gram
            g[:, self.j, self.i] = self.gram
            self._beta = np.linalg.solve(g, self.rhs)
        return self._beta

    def predict(self, x):
        beta = self.beta()
        return x @ beta[0] if self.shared else np.einsum('cp,cpk->ck', x, beta)

    def update(self, x, y, keep):
        if self.shared:
            if not keep.any():
                return
            x, y = x[keep], y[keep]
            self.gram *= self.decay
            self.gram += (x[:, self.i] * x[:, self.j]).sum(0, keepdims=True)
            self.gram[:, self.diag] += (1 - self.decay) * self.ridge
            self.rhs *= self.decay
            self.rhs += (x[:, :, None] * y[:, None, :]).sum(0, keepdims=True)
            self._beta = None
            return
        if not keep.any():
            return
        gram, rhs = self.gram * self.decay, self.rhs * self.decay
        gram += x[:, self.i] * x[:, self.j]
        gram[:, self.diag] += (1 - self.decay) * self.ridge
        rhs += x[:, :, None] * y[:, None, :]
        self.gram[keep], self.rhs[keep] = gram[keep], rhs[keep]
        self._beta = None


class _Blend:
    """Their `Blend`: shared weight clip(<c, r> / <c, c>, 0, 1) over the last `window` blocks."""

    def __init__(self, window):
        self.history, self.sums, self.t = np.zeros((window, 2)), np.zeros(2), 0

    def weight(self):
        a, b = self.sums
        return float(np.clip(b / a, 0, 1)) if a > 0 else 0.

    def update(self, correction, residual):
        value = np.array([np.mean(correction ** 2), np.mean(correction * residual)])
        slot = self.t % len(self.history)
        self.sums += value - self.history[slot]
        self.history[slot] = value
        self.t += 1


class DRiC(OnlineMethod):
    name = "dric"

    def __init__(self, variant: str = "l8", local: int | None = None, local_blocks: int = 1,
                 global_: bool | None = None, guard: bool = True, tau: float = 10., ridge: float = 1.,
                 half_life: float = 128., guard_half_life: float = 128., blend_window: int = BLEND_WINDOW,
                 lookback: int | None = None, update_stride: int | None = None, phases: str = "single",
                 skip_nonfinite: bool = True, **kw):
        super().__init__(**kw)
        if phases not in ("single", "all"):
            raise ValueError("phases must be 'single' or 'all'")
        self.phases = phases
        v = variant.lower()
        if v not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        self.variant = v
        self.local = VARIANTS[v]["local"] if local is None else int(local)
        self.global_ = VARIANTS[v]["global_"] if global_ is None else bool(global_)
        if not (self.local or self.global_):
            raise ValueError("at least one ridge is required")
        self.local_blocks, self.guard, self.tau, self.ridge = int(local_blocks), bool(guard), float(tau), float(ridge)
        self.half_life, self.guard_half_life, self.blend_window = float(half_life), float(guard_half_life), blend_window
        self.lookback_cfg, self.stride_cfg, self.skip_nonfinite = lookback, update_stride, bool(skip_nonfinite)
        self.name = f"dric_{v}" + ("_phased" if phases == "all" else "")

    # ------------------------------------------------------------------ setup
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
        H, C, L = ctx.H, ctx.C, ctx.L
        J = GUARD_COMPONENTS
        if H < J:
            raise ValueError(f"DRiC needs H >= {J} (DCT components)")
        self.H, self.C = H, C
        lb = self.lookback_cfg or max(H, (L // H) * H)
        if lb % H:
            raise ValueError("lookback must be a multiple of H")
        self.lb, self.blocks = lb, lb // H
        self.stride = int(self.stride_cfg or H)
        per_block = H / self.stride                       # updates per block of H steps
        self.u = dct_basis(H, J)
        self.ids = np.arange(J, dtype=np.int64)           # component ids (counted in their state)
        kl = self.local
        decay = 2. ** (-1. / (self.half_life * per_block))
        self.glob = _Ridge(C, 2 + J + self.blocks * J, J, True, self.ridge, decay) if self.global_ else None
        self.loc = _Ridge(C, 1 + kl + (kl * self.local_blocks + 1 if self.local_blocks else 0), kl, False,
                          self.ridge, decay) if kl else None
        self.scale = np.full(C, np.nan) if self.guard else None
        self.gdecay = 2. ** (-1. / (self.guard_half_life * per_block))
        self.blend = _Blend(int(round(self.blend_window * per_block)))
        self.first = ctx.first_origin
        self.pending: dict[int, tuple] = {}
        self.n_updates = self.n_skipped = self.n_blend_skips = 0

    # ------------------------------------------------------------------ their Corrector, per origin
    def _window(self, x, t):
        if x.shape[0] >= self.lb:
            return x[-self.lb:]
        return self.ctx.history.inputs(t - self.lb + 1, t + 1)

    def _features(self, base, window):
        u, h, c = self.u, self.H, base.shape[1]
        ones, a = np.ones((1, c)), u.T @ base
        sub = window.reshape(self.blocks, h, c)[::-1]
        last = np.sqrt(h) * window[-1:]
        xg = np.concatenate([ones, a] + [u.T @ s for s in sub] + [last]).T if self.glob else None
        kl, nb = self.local, self.local_blocks
        xl = np.concatenate([ones, a[:kl]] + ([(u.T @ s)[:kl] for s in sub[:nb]] + [last] if nb else [])).T \
            if self.loc else None
        return xg, xl

    def _correct(self, base, window):
        xg, xl = self._features(base, window)
        c, k = base.shape[1], GUARD_COMPONENTS
        zg = self.glob.predict(xg) if self.glob else np.zeros((c, k))
        z = zg.copy()
        if self.loc:
            z[:, :self.local] += self.loc.predict(xl)
        if self.scale is not None:
            limit = self.tau * np.sqrt(k * np.nan_to_num(self.scale, nan=np.inf))
            z *= np.minimum(1., limit / np.maximum(np.linalg.norm(z, axis=1), 1e-300))[:, None]
        return self.u @ z.T, (xg, xl, zg)

    def _is_grid(self, t):
        return (t - self.first) % self.stride == 0

    def _sub(self, t):
        return self.subs[(t - self.first) % self.stride]

    def predict(self, x, base_pred, t):
        if self.subs is not None:
            return self._sub(t).predict(x, base_pred, t)
        base = np.asarray(base_pred, np.float64)
        window = np.asarray(self._window(x, t), np.float64)
        correction, pend = self._correct(base, window)
        if self._is_grid(t):
            self.pending[t] = (pend, correction)
        return (base + self.blend.weight() * correction).astype(np.float32)

    def update(self, info: RevealedInfo):
        for m in info.matured:
            if self.subs is not None:
                self._sub(m.origin)._update_one(m)
            else:
                self._update_one(m)

    def _update_one(self, m):
        item = self.pending.pop(m.origin, None)
        if item is None:                              # off-grid origin: not a DRiC block
            return
        (xg, xl, zg), correction = item
        base = np.asarray(m.base_pred, np.float64)
        residual = np.asarray(m.y, np.float64) - base
        finite_blk = True
        if self.skip_nonfinite:
            finite_blk = bool(np.isfinite(residual).all() and np.isfinite(correction).all())
        if finite_blk:
            self.blend.update(correction, residual)
        else:
            self.n_blend_skips += 1
        z = (self.u.T @ residual).T
        keep = np.ones(len(z), bool)
        if self.skip_nonfinite:
            fin = np.isfinite(z).all(1)
            if xg is not None:
                fin &= np.isfinite(xg).all(1)
            if xl is not None:
                fin &= np.isfinite(xl).all(1)
            fin &= np.isfinite(zg).all(1)
            keep = fin.copy()
        if self.scale is not None:
            power = np.mean(z ** 2, axis=1)
            fresh = np.isnan(self.scale)
            if self.skip_nonfinite:
                with np.errstate(invalid="ignore"):
                    keep = fin & (fresh | (power <= self.tau ** 2 * self.scale))
                self.scale = np.where(fresh & fin, power,
                                      np.where(keep, self.gdecay * self.scale + (1 - self.gdecay) * power,
                                               self.scale))
            else:
                keep = fresh | (power <= self.tau ** 2 * self.scale)
                self.scale = np.where(fresh, power, np.where(keep, self.gdecay * self.scale +
                                                             (1 - self.gdecay) * power, self.scale))
        if self.skip_nonfinite and not keep.all():
            z = np.where(keep[:, None], z, 0.)
            xg = None if xg is None else np.where(keep[:, None], xg, 0.)
            xl = None if xl is None else np.where(keep[:, None], xl, 0.)
            zg = np.where(keep[:, None], zg, 0.)
        if self.glob:
            self.glob.update(xg, z, keep)
        if self.loc:
            self.loc.update(xl, (z - zg)[:, :self.local], keep)
        self.n_skipped += int((~keep).sum())
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
        st = {"basis": self.u, "component_ids": self.ids, "blend": [self.blend.history, self.blend.sums]}
        if self.glob:
            st["global_ridge"] = [self.glob.gram, self.glob.rhs]
        if self.loc:
            st["local_ridge"] = [self.loc.gram, self.loc.rhs]
        if self.scale is not None:
            st["guard_scale"] = self.scale
        st["pending_issue_internals"] = [[a for a in p if a is not None] + [c] for p, c in self.pending.values()]
        return st

    def their_state_bytes(self) -> int:
        """Their `Corrector.state_bytes` convention (ridges, basis, ids, guard scale, blend); summed over phases."""
        if self.subs is not None:
            return sum(sub.their_state_bytes() for sub in self.subs)
        s = self.u.nbytes + 8 * GUARD_COMPONENTS + self.blend.history.nbytes + self.blend.sums.nbytes
        s += sum(r.gram.nbytes + r.rhs.nbytes for r in (self.glob, self.loc) if r)
        return s + (self.scale.nbytes if self.scale is not None else 0)

    def extra_metrics(self):
        if self.subs is not None:
            ms = [sub.extra_metrics() for sub in self.subs]
            return {"n_updates": sum(m["n_updates"] for m in ms),
                    "skipped_channel_updates": sum(m["skipped_channel_updates"] for m in ms),
                    "blend_skips": sum(m["blend_skips"] for m in ms),
                    "blend_weight_mean": float(np.mean([m["blend_weight"] for m in ms])),
                    "their_state_bytes": self.their_state_bytes(), "phases": len(self.subs), "lookback": ms[0]["lookback"]}
        return {"n_updates": self.n_updates, "skipped_channel_updates": self.n_skipped,
                "blend_skips": self.n_blend_skips, "blend_weight": self.blend.weight(),
                "their_state_bytes": self.their_state_bytes(), "stride": self.stride, "lookback": self.lb}


def make_dric(variant: str = "l8", **kw) -> DRiC:
    return DRiC(variant=variant, **kw)


__all__ = ["DRiC", "dct_basis", "GUARD_COMPONENTS", "make_dric"]
