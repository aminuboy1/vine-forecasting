"""VINE: learn from forecast REVISIONS under truthful label delay (paper 2).

Setting (methodology/paper2_online_tsf_methodology.md, Sec. 2-3). A frozen backbone issues yhat_t (H, C) at every
origin t. The deployed forecast is yhat_t + c_t with a linear correction c_t[:, ch] = W[ch] phi_t[ch] (W: H x p per
channel, phi: p features per channel, all G_t-measurable: inputs, the backbone's own forecasts, released labels).

Information model. Label blocks Y_s = y[s+1..s+H] are released at s + d (d >= H). Hence every single value y_tau is
released by time tau + l with POINT LAG l = d - H (the harness exposes this as HistoryView.label_limit = t - l).
Matured-block learners wait d steps for a whole block; the information actually available is one new value per step,
l steps old.

Two learning channels, solved jointly as an exact least-squares fixed point:
  * revision channel (label-free, one step later). For h >= 2 the target of origin s is the CORRECTED next forecast
        z_{s,h} = yhat_{s+1,h-1} + w_{h-1} phi_{s+1};   residual form:  rho_{s+1,h} + w_{h-1} phi_{s+1},
        rho_{s+1,h} = yhat_{s+1,h-1} - yhat_{s,h}  (the revision).
    Normal equations  w_h C = Q_h + w_{h-1} D  with exponentially weighted (gamma) moments
        C = sum phi_s phi_s^T,   D = sum phi_{s+1} phi_s^T,   Q_h = sum rho_{s+1,h} phi_s^T   (+ ridge eps I on C).
  * anchor channel (labels). Row 1 regresses the 1-step backbone residual e_{s,1} = y_{s+1} - yhat_{s,1} on phi_s,
    learned as soon as y_{s+1} is released: at time s + 1 + l (anchor="lag", default), or with the whole block at
    s + d (anchor="block", the methodology's original, more conservative variant). Only this row sees labels.
  Solve (theory: methodology/paper2_theory_revised.md, Lemma S and Prop. 1):
        w_1 = r_a (C_a + eps I)^{-1},     w_h = Q_h (C + eps I)^{-1} + w_{h-1} K M,    M = D C^{-1},
  i.e. the ridge penalises each row's INNOVATION w_h - w_{h-1} K M (it does not compound over horizons);
  K = diag(1, kappa, ..., kappa) is the propagation discount (kappa = 1: consistent fixed point; kappa = 0: DSOF-style
  teacher target; tuned on validation, a TD(lambda)-like bias-variance knob) and leaves the intercept undamped because
  the non-intercept features are centred online; the singular values of the whitened transition L^{-1} D L^{-T}
  (C = L L^T) are clipped individually at rho_max = 1.02 at every solve; propagation starts after 2p revision samples.
  All horizons at once by a log-depth prefix scan.
  NOTE: VINELambda (below) still uses the earlier compounding-ridge solve without K or the stabiliser (ablation only).

Baselines on the SAME features (isolate revision learning from the features):
  * MC-LS ("mc_ls", lam = 0): every row regresses the matured block residual e_{s,h} on phi_s, released at s + d.
  * MC-LS-PR ("mc_ls_pr"): per-row maturity at the point lag; row h of origin s is learned at s + h + l, i.e. as soon
    as y_{s+h} is released (the strongest label-only linear learner; EFC/partial-error style).
  The lambda-family blends the bootstrapped and block-matured equations for rows >= 2 (ablation A1):
        w_h [lam C + (1-lam) C_m] = lam (Q_h + w_{h-1} D) + (1-lam) R_{m,h}.
  `target="teacher"` drops the w_{h-1} D term (DSOF-style bootstrapping from the UNcorrected next forecast; ablation
  A2, inconsistent by Proposition 1(iii)).

Causality: revisions and features use inputs and frozen forecasts issued at origins <= now; labels are read only
through the guarded HistoryView (<= label_limit) or `info.matured`. The future-perturbation leak test covers it.

`SafeMix` ("safe_mix"): delayed exponentially weighted mixture with fixed share over expert correctors (Sec. 3.5),
one weight vector per horizon block, updated on matured block losses.

Statistics are float64 and vectorised over channels (VINE-L). `pool=True` shares one set of moments and one W
across channels (VINE-G), more sample-efficient on wide streams.
"""
from __future__ import annotations

import numpy as np

from ..base import OnlineContext, OnlineMethod
from ..protocol import RevealedInfo


def revision(ctx: OnlineContext, t: int) -> np.ndarray | None:
    """rho_{t,h} = yhat_{t,h-1} - yhat_{t-1,h} for h = 2..H, as an (H-1, C) array (label-free).

    Row k (0-based) is the revision of target t+k between origins t-1 and t. None at the first origin.
    """
    if t - 1 < ctx.first_origin:
        return None
    cur, prev = ctx.base_forecast(t), ctx.base_forecast(t - 1)
    return cur[:-1] - prev[1:]


def dct_basis(n: int, k: int) -> np.ndarray:
    """Orthonormal DCT-II basis, first k rows, shape (k, n)."""
    k = min(k, n)
    if k <= 0:
        return np.zeros((0, n))
    i = np.arange(n)
    B = np.cos(np.pi * (i[None, :] + 0.5) * np.arange(k)[:, None] / n) * np.sqrt(2.0 / n)
    B[0] /= np.sqrt(2.0)
    return B


class VINEFeatures:
    """Per-channel feature map phi_t (p features), G_t-measurable.

    [1 | DCT_{K_f}(yhat_t - x_t) | DCT_{K_r}(revision_t) | iota_t | EWMA_a(iota) for a in rates |
     x_t - mean_S(x) | x_{t+1-S} - mean_S(x)]   with iota_t = x_t - yhat_{t-1,1} (input innovation).
    """

    def __init__(self, H: int, C: int, k_f: int = 3, k_r: int = 3, rates=(0.5, 0.95), season: int = 24,
                 use_level: bool = True, use_revision: bool = True, use_ewma: bool = True, k_lb: int = 0,
                 n_lb: int = 2, L: int = 96, use_last: bool = False):
        self.H, self.C = H, C
        # lookback-block shapes (DRiC-style information): DCT of the last n_lb blocks of the input window, each
        # minus x_t; block length min(H, L // n_lb)
        self.n_lb = n_lb if k_lb > 0 else 0
        self.blen = max(2, min(H, L // max(n_lb, 1)))
        self.Blb = dct_basis(self.blen, k_lb) if k_lb > 0 else np.zeros((0, self.blen))
        self.use_last = use_last
        self.Bf = dct_basis(H, k_f)
        self.Br = dct_basis(max(H - 1, 1), k_r) if (use_revision and H > 1) else np.zeros((0, max(H - 1, 1)))
        self.rates = np.asarray(rates if use_ewma else (), float)
        self.season, self.use_level = int(season), use_level
        self.p = (1 + self.Bf.shape[0] + self.Br.shape[0] + 1 + len(self.rates) + (2 if use_level else 0)
                  + self.n_lb * self.Blb.shape[0] + int(use_last))
        self.ewma = np.zeros((len(self.rates), C))
        self.names = (["bias"] + [f"shape{k}" for k in range(self.Bf.shape[0])]
                      + [f"rev{k}" for k in range(self.Br.shape[0])] + ["innov"]
                      + [f"ewma{a}" for a in self.rates] + (["level", "snaive"] if use_level else [])
                      + [f"lb{b}_{k}" for b in range(self.n_lb) for k in range(self.Blb.shape[0])]
                      + (["last"] if use_last else []))

    def __call__(self, x: np.ndarray, yhat: np.ndarray, yhat_prev: np.ndarray | None) -> np.ndarray:
        """x (L, C) input window ending at t, yhat (H, C) forecast at t, yhat_prev (H, C) forecast at t-1 or None."""
        C = self.C
        phi = np.empty((C, self.p))
        j = 0
        phi[:, j] = 1.0
        j += 1
        xt = x[-1].astype(np.float64)
        kf = self.Bf.shape[0]
        phi[:, j:j + kf] = (self.Bf @ (yhat - xt[None, :])).T
        j += kf
        kr = self.Br.shape[0]
        if yhat_prev is not None:
            if kr:
                phi[:, j:j + kr] = (self.Br @ (yhat[:-1] - yhat_prev[1:])).T
            innov = xt - yhat_prev[0]
        else:
            if kr:
                phi[:, j:j + kr] = 0.0
            innov = np.zeros(C)
        j += kr
        phi[:, j] = innov
        j += 1
        if len(self.rates):
            iv = np.where(np.isfinite(innov), innov, 0.0)
            self.ewma = self.rates[:, None] * self.ewma + (1.0 - self.rates[:, None]) * iv[None, :]
            phi[:, j:j + len(self.rates)] = self.ewma.T
            j += len(self.rates)
        if self.use_level:
            S = max(1, min(self.season, x.shape[0]))
            m = x[-S:].mean(0)
            phi[:, j] = xt - m
            phi[:, j + 1] = x[-S] - m          # value one season before the first target, minus the daily mean
            j += 2
        k = self.Blb.shape[0]
        for b in range(self.n_lb):
            blk = x[x.shape[0] - (b + 1) * self.blen: x.shape[0] - b * self.blen].astype(np.float64)
            phi[:, j:j + k] = (self.Blb @ (blk - xt[None, :])).T
            j += k
        if self.use_last:
            phi[:, j] = xt
            j += 1
        return phi


def _whitened_clip(D: np.ndarray, A: np.ndarray, Ainv: np.ndarray, rho_max: float | None):
    """M = D A^{-1}, rescaled so that the whitened transition T = L^{-1} D L^{-T} (A = L L^T) has 2-norm <= rho_max.

    With u_h = w_h L the recursion w_h = G_h + w_{h-1} M reads u_h = G_h L + u_{h-1} T, so ||T||_2 is the per-horizon
    gain in prediction norm (the empirical top canonical correlation of consecutive features, <= ~1).
    """
    M = D @ Ainv
    Lc = np.linalg.cholesky(A)
    Linv = np.linalg.inv(Lc)
    T = Linv @ D @ np.swapaxes(Linv, -1, -2)
    nrm = np.linalg.norm(T, ord=2, axis=(-2, -1))
    if rho_max is None:
        return M, nrm
    scale = np.minimum(1.0, rho_max / np.maximum(nrm, 1e-12))
    return M * scale[:, None, None], nrm


def stable_transition(D: np.ndarray, A: np.ndarray, rho_max: float | None):
    """M = D A^{-1} with the singular values of the whitened transition T = L^{-1} D L^{-T} (A = L L^T) clipped
    INDIVIDUALLY at rho_max:  T' = U min(S, rho_max) V^T,  M = L T' L^{-1}.

    Only explosive directions are touched (rho_max = 1 keeps the near-unit intercept/slow-EWMA directions intact, so a
    constant bias propagates across horizons undamped). Returns (M, raw top singular value, #clipped).
    """
    Lc = np.linalg.cholesky(A)
    Linv = np.linalg.inv(Lc)
    T = Linv @ D @ np.swapaxes(Linv, -1, -2)
    U, S, Vt = np.linalg.svd(T)
    top = S[:, 0]
    if rho_max is not None:
        clipped = (S > rho_max).sum(1)
        T = (U * np.minimum(S, rho_max)[:, None, :]) @ Vt
    else:
        clipped = np.zeros(len(S), int)
    return Lc @ T @ Linv, top, clipped


def prefix_scan(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Solve x_h = a_h + x_{h-1} M (x_1 = a_1) for all h at once: Hillis-Steele scan, ceil(log2 H) batched products.

    Y (B, H, p) holds a_1..a_H on input; M (B, p, p). Returns x (B, H, p).
    """
    Y = Y.copy()
    P = M
    s, H = 1, Y.shape[1]
    while s < H:
        Y[:, s:] = Y[:, s:] + np.einsum("bhi,bij->bhj", Y[:, :-s], P)
        s *= 2
        if s < H:
            P = P @ P
    return Y


class _CorrectorBase(OnlineMethod):
    """Shared plumbing: features, guards, clipping, pending queues, state accounting."""

    def __init__(self, ridge: float = 1.0, k_f: int = 3, k_r: int = 3, pool: bool = False, clip: float = 3.0,
                 guard: float = 6.0, rates=(0.5, 0.95), use_level: bool = True, use_revision_feat: bool = True,
                 use_ewma: bool = True, season: int | None = None, warmup: int = 50, solve_every: int = 1,
                 k_lb: int = 0, n_lb: int = 2, use_last: bool = False, gate: int = 0, k_h: int = 0, **kw):
        super().__init__(**kw)
        self.k_h = int(k_h)        # >0: project the correction onto the first k_h horizon-DCT components
        self.k_lb, self.n_lb, self.use_last, self.gate = k_lb, n_lb, use_last, int(gate)
        self.ridge, self.k_f, self.k_r, self.pool, self.clip, self.guard = ridge, k_f, k_r, pool, clip, guard
        self.rates, self.use_level, self.use_rev, self.use_ewma = tuple(rates), use_level, use_revision_feat, use_ewma
        self.season, self.warmup, self.solve_every = season, int(warmup), max(1, int(solve_every))

    def setup(self, ctx: OnlineContext) -> None:
        super().setup(ctx)
        season = self.season or int((ctx.pretrain_info or {}).get("steps_per_day")
                                    or (ctx.extra or {}).get("steps_per_day", 24))
        self.feat = VINEFeatures(ctx.H, ctx.C, self.k_f, self.k_r, self.rates, season, self.use_level, self.use_rev,
                                 self.use_ewma, self.k_lb, self.n_lb, ctx.L, self.use_last)
        # optional shrinkage gate towards the frozen forecast (DRiC-style): alpha = clip(sum<c,e>/sum|c|^2, 0, 1)
        # over the last `gate` matured origins (labels used only once released: block at s + d)
        self.Ph = None
        if 0 < self.k_h < ctx.H:
            U = dct_basis(ctx.H, self.k_h)
            self.Ph = U.T @ U                                 # (H, H) orthogonal projector onto smooth corrections
        self.gate_hist: list = []
        self.gate_pending: dict[int, np.ndarray] = {}
        self.alpha = 1.0
        self.p = self.feat.p
        self.B = 1 if self.pool else ctx.C
        self.W = np.zeros((ctx.C, ctx.H, self.p))
        self.prev_phi = self.prev_yhat = None
        self.scale = np.ones(ctx.C)                    # EWMA RMS of the input innovation (guards / clipping)
        self.n_steps = self.n_solve = 0
        self.t_last_solve = None

    def _agg(self, outer: np.ndarray) -> np.ndarray:
        """(C, ...) per-channel contributions -> (B, ...) statistic increments."""
        return outer.sum(0, keepdims=True) if self.pool else outer

    def _guard_mask(self, v: np.ndarray) -> np.ndarray:
        """Channels whose value v (C,) is finite and not an outlier relative to the running innovation scale."""
        ok = np.isfinite(v)
        if self.guard and self.n_steps > self.warmup:
            ok &= np.abs(np.nan_to_num(v)) <= self.guard * self.scale
        return ok

    def _update_scale(self, innov):
        a = 0.99
        self.scale = np.maximum(np.sqrt(a * self.scale ** 2 + (1 - a) * np.nan_to_num(innov) ** 2), 1e-3)

    def _correct(self, yhat, phi, t=None):
        corr = np.einsum("chp,cp->hc", self.W, np.nan_to_num(phi))
        if self.Ph is not None:
            corr = self.Ph @ corr
        if self.clip:
            lim = self.clip * self.scale[None, :]
            corr = np.clip(corr, -lim, lim)
        if self.gate:
            if t is not None:
                self.gate_pending[t] = corr.astype(np.float32)
            corr = self.alpha * corr
        return (yhat + corr).astype(np.float32)

    def _gate_update(self, info):
        """Matured block of origin s (released at s + d): accumulate <c_s, e_s> and |c_s|^2 (window `gate`)."""
        if not self.gate:
            return
        for m in info.matured:
            c = self.gate_pending.pop(m.origin, None)
            if c is None:
                continue
            e = np.nan_to_num((m.y - m.base_pred).astype(np.float64))
            self.gate_hist.append((float((c * e).sum()), float((c.astype(np.float64) ** 2).sum())))
            if len(self.gate_hist) > self.gate:
                self.gate_hist.pop(0)
            b, a = (sum(v) for v in zip(*self.gate_hist))
            self.alpha = float(np.clip(b / a, 0.0, 1.0)) if a > 0 else 1.0

    def _maybe_solve(self, t):
        if self.t_last_solve is None or t - self.t_last_solve >= self.solve_every:
            self._solve()
            self.t_last_solve = t
            self.n_solve += 1

    def _set_W(self, W):
        self.W = np.broadcast_to(W, (self.ctx.C,) + W.shape[1:]).copy() if self.pool else W


class VINE(_CorrectorBase):
    """VINE-L (per channel) / VINE-G (pool=True). lam=1: VINE; lam=0: MC-LS; target='teacher': DSOF-style target."""
    name = "vine"

    def __init__(self, gamma: float = 0.995, gamma_a: float = 0.995, rho_max: float | None = 1.02,
                 lam: float = 1.0, target: str = "corrected", anchor: str = "lag", gamma_m: float = 0.999,
                 norm_every: int = 25, kappa: float = 0.9, center: bool = True, **kw):
        super().__init__(**kw)
        if target not in ("corrected", "teacher"):
            raise ValueError("target must be 'corrected' or 'teacher'")
        if anchor not in ("lag", "block"):
            raise ValueError("anchor must be 'lag' (released value at point lag l) or 'block' (whole block at s+d)")
        self.gamma, self.gamma_a, self.gamma_m, self.rho_max = gamma, gamma_a, gamma_m, rho_max
        self.lam, self.target, self.anchor = float(lam), target, anchor
        self.norm_every = max(1, int(norm_every))     # (unused since the per-solve stabiliser; kept for old configs)
        # propagation discount: w_h = G_h + kappa w_{h-1} M. kappa = 1: consistent fixed point (Prop. 1);
        # kappa = 0: DSOF-style teacher target (inconsistent, Prop. 1(iii)); 0 < kappa < 1 trades bias for variance
        # exactly like lambda in TD(lambda). Tuned on validation.
        self.kappa = float(kappa)
        self.center = bool(center)

    def setup(self, ctx: OnlineContext) -> None:
        super().setup(ctx)
        H, p, B = ctx.H, self.p, self.B
        z = lambda *s: np.zeros(s)                     # noqa: E731
        self.Cr, self.D, self.Q = z(B, p, p), z(B, p, p), z(B, max(H - 1, 0), p)   # revision channel
        self.Ca, self.ra = z(B, p, p), z(B, p)                                     # anchor (row 1)
        self.Cm, self.Rm = z(B, p, p), z(B, H, p)                                  # block-matured (lam < 1)
        self.needs_block = self.lam < 1.0 or self.anchor == "block"
        self.needs_lag = self.lam > 0.0 and self.anchor == "lag"
        self.pending_phi: dict[int, np.ndarray] = {}
        self.pending_y1: dict[int, np.ndarray] = {}
        self.next_anchor = ctx.first_origin
        self.n_rev = self.n_anchor = self.n_block = 0
        self.diag_norm, self.diag_clip = [], []
        self.fmean, self.fmean_w = np.zeros((ctx.C, p)), 0.0
        self.pending_anchor: dict[int, tuple] = {}     # (phi_s, yhat_{s,1}) for the point-lag anchor (own queue)

    # ------------------------------------------------------------------ statistics
    def _add_anchor(self, phi, e1):
        ok = self._guard_mask(e1) & np.isfinite(phi).all(1)
        phi_ok, e_ok = np.where(ok[:, None], phi, 0.0), np.where(ok, e1, 0.0)
        g = self.gamma_a
        self.Ca *= g
        self.ra *= g
        self.Ca += self._agg(np.einsum("ci,cj->cij", phi_ok, phi_ok))
        self.ra += self._agg(e_ok[:, None] * phi_ok)
        self.n_anchor += 1

    def _consume_lag_anchors(self, t):
        """Anchor updates for every origin s < t whose 1-step target y_{s+1} is released (s + 1 <= label_limit)."""
        lim = self.ctx.history.label_limit()
        while self.next_anchor < t and self.next_anchor + 1 <= lim:
            s = self.next_anchor
            y1 = self.ctx.history.labels(s + 1, s + 2)[0].astype(np.float64)
            phi, yh1 = self.pending_anchor.pop(s)
            self._add_anchor(phi, y1 - yh1)
            self.next_anchor += 1

    def update(self, info: RevealedInfo) -> None:
        self._gate_update(info)
        if not self.needs_block:
            return
        g = self.gamma_m
        for m in info.matured:
            phi = self.pending_phi.pop(m.origin, None)
            if phi is None:
                continue
            e = (m.y - m.base_pred).astype(np.float64)                                   # (H, C) backbone residual
            if self.anchor == "block" and self.lam > 0:
                self.pending_y1.pop(m.origin, None)
                self._add_anchor(phi, e[0])
            if self.lam < 1.0:
                ok = self._guard_mask(e[0]) & np.isfinite(e).all(0) & np.isfinite(phi).all(1)
                phi_ok, e_ok = np.where(ok[:, None], phi, 0.0), np.where(ok[None, :], e, 0.0)
                self.Cm *= g
                self.Rm *= g
                self.Cm += self._agg(np.einsum("ci,cj->cij", phi_ok, phi_ok))
                self.Rm += self._agg(np.einsum("hc,ci->chi", e_ok, phi_ok))
            self.n_block += 1

    def _solve(self):
        H, p, lam = self.ctx.H, self.p, self.lam
        I = self.ridge * np.eye(p)[None]
        W = np.zeros((self.B, H, p))
        if lam <= 0.0:                                                                   # MC-LS
            W[:] = np.swapaxes(np.linalg.solve(self.Cm + I, np.swapaxes(self.Rm, -1, -2)), -1, -2)
            self._set_W(W)
            return
        W[:, 0] = np.linalg.solve(self.Ca + I, self.ra[..., None])[..., 0]
        if H > 1:
            A = self.Cr + I if lam >= 1.0 else lam * self.Cr + (1.0 - lam) * self.Cm + I
            Ainv = np.linalg.inv(A)
            R = lam * self.Q if lam >= 1.0 else lam * self.Q + (1.0 - lam) * self.Rm[:, 1:]
            G = R @ Ainv                                                                 # (B, H-1, p)
            if self.target == "corrected" and self.n_rev >= 2 * p:   # propagate only once D is estimable
                # Ridge acts on each row's NEW information: penalty eps |w_h - w_{h-1} M|^2 gives
                #   w_h = Q_h (C + eps I)^{-1} + w_{h-1} D C^{-1},
                # so the transition uses (numerically regularised) C and the ridge does not compound over horizons.
                A0 = A - I
                tr = np.trace(A0, axis1=-2, axis2=-1)[:, None, None] / p
                A0 = A0 + (1e-6 * tr + 1e-9) * np.eye(p)[None]
                M, top, clipped = stable_transition(lam * self.D, A0, self.rho_max)   # refreshed at EVERY solve
                if len(self.diag_norm) < 20000:
                    self.diag_norm.append(float(np.median(top)))
                    self.diag_clip.append(float(np.mean(clipped > 0)))
                W[:, 1:] = G
                # kappa discounts only the stochastic feature directions; the intercept maps onto itself exactly
                # (row 0 of D equals row 0 of C), so a constant bias propagates undamped at no variance cost
                K = np.full(p, self.kappa)
                K[0] = 1.0
                W = prefix_scan(W, K[None, :, None] * M)   # w_h = G_h + w_{h-1} K M for all h, O(log H) products
            else:
                W[:, 1:] = G
        self._set_W(W)

    # ------------------------------------------------------------------ protocol
    def predict(self, x: np.ndarray, base_pred: np.ndarray, t: int) -> np.ndarray:
        yhat = base_pred.astype(np.float64)
        if self.needs_lag:
            self._consume_lag_anchors(t)
        phi = self.feat(x, yhat, self.prev_yhat)                                          # (C, p)
        if self.center:
            # centre the non-intercept features with a running (gamma) mean, so all mean structure sits in the
            # intercept, which the propagation discount kappa leaves undamped (no bias decay over horizons)
            fin = np.isfinite(phi).all(1)
            self.fmean[fin] = self.gamma * self.fmean[fin] + (1.0 - self.gamma) * phi[fin]
            self.fmean_w = self.gamma * self.fmean_w + (1.0 - self.gamma)
            phi = phi.copy()
            phi[:, 1:] -= (self.fmean[:, 1:] / max(self.fmean_w, 1e-12))
        if self.prev_yhat is not None:
            innov = x[-1].astype(np.float64) - self.prev_yhat[0]
            if self.lam > 0 and self.ctx.H > 1:                                           # revision channel
                ok = self._guard_mask(innov) & np.isfinite(phi).all(1) & np.isfinite(self.prev_phi).all(1)
                pp, pn = np.where(ok[:, None], self.prev_phi, 0.0), np.where(ok[:, None], phi, 0.0)
                rho = np.where(ok[None, :], yhat[:-1] - self.prev_yhat[1:], 0.0)        # (H-1, C)
                g = self.gamma
                self.Cr *= g
                self.D *= g
                self.Q *= g
                self.Cr += self._agg(np.einsum("ci,cj->cij", pp, pp))
                self.D += self._agg(np.einsum("ci,cj->cij", pn, pp))
                self.Q += self._agg(np.einsum("hc,ci->chi", rho, pp))
                self.n_rev += 1
            self._update_scale(innov)
        self.n_steps += 1
        self._maybe_solve(t)
        out = self._correct(yhat, phi, t)
        if self.needs_block:
            self.pending_phi[t] = phi
        if self.needs_lag:
            self.pending_anchor[t] = (phi, yhat[0].copy())
        elif self.lam > 0:                       # block anchor
            self.pending_y1[t] = yhat[0].copy()
        self.prev_phi, self.prev_yhat = phi, yhat
        return out

    # ------------------------------------------------------------------ accounting
    def state_breakdown(self):
        out = {"W": self.W, "feat_ewma": self.feat.ewma, "scale": self.scale}
        if self.lam > 0:
            out.update({"C_anchor": self.Ca, "r_anchor": self.ra})
            if self.ctx.H > 1:
                out.update({"C_rev": self.Cr, "D": self.D, "Q": self.Q})
        if self.lam < 1:
            out.update({"C_mat": self.Cm, "R_mat": self.Rm})
        out["pending_phi"] = list(self.pending_phi.values())
        out["pending_y1"] = list(self.pending_y1.values())
        out["pending_anchor"] = [a for pair in self.pending_anchor.values() for a in pair]
        if self.gate:
            out["pending_gate_corr"] = list(self.gate_pending.values())
        if self.prev_phi is not None:
            out["prev_phi"], out["prev_yhat"] = self.prev_phi, self.prev_yhat
        return out

    def extra_metrics(self):
        return {"n_rev_updates": self.n_rev, "n_anchor_updates": self.n_anchor, "n_block_updates": self.n_block,
                "n_solves": self.n_solve, "p": self.p,
                "whitened_norm_median": float(np.median(self.diag_norm)) if self.diag_norm else None,
                "frac_solves_clipped": float(np.mean(self.diag_clip)) if self.diag_clip else None}


class MCLS(VINE):
    """Matured-only least squares on VINE's features (lam = 0): every row learns at s + d from the released block."""
    name = "mc_ls"

    def __init__(self, gamma_m: float = 0.995, **kw):
        kw["lam"] = 0.0
        super().__init__(gamma_m=gamma_m, **kw)


class MCLSPerRow(_CorrectorBase):
    """Label-only least squares with PER-ROW maturity at the point lag l: row h of origin s is learned at s + h + l,
    the earliest time y_{s+h} is released (partial-error / EFC-style use of the label stream). Same features as VINE.
    """
    name = "mc_ls_pr"

    def __init__(self, gamma_m: float = 0.995, **kw):
        super().__init__(**kw)
        self.gamma_m = gamma_m

    def setup(self, ctx):
        super().setup(ctx)
        H, p, B = ctx.H, self.p, self.B
        self.Ch = np.zeros((B, H, p, p))
        self.Rh = np.zeros((B, H, p))
        self.pending_phi: dict[int, np.ndarray] = {}
        self.pending_yhat: dict[int, np.ndarray] = {}
        self.last_lim = None
        self.n_upd = 0

    def _ingest(self, lim):
        """New released value y_lim: one sample per row h from origin lim - h (if that origin was issued)."""
        H, C = self.ctx.H, self.ctx.C
        y = self.ctx.history.labels(lim, lim + 1)[0].astype(np.float64)
        ok_y = np.isfinite(y)
        g = self.gamma_m
        self.Ch *= g
        self.Rh *= g
        phis = np.zeros((H, C, self.p))
        errs = np.zeros((H, C))
        for h in range(1, H + 1):
            s = lim - h
            if s in self.pending_phi:
                phis[h - 1] = self.pending_phi[s]
                errs[h - 1] = y - self.pending_yhat[s][h - 1]
        ok = ok_y[None, :] & np.isfinite(errs) & np.isfinite(phis).all(-1)
        if self.guard and self.n_steps > self.warmup:            # same outlier guard as VINE's label residuals
            ok &= np.abs(np.nan_to_num(errs)) <= self.guard * self.scale[None, :] * np.sqrt(np.arange(1, H + 1))[:, None]
        phis = np.where(ok[..., None], phis, 0.0)
        errs = np.where(ok, errs, 0.0)
        outer = np.einsum("hci,hcj->chij", phis, phis)
        rr = np.einsum("hc,hci->chi", errs, phis)
        self.Ch += self._agg(outer)
        self.Rh += self._agg(rr)
        for s in [s for s in self.pending_phi if s <= lim - H]:
            del self.pending_phi[s], self.pending_yhat[s]
        self.n_upd += 1

    def update(self, info):
        self._gate_update(info)

    def _solve(self):
        I = self.ridge * np.eye(self.p)[None, None]
        W = np.linalg.solve(self.Ch + I, self.Rh[..., None])[..., 0]                    # (B, H, p)
        self._set_W(W)

    def predict(self, x, base_pred, t):
        yhat = base_pred.astype(np.float64)
        lim = min(self.ctx.history.label_limit(), t)
        start = (self.ctx.first_origin + 1) if self.last_lim is None else self.last_lim + 1
        for v in range(max(start, self.ctx.first_origin + 1), lim + 1):
            self._ingest(v)
        self.last_lim = max(lim, self.last_lim or lim)
        phi = self.feat(x, yhat, self.prev_yhat)
        if self.prev_yhat is not None:
            self._update_scale(x[-1].astype(np.float64) - self.prev_yhat[0])
        self.n_steps += 1
        if self.n_upd:
            self._maybe_solve(t)
        out = self._correct(yhat, phi, t)
        self.pending_phi[t], self.pending_yhat[t] = phi, yhat
        self.prev_phi, self.prev_yhat = phi, yhat
        return out

    def state_breakdown(self):
        return {"C_rows": self.Ch, "R_rows": self.Rh, "W": self.W, "feat_ewma": self.feat.ewma, "scale": self.scale,
                "pending_phi": list(self.pending_phi.values()), "pending_yhat": list(self.pending_yhat.values()),
                "pending_gate_corr": list(self.gate_pending.values())}


class SafeMix(OnlineMethod):
    """Delayed exponentially weighted mixture with fixed share over expert correctors (Sec. 3.5).

    Experts are built from specs (default: frozen, MC-LS-PR, VINE fast, VINE slow). One weight vector per horizon
    block, shared across channels, updated with each expert's squared loss on the block when origin s matures
    (time s + d). Output: convex combination of the expert forecasts. Pending expert forecasts are stored exactly.
    """
    name = "safe_mix"

    def __init__(self, experts: list | None = None, eta: float = 0.5, share: float = 1e-3,
                 blocks=(6, 24, 48, 96), **kw):
        super().__init__(**kw)
        self.expert_specs = experts or [{"type": "frozen"}, {"type": "mc_ls_pr"},
                                        {"type": "vine", "gamma": 0.99}, {"type": "vine", "gamma": 0.999}]
        self.eta, self.share, self.block_ends = eta, share, tuple(blocks)

    def setup(self, ctx):
        super().setup(ctx)
        from . import build_method
        self.experts = [build_method(dict(s)) for s in self.expert_specs]
        for e in self.experts:
            e.setup(ctx)
        H = ctx.H
        ends = sorted({min(b, H) for b in self.block_ends if b > 0} | {H})
        starts = [0] + ends[:-1]
        self.blocks = [(a, b) for a, b in zip(starts, ends) if b > a]
        self.logw = np.zeros((len(self.blocks), len(self.experts)))
        self.pending: dict[int, np.ndarray] = {}
        self.weight_trace = []

    def _weights(self):
        w = np.exp(self.logw - self.logw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def update(self, info):
        for e in self.experts:
            e.update(info)
        for m in info.matured:
            preds = self.pending.pop(m.origin, None)
            if preds is None:
                continue
            y = m.y.astype(np.float64)
            for b, (a, z) in enumerate(self.blocks):
                loss = ((preds[:, a:z] - y[None, a:z]) ** 2).mean(axis=(1, 2))           # (N,)
                lw = self.logw[b] - self.eta * loss
                w = np.exp(lw - lw.max())
                w /= w.sum()
                w = (1 - self.share) * w + self.share / len(w)                            # fixed share
                self.logw[b] = np.log(w)

    def predict(self, x, base_pred, t):
        preds = np.stack([np.asarray(e.predict(x, base_pred, t), np.float64) for e in self.experts])  # (N, H, C)
        self.pending[t] = preds.astype(np.float32)
        W = self._weights()
        out = np.empty(preds.shape[1:])
        for b, (a, z) in enumerate(self.blocks):
            out[a:z] = np.einsum("n,nhc->hc", W[b], preds[:, a:z])
        if t % 50 == 0:
            self.weight_trace.append(W.copy())
        return out.astype(np.float32)

    def state_breakdown(self):
        out = {"logw": self.logw, "pending_expert_preds": list(self.pending.values())}
        for i, e in enumerate(self.experts):
            for k, v in e.state_breakdown().items():
                out[f"{k}_e{i}" if k.startswith(("pending", "raw")) else f"e{i}_{k}"] = v
        return out

    def extra_metrics(self):
        return {"final_weights": self._weights().tolist(), "experts": [e.name for e in self.experts],
                "mean_weights": np.mean(self.weight_trace, 0).tolist() if self.weight_trace else None}


class VINELambda(VINE):
    """TD(lambda)-style VINE: rows h >= 2 blend the revision (bootstrapped) normal equations with PER-ROW matured
    labels at the point lag (the freshest label information, as in MC-LS-PR):
        w_h [lam C + (1-lam) C_h] = lam (Q_h + w_{h-1} D) + (1-lam) R_h ,   row 1: anchor at the point lag.
    lam = 1 recovers VINE; lam = 0 is MC-LS-PR with the anchor row. lam is tuned on validation (bias of bootstrapping
    vs staleness of labels, Corollary 1). Row-dependent transitions => sequential back-substitution over h.
    """
    name = "vine_lam"

    def __init__(self, lam: float = 0.5, gamma_m: float | None = None, **kw):
        super().__init__(lam=1.0, **kw)          # VINE plumbing (revision + anchor channels) with lam = 1 ...
        self.mix = float(lam)                    # ... and the per-row blend weight kept separately
        self.gamma_pr = gamma_m

    def setup(self, ctx):
        super().setup(ctx)
        H, p, B = ctx.H, self.p, self.B
        self.Ch = np.zeros((B, H, p, p))
        self.Rh = np.zeros((B, H, p))
        self.pr_phi: dict[int, np.ndarray] = {}
        self.pr_yhat: dict[int, np.ndarray] = {}
        self.last_lim = None
        self.g_pr = self.gamma_pr if self.gamma_pr is not None else self.gamma

    def _ingest(self, lim):
        H, C = self.ctx.H, self.ctx.C
        y = self.ctx.history.labels(lim, lim + 1)[0].astype(np.float64)
        phis, errs = np.zeros((H, C, self.p)), np.zeros((H, C))
        for h in range(1, H + 1):
            s = lim - h
            if s in self.pr_phi:
                phis[h - 1], errs[h - 1] = self.pr_phi[s], y - self.pr_yhat[s][h - 1]
        ok = np.isfinite(y)[None, :] & np.isfinite(errs) & np.isfinite(phis).all(-1)
        phis, errs = np.where(ok[..., None], phis, 0.0), np.where(ok, errs, 0.0)
        self.Ch *= self.g_pr
        self.Rh *= self.g_pr
        self.Ch += self._agg(np.einsum("hci,hcj->chij", phis, phis))
        self.Rh += self._agg(np.einsum("hc,hci->chi", errs, phis))
        for s in [s for s in self.pr_phi if s <= lim - H]:
            del self.pr_phi[s], self.pr_yhat[s]

    def _solve(self):
        H, p, mix = self.ctx.H, self.p, self.mix
        I = self.ridge * np.eye(p)[None]
        W = np.zeros((self.B, H, p))
        W[:, 0] = np.linalg.solve(self.Ca + I, self.ra[..., None])[..., 0]
        for h in range(1, H):
            A = mix * self.Cr + (1.0 - mix) * self.Ch[:, h] + I
            rhs = mix * (self.Q[:, h - 1] + np.einsum("bi,bij->bj", W[:, h - 1], self.D)) + (1.0 - mix) * self.Rh[:, h]
            W[:, h] = np.linalg.solve(A, rhs[..., None])[..., 0]
        self._set_W(W)

    def predict(self, x, base_pred, t):
        lim = min(self.ctx.history.label_limit(), t)
        start = (self.ctx.first_origin + 1) if self.last_lim is None else self.last_lim + 1
        for v in range(max(start, self.ctx.first_origin + 1), lim + 1):
            self._ingest(v)
        self.last_lim = max(lim, self.last_lim or lim)
        out = super().predict(x, base_pred, t)
        self.pr_phi[t], self.pr_yhat[t] = self.prev_phi, self.prev_yhat
        return out

    def state_breakdown(self):
        out = super().state_breakdown()
        out.update({"C_rows": self.Ch, "R_rows": self.Rh, "pending_pr_phi": list(self.pr_phi.values()),
                    "pending_pr_yhat": list(self.pr_yhat.values())})
        return out
