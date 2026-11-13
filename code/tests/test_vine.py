"""VINE (paper 2): exact fixed-point solve, telescoping identity, causality, and the synthetic theory checks (A11)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tsl.data.synth_online import UnderReactingBackbone, revision_dgp
from tsl.online import OnlineEvaluator, build_method
from tsl.online.leaktest import future_perturbation_test
from tsl.online.methods.vine import VINE, _whitened_clip, dct_basis

L, H = 48, 24


def test_dct_basis_orthonormal():
    B = dct_basis(24, 24)
    np.testing.assert_allclose(B @ B.T, np.eye(24), atol=1e-12)


def test_telescoping_identity():
    """e_{s,h} = (y_{s+h} - yhat_{s+h-1,1}) + sum_{k=1}^{h-1} rho_{s+k, h-k+1} exactly (any forecaster)."""
    rng = np.random.default_rng(0)
    T, Hh = 60, 6
    y = rng.standard_normal(T)
    yhat = rng.standard_normal((T, Hh))           # yhat[s, h-1] forecasts y[s+h]
    for s in range(5, T - Hh - 1):
        for h in range(1, Hh + 1):
            e = y[s + h] - yhat[s, h - 1]
            tel = y[s + h] - yhat[s + h - 1, 0]
            for k in range(1, h):
                tel += yhat[s + k, h - k - 1] - yhat[s + k - 1, h - k]    # rho_{s+k, h-k+1}
            assert abs(e - tel) < 1e-12


def _random_spd(rng, B, p):
    A = rng.standard_normal((B, p, 3 * p))
    return A @ np.swapaxes(A, -1, -2) / (3 * p) + 0.1 * np.eye(p)


def test_fixed_point_satisfies_normal_equations():
    """Back-substitution solves the stacked Stein system exactly (rho_max off): w_h A = Q_h + w_{h-1} D, row 1 anchor."""
    rng = np.random.default_rng(1)
    B, p, Hh = 3, 7, 10
    m = VINE(ridge=0.0, rho_max=None)

    class Ctx:
        C, H = B, Hh
    m.ctx, m.p, m.B, m.pool, m.lam, m.target = Ctx, p, B, False, 1.0, "corrected"
    m.diag_norm, m.diag_clip, m.n_solve, m.n_rev, m.kappa = [], [], 0, 10 ** 6, 1.0
    m.Cr, m.Ca = _random_spd(rng, B, p), _random_spd(rng, B, p)
    m.D = 0.3 * rng.standard_normal((B, p, p))
    m.Q, m.ra = rng.standard_normal((B, Hh - 1, p)), rng.standard_normal((B, p))
    m._solve()
    W = m.W
    np.testing.assert_allclose(np.einsum("bi,bij->bj", W[:, 0], m.Ca), m.ra, atol=1e-9)
    for h in range(1, Hh):
        lhs = np.einsum("bi,bij->bj", W[:, h], m.Cr)
        rhs = m.Q[:, h - 1] + np.einsum("bi,bij->bj", W[:, h - 1], m.D)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-5)


def test_whitened_norm_is_canonical_correlation():
    """For a stationary VAR(1) feature process the whitened transition norm is <= 1 (approximately)."""
    rng = np.random.default_rng(2)
    p, n = 4, 20000
    A = np.diag([0.9, 0.5, -0.3, 0.0])
    f = np.zeros((n, p))
    for t in range(1, n):
        f[t] = f[t - 1] @ A.T + rng.standard_normal(p)
    C = (f[:-1].T @ f[:-1])[None]
    D = (f[1:].T @ f[:-1])[None]
    _, nrm = _whitened_clip(D, C, np.linalg.inv(C), None)
    assert 0.85 < nrm[0] < 0.95


@pytest.mark.parametrize("spec", [{"type": "vine"}, {"type": "mc_ls_pr"}, {"type": "safe_mix"}])
@pytest.mark.parametrize("dfac", [1, 3])
def test_vine_family_passes_future_perturbation_two_channel(spec, dfac):
    y, _ = revision_dgp(T=900, C=2, regime_len=100, seed=3)
    x = y + 0.05 * np.random.default_rng(4).standard_normal(y.shape).astype(np.float32)   # provisional inputs
    ok, rep = future_perturbation_test(lambda: build_method(dict(spec)), UnderReactingBackbone(L, H), y, 300, L, H,
                                       delay=dfac * H, x=x, t0_offsets=(60, 250))
    assert ok, rep


def _synthetic_scores(d, regime_len=400, T=6000):
    y, _ = revision_dgp(T=T, C=3, regime_len=regime_len, seed=0)
    bb = UnderReactingBackbone(L, H)
    ev = OnlineEvaluator(y, 1200, None, L, H, delay=d)
    arch = ev.make_archive(bb)
    out = {}
    for spec in ({"type": "frozen"}, {"type": "mc_ls"}, {"type": "mc_ls_pr"}, {"type": "vine"},
                 {"type": "vine", "target": "teacher", "name": "teacher"}):
        r = ev.run(build_method(spec), bb, archive=arch)
        out[r.method] = float(r.se.mean())
    return out


def test_synthetic_theory_predictions():
    """A11: revision learning is (i) insensitive to the delay, (ii) better than label-only learners at long delay,
    (iii) better than teacher bootstrapping (Prop. 1(iii)), and (iv) better than frozen."""
    short, long_ = _synthetic_scores(H), _synthetic_scores(8 * H)
    # H5: VINE's sensitivity to the delay is at most half of the matured-only learner's
    assert abs(long_["vine"] - short["vine"]) <= 0.5 * (long_["mc_ls"] - short["mc_ls"])
    assert long_["vine"] < long_["mc_ls"] and long_["vine"] < long_["mc_ls_pr"]
    assert long_["mc_ls"] > short["mc_ls"]                       # label-only learners degrade with delay
    assert long_["vine"] < long_["teacher"]                     # consistent target beats teacher bootstrapping
    assert long_["vine"] < long_["frozen"]


def test_prefix_scan_matches_sequential_recursion():
    from tsl.online.methods.vine import prefix_scan
    rng = np.random.default_rng(5)
    for Hh in (1, 2, 7, 24, 96):
        A = rng.standard_normal((3, Hh, 5))
        M = 0.3 * rng.standard_normal((3, 5, 5))
        ref = A.copy()
        for h in range(1, Hh):
            ref[:, h] = A[:, h] + np.einsum("bi,bij->bj", ref[:, h - 1], M)
        np.testing.assert_allclose(prefix_scan(A, M), ref, atol=1e-10)


class _BiasedAR1(torch.nn.Module):
    """yhat_{t,h} = a^h x_t + bias: the AR(1) conditional mean plus a constant bias at every horizon."""

    def __init__(self, Hh, a, bias):
        super().__init__()
        self.register_buffer("pw", torch.tensor([a ** h for h in range(1, Hh + 1)], dtype=torch.float32))
        self.bias = bias

    def forward(self, x, **_):
        return self.pw[None, :, None] * x[:, -1:, :] + self.bias


@pytest.mark.parametrize("Hh", [24, 96])
def test_constant_bias_propagates_to_long_horizons(Hh):
    """Reviewer C1 regression: a constant backbone bias must be removed at EVERY horizon (the stabiliser must not
    damp the intercept direction as 0.98^(h-1))."""
    rng = np.random.default_rng(0)
    T, a = 9000, 0.9
    y = np.zeros((T, 1), np.float32)
    for t in range(1, T):
        y[t] = a * y[t - 1] + rng.standard_normal()
    bb = _BiasedAR1(Hh, a, 0.5)
    ev = OnlineEvaluator(y, 200, None, 48, Hh, delay=Hh, keep_preds=True)
    arch = ev.make_archive(bb)
    base = arch.all()
    corr = {}
    for name in ("vine", "mc_ls_pr"):
        r = ev.run(build_method({"type": name}), bb, archive=arch)
        corr[name] = (r.preds - base)[len(base) // 2:, :, 0].mean(0)
    # no propagation damping: VINE removes the bias at the longest horizon as well as the label-only learner does
    assert abs(corr["vine"][-1] - corr["mc_ls_pr"][-1]) < 0.08, (corr["vine"][[0, -1]], corr["mc_ls_pr"][[0, -1]])
    assert corr["vine"][-1] < -0.3 and abs(corr["vine"][0] + 0.5) < 0.05
