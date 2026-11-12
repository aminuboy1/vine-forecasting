"""Proofs-by-test that the online protocol does not leak future values.

Main test (model-agnostic causality): run every method on two streams that are identical up to row t0
and differ afterwards (future replaced by NaN). Every forecast issued at an origin s < t0 must be
bit-identical in both runs. A method that touched any unrevealed value would either change or NaN-out
one of those forecasts. The same test must FAIL (detect the leak) for the FSNet-style leaky mode.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tsl.data.registry import synthetic_dataset
from tsl.models import build_backbone
from tsl.online import LeakageError, OnlineEvaluator, OnlineMethod, build_method

L, H = 24, 8
T = 700
FIRST = 300


def _z():
    return synthetic_dataset(T=T, C=3, seed=1).values


def _backbone():
    torch.manual_seed(0)
    return build_backbone("DLinear", L, H)


METHOD_SPECS = [
    {"type": "frozen"},
    {"type": "rls", "n_lags": 2},
    {"type": "elf", "update_every": 20, "warmup_updates": 1},
    {"type": "elf", "update_every": 20, "warmup_updates": 1, "warm_start_rows": 200, "name": "elf_warm"},
    {"type": "ogd", "scope": "full", "lr": 1e-3, "batch": 4},
    {"type": "ogd", "scope": "head", "lr": 1e-3},
    {"type": "periodic", "period": 50, "epochs": 1, "max_steps": 5},
    {"type": "periodic", "period": 50, "epochs": 1, "max_steps": 5, "window": 150, "name": "periodic_win"},
]


def _run(z, spec, mode="matured", delay=None, allow_leaky=False, last=None):
    ev = OnlineEvaluator(z, FIRST, last, L, H, delay=delay, mode=mode, allow_leaky=allow_leaky,
                         check_leaks=mode != "leaky", keep_preds=True)
    res = ev.run(build_method(spec), _backbone(), device="cpu", seed=0)
    return res


@pytest.mark.parametrize("spec", METHOD_SPECS, ids=lambda s: s.get("name", s["type"] + "_" + str(s.get("scope", ""))))
@pytest.mark.parametrize("mode,delay", [("matured", None), ("matured", 2 * H), ("prefix", None)])
def test_forecasts_before_t0_do_not_depend_on_future(spec, mode, delay):
    z = _z()
    t0 = FIRST + 120
    z_poison = z.copy()
    z_poison[t0:] = np.nan
    last = t0 - 1 + 5  # a few origins beyond t0 are fine (they read NaN inputs, but are not compared)
    if spec["type"] == "ogd" and mode == "prefix":
        spec = {**spec, "use_prefix": True}
    a = _run(z, spec, mode, delay, last=last)
    b = _run(z_poison, spec, mode, delay, last=last)
    n_cmp = t0 - FIRST
    pa, pb = a.preds[:n_cmp], b.preds[:n_cmp]
    assert np.isfinite(pb).all(), "a forecast before t0 used a future (NaN) value"
    np.testing.assert_array_equal(pa, pb)


def test_leaky_mode_is_detected_by_the_causality_test():
    z = _z()
    t0 = FIRST + 120
    z_poison = z.copy()
    z_poison[t0:] = np.nan
    spec = {"type": "ogd", "scope": "full", "lr": 1e-2}
    a = _run(z, spec, "leaky", allow_leaky=True, last=t0 + 5)
    b = _run(z_poison, spec, "leaky", allow_leaky=True, last=t0 + 5)
    n_cmp = t0 - FIRST
    assert not np.isfinite(b.preds[:n_cmp]).all() or not np.array_equal(a.preds[:n_cmp], b.preds[:n_cmp])
    assert a.protocol["leaky"] and a.protocol["max_leak_steps"] == H


def test_leaky_requires_explicit_flag_and_small_delay_rejected():
    z = _z()
    with pytest.raises(ValueError):
        OnlineEvaluator(z, FIRST, None, L, H, mode="leaky")
    with pytest.raises(ValueError):
        OnlineEvaluator(z, FIRST, None, L, H, delay=H - 1)


class _Spy(OnlineMethod):
    name = "spy"
    uses_prefix = True

    def setup(self, ctx):
        super().setup(ctx)
        self.log = []
        self.z_ref = None

    def predict(self, x, base_pred, t):
        return base_pred

    def update(self, info):
        for m in info.matured:
            self.log.append(("m", info.now, m.origin, m.y.copy()))
        for p in info.prefixes:
            self.log.append(("p", info.now, p.origin, p.k))


@pytest.mark.parametrize("delay", [H, H + 5])
def test_labels_released_exactly_at_origin_plus_delay(delay):
    z = _z()
    ev = OnlineEvaluator(z, FIRST, None, L, H, delay=delay, mode="matured")
    spy = _Spy()
    ev.run(spy, _backbone())
    ms = [r for r in spy.log if r[0] == "m"]
    assert len(ms) > 0
    for _, now, o, y in ms:
        assert now == o + delay
        np.testing.assert_array_equal(y, z[o + 1:o + H + 1])
    assert not [r for r in spy.log if r[0] == "p"], "no prefixes may be exposed in matured mode"


def test_prefix_mode_only_exposes_observed_rows():
    z = _z()
    ev = OnlineEvaluator(z, FIRST, FIRST + 60, L, H, delay=2 * H, mode="prefix")
    spy = _Spy()
    ev.run(spy, _backbone())
    ps = [r for r in spy.log if r[0] == "p"]
    assert ps
    for _, now, o, k in ps:
        assert o + k <= now and o < now


class _Cheater(OnlineMethod):
    name = "cheater"

    def __init__(self, when="update", ahead=1):
        super().__init__()
        self.when, self.ahead = when, ahead

    def predict(self, x, base_pred, t):
        if self.when == "predict":
            self.ctx.history.get(0, t + 1 + self.ahead)
        return base_pred

    def update(self, info):
        if self.when == "update":
            self.ctx.history.get(0, info.now + 1 + self.ahead)


@pytest.mark.parametrize("when", ["update", "predict"])
def test_history_access_beyond_revealed_raises(when):
    z = _z()
    ev = OnlineEvaluator(z, FIRST, FIRST + 100, L, H, mode="matured")
    with pytest.raises(LeakageError):
        ev.run(_Cheater(when), _backbone())


def test_matured_mode_update_limit_with_long_delay():
    """With d > H an update may read rows only up to now - d + H."""
    z = _z()
    d = 3 * H
    ev = OnlineEvaluator(z, FIRST, FIRST + 100, L, H, delay=d, mode="matured")
    # reading exactly up to now-d+H is fine...
    class Ok(_Cheater):
        def update(self, info):
            self.ctx.history.get(0, info.now - d + H + 1)
    ev.run(Ok(), _backbone())
    # ...one row more is a leak
    class Bad(_Cheater):
        def update(self, info):
            self.ctx.history.get(0, info.now - d + H + 2)
    with pytest.raises(LeakageError):
        ev.run(Bad(), _backbone())


def test_drop_last_false_counts_every_complete_window():
    z = _z()
    ev = OnlineEvaluator(z, FIRST, None, L, H)
    res = ev.run(build_method({"type": "frozen"}), _backbone())
    # origins FIRST .. T-H-1 inclusive
    assert len(res.origins) == T - H - FIRST
    assert res.origins[-1] + H == T - 1


# ----------------------------------------------------------------------------- interfaces needed by VINE
from tsl.online import future_perturbation_test, make_provisional  # noqa: E402
from tsl.online.methods.vine import revision  # noqa: E402


@pytest.mark.parametrize("d,lag", [(2 * H, None), (2 * H, 0), (3 * H, H)])
def test_prefix_lag_rule(d, lag):
    """P-prefix: y_tau usable once tau <= now - l (l defaults to d - H)."""
    z = _z()
    ev = OnlineEvaluator(z, FIRST, FIRST + 80, L, H, delay=d, mode="prefix", prefix_lag=lag)
    spy = _Spy()
    ev.run(spy, _backbone())
    ell = d - H if lag is None else lag
    ps = [r for r in spy.log if r[0] == "p"]
    assert ps and max(now - (o + k) for _, now, o, k in ps) >= ell
    for _, now, o, k in ps:
        assert o + k <= now - ell and 1 <= k <= H


class _ArchiveUser(OnlineMethod):
    name = "archive_user"

    def __init__(self, ahead=0):
        super().__init__()
        self.ahead = ahead
        self.revs = []

    def predict(self, x, base_pred, t):
        f = self.ctx.base_forecast(t + self.ahead)
        np.testing.assert_array_equal(f, base_pred)
        r = revision(self.ctx, t)
        if r is not None:
            self.revs.append(r)
        return base_pred


def test_archive_gives_past_forecasts_and_blocks_future_ones():
    z = _z()
    ev = OnlineEvaluator(z, FIRST, FIRST + 50, L, H)
    m = _ArchiveUser(0)
    ev.run(m, _backbone())
    assert len(m.revs) == 50 and m.revs[0].shape == (H - 1, 3)
    with pytest.raises(LeakageError):
        ev.run(_ArchiveUser(1), _backbone())


def test_archive_chunked_cache_matches_full():
    z = _z()
    ev = OnlineEvaluator(z, FIRST, None, L, H)
    bb = _backbone()
    full = ev.make_archive(bb).all()
    small = ev.make_archive(bb, chunk=32, max_cache_bytes=1)   # forces eviction + recomputation
    for s in (FIRST, FIRST + 100, FIRST + 5, T - H - 1, FIRST + 7):
        np.testing.assert_array_equal(small.get(s, guard=False), full[s - FIRST])


@pytest.mark.parametrize("spec", [{"type": "rls"}, {"type": "elf", "update_every": 20, "warmup_updates": 1},
                                  {"type": "ogd", "scope": "head", "lr": 1e-3}])
@pytest.mark.parametrize("mode", ["strict", "prefix"])
def test_reusable_leak_test_two_channel(spec, mode):
    y = _z()
    x = make_provisional(y, seed=7)
    s = dict(spec)
    if mode == "prefix" and s["type"] == "ogd":
        s["use_prefix"] = True
    ok, rep = future_perturbation_test(lambda: build_method(s), _backbone(), y, FIRST, L, H, delay=2 * H,
                                       mode=mode, x=x)
    assert ok, rep


def test_reusable_leak_test_flags_leaky_mode():
    y = _z()
    ok, rep = future_perturbation_test(lambda: build_method({"type": "ogd", "scope": "full", "lr": 1e-2}),
                                       _backbone(), y, FIRST, L, H, mode="leaky", allow_leaky=True)
    assert not ok and rep[0]["first_affected_origin"] is not None


def test_inputs_are_features_not_targets():
    """Under two channels, history.inputs is readable up to now but labels only up to now-d+H."""
    y = _z()
    x = make_provisional(y, seed=3)
    d = 2 * H

    class Reader(OnlineMethod):
        name = "reader"

        def predict(self, xw, bp, t):
            np.testing.assert_array_equal(self.ctx.history.inputs(t - L + 1, t + 1), xw)
            return bp

        def update(self, info):
            self.ctx.history.labels(0, info.now - d + H + 1)
            with pytest.raises(LeakageError):
                self.ctx.history.labels(0, info.now + 1)

    OnlineEvaluator(y, FIRST, FIRST + 60, L, H, delay=d, x=x).run(Reader(), _backbone())


def test_vine_stub_refuses_to_run():
    with pytest.raises(NotImplementedError):
        OnlineEvaluator(_z(), FIRST, FIRST + 10, L, H).run(build_method({"type": "vine"}), _backbone())
