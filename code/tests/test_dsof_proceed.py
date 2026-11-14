"""Ports of DSOF (ICLR 2025) and Proceed (KDD 2025) to the truthful delayed-label protocol.

1. causality: future-perturbation test with two channels (inputs != labels), d = H and d = 2H, NaN and noise futures;
2. Proceed's adapters start at the identity (zero-initialised generator output) and match the original layer maths;
3. sanity: both methods learn something on a toy stream with a strong daily cycle and level shift.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tsl.data.registry import synthetic_dataset
from tsl.models import build_backbone
from tsl.online import OnlineEvaluator, future_perturbation_test, make_provisional
from tsl.online.methods.dsof import DSOF
from tsl.online.methods.proceed import AdaptedLinear, Proceed, ProceedNet

L, H = 48, 8
T, FIRST = 700, 300
FACTORIES = {
    "dsof": lambda: DSOF(train_epochs=2),
    "dsof_k124": lambda: DSOF(train_epochs=2, td_k=[1, 2, 4]),
    "proceed": lambda: Proceed(train_epochs=2, online_lr=1e-3),
}


def _backbone():
    torch.manual_seed(0)
    return build_backbone("DLinear", L, H)


@pytest.mark.parametrize("key", list(FACTORIES))
@pytest.mark.parametrize("dmult", [1, 2])
@pytest.mark.parametrize("fill", ["nan", "noise"])
def test_future_perturbation_strict(key, dmult, fill):
    y = synthetic_dataset(T=T, C=3, seed=1).values
    x = make_provisional(y, seed=7)
    ok, rep = future_perturbation_test(FACTORIES[key], _backbone(), y, FIRST, L, H, delay=dmult * H,
                                       mode="strict", x=x, fill=fill)
    assert ok, rep


@pytest.mark.parametrize("key", ["dsof", "proceed"])
def test_offline_cache_is_not_mutated_by_online_runs(key):
    """Repeated runs in one process (tuning, then test) must not depend on run order through the offline cache."""
    y = synthetic_dataset(T=T, C=3, seed=1).values
    bb = _backbone()
    outs = []
    for last in (FIRST + 120, FIRST + 40, FIRST + 120):
        ev = OnlineEvaluator(y, FIRST, last, L, H, delay=H, keep_preds=True)
        outs.append(ev.run(FACTORIES[key](), bb).preds)
    np.testing.assert_array_equal(outs[0], outs[2])
    np.testing.assert_array_equal(outs[0][:41], outs[1])


def test_adapted_linear_identity_and_original_maths():
    torch.manual_seed(0)
    lin = nn.Linear(6, 4)
    ad = AdaptedLinear(lin)
    x = torch.randn(3, 5, 6)
    assert torch.allclose(ad(x), lin(x))
    a = torch.randn(3, 4 + 6 + 4)
    ad.adaptation = a
    scale, scale2, shift = a[:, :4] + 1, a[:, 4:10] + 1, a[:, 10:]
    ref = torch.stack([((x[i] * scale2[i]) @ lin.weight.T) * scale[i] + shift[i] + lin.bias for i in range(3)])
    assert torch.allclose(ad(x), ref, atol=1e-6)


def test_proceed_net_starts_at_backbone():
    bb = _backbone()
    net = ProceedNet(bb, L, H, 3, 50, 32)
    x = torch.randn(4, L, 3)
    with torch.no_grad():
        assert torch.allclose(net(x), bb(x), atol=1e-6)


def _toy(N=900, C=2, seed=0):
    rng = np.random.default_rng(seed)
    n = FIRST + N + H + 1
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24)[:, None] * rng.uniform(.5, 1.5, C) + 0.1 * rng.standard_normal((n, C))
    y[FIRST + 200:] += 1.5                       # level shift inside the stream
    return y.astype(np.float32)


@pytest.mark.parametrize("key", ["dsof", "proceed"])
def test_learns_on_toy(key):
    y = _toy()
    bb = _backbone()
    ev = OnlineEvaluator(y, FIRST, FIRST + 899, L, H, delay=H, keep_preds=True)
    tgt = np.stack([y[s + 1:s + 1 + H] for s in range(FIRST, FIRST + 900)])
    res = ev.run(FACTORIES[key](), bb)
    fro = ev.run(build_method_frozen(), bb)
    sel = slice(450, 900)
    mse = np.mean((res.preds[sel] - tgt[sel]) ** 2)
    mse_f = np.mean((fro.preds[sel] - tgt[sel]) ** 2)
    assert np.isfinite(mse) and mse < mse_f, (key, mse, mse_f)


def build_method_frozen():
    from tsl.online.methods.simple import Frozen
    return Frozen()
