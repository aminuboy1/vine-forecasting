from __future__ import annotations

import numpy as np
import torch

from tsl.data.registry import synthetic_dataset
from tsl.data.splits import SplitSpec, compute_borders, prepare_split
from tsl.data.windows import make_windows
from tsl.models import build_backbone
from tsl.models.backbones import Naive
from tsl.online import OnlineEvaluator, build_method
from tsl.stats import diebold_mariano, holm


def test_fsnet_split_ratios_and_standard_ett():
    b = compute_borders(10000, SplitSpec("fsnet"))
    assert b == {"train_end": 2000, "val_end": 2500, "test_end": 10000}
    e = compute_borders(17420, SplitSpec("standard"), is_ett=True, steps_per_day=24)
    assert e == {"train_end": 8640, "val_end": 11520, "test_end": 14400}
    o = compute_borders(1000, SplitSpec("standard"))
    assert o == {"train_end": 700, "val_end": 800, "test_end": 1000}


def test_scaler_fitted_on_train_rows_only():
    d = synthetic_dataset(T=1000, C=2)
    p1 = prepare_split(d, "fsnet")
    d.values[p1.borders["train_end"]:] += 1000.0          # change val/test rows drastically
    p2 = prepare_split(d, "fsnet")
    np.testing.assert_allclose(p1.scaler.mean, p2.scaler.mean)
    np.testing.assert_allclose(p1.scaler.std, p2.scaler.std)
    assert p1.z.dtype == np.float32


def test_windows_content_and_no_drop_last():
    z = np.arange(30, dtype=np.float32)[:, None]
    X, Y, o = make_windows(z, 0, 30, 5, 3)
    assert len(X) == 30 - 5 - 3 + 1
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(Y[0, :, 0], [5, 6, 7])
    assert o[0] == 4 and o[-1] == 26


def test_backbones_shapes_and_channel_agnostic():
    x = torch.randn(4, 48, 5)
    for name, kw in [("DLinear", {}), ("PatchTST", {"d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32}),
                     ("iTransformer", {"d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32})]:
        m = build_backbone(name, 48, 12, **kw)
        assert m(x).shape == (4, 12, 5)
        assert m(torch.randn(2, 48, 9)).shape == (2, 12, 9)          # different channel count
        mask = torch.tensor([[1, 1, 0, 1, 0]] * 4, dtype=torch.bool)
        assert torch.isfinite(m(x, channel_mask=mask)).all()
        assert len(list(m.head_parameters())) > 0


class _Biased(Naive):
    """Naive forecaster with a constant bias -> residual is learnable."""

    def forward(self, x, channel_mask=None):
        return super().forward(x) - 0.7


def _sine(T=3000, C=2):
    t = np.arange(T)
    return np.stack([np.sin(2 * np.pi * t / 24) + 0.05 * np.random.default_rng(c).standard_normal(T)
                     for c in range(C)], 1).astype(np.float32)


def test_rls_removes_constant_bias():
    z = _sine()
    L, H = 48, 12
    ev = OnlineEvaluator(z, 1000, None, L, H)
    fr = ev.run(build_method({"type": "frozen"}), _Biased(L, H)).summary()["mse"]
    rl = ev.run(build_method({"type": "rls", "n_lags": 1}), _Biased(L, H)).summary()["mse"]
    assert rl < 0.5 * fr, (rl, fr)


def test_elf_improves_poor_backbone_on_periodic_signal():
    z = _sine()
    L, H = 48, 12
    ev = OnlineEvaluator(z, 1000, None, L, H)
    fr = ev.run(build_method({"type": "frozen"}), _Biased(L, H)).summary()
    el = ev.run(build_method({"type": "elf", "update_every": 50}), _Biased(L, H)).summary()
    assert el["mse"] < 0.5 * fr["mse"], (el["mse"], fr["mse"])
    assert el["state_bytes_final"] > 0


def test_state_bytes_and_latency_reported():
    z = _sine(1500)
    ev = OnlineEvaluator(z, 800, None, 48, 12)
    s = ev.run(build_method({"type": "rls"}), _Biased(48, 12)).summary()
    assert s["state_bytes_final"] > 0 and s["update_latency_us_mean"] > 0 and s["n_forecasts"] == 1500 - 12 - 800


def test_holm_and_dm():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.2])
    rng = np.random.default_rng(0)
    a = rng.random(500)
    r = diebold_mariano(a, a + 0.1, h=4)
    assert r["stat"] < 0 and r["p"] < 1e-6
    r2 = diebold_mariano(a, a[::-1].copy(), h=4)
    assert r2["p"] > 0.01


def test_block_bootstrap():
    from tsl.stats import block_bootstrap_mean_diff
    rng = np.random.default_rng(1)
    a = rng.random(2000)
    r = block_bootstrap_mean_diff(a, a + 0.05, block=48, B=500)
    assert r["ci_high"] < 0 and r["p"] <= 0.01
    r2 = block_bootstrap_mean_diff(a, rng.permutation(a), block=48, B=500)
    assert r2["ci_low"] < 0 < r2["ci_high"]
