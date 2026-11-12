"""Future-perturbation test: turns "leak-free" from a claim into a check applicable to ANY method.

Run the stream twice: once as is, once with every label AND input at rows >= t0 replaced (NaN by default,
or random noise). All forecasts issued at origins < t0 must be bit-identical (and finite). This covers
leaks through labels, prefixes, history access, archives, and any hidden reference to the full arrays
(e.g. a method that kept a pointer to the data).

    from tsl.online.leaktest import future_perturbation_test
    ok, report = future_perturbation_test(lambda: MyMethod(...), backbone, y, first, L, H, delay=H)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .evaluator import OnlineEvaluator


def future_perturbation_test(method_factory: Callable, backbone, y: np.ndarray, first: int, L: int, H: int,
                             delay: int | None = None, mode: str = "strict", t0_offsets=(50, 200),
                             fill: str = "nan", x: np.ndarray | None = None, allow_leaky: bool = False,
                             device="cpu", seed: int = 0, prefix_lag: int | None = None) -> tuple[bool, list]:
    report = []
    ok = True
    x = y if x is None else x
    for off in t0_offsets:
        t0 = first + int(off)
        last = t0 + 2
        y2, x2 = y.copy(), x.copy()
        if fill == "nan":
            y2[t0:] = np.nan
            x2[t0:] = np.nan
        else:
            rng = np.random.default_rng(seed + off)
            y2[t0:] = rng.standard_normal(y2[t0:].shape) * 10
            x2[t0:] = rng.standard_normal(x2[t0:].shape) * 10
        runs = []
        for yy, xx in ((y, x), (y2, x2)):
            ev = OnlineEvaluator(yy, first, last, L, H, delay=delay, mode=mode, allow_leaky=allow_leaky,
                                 check_leaks=mode != "leaky", keep_preds=True, x=xx, prefix_lag=prefix_lag)
            runs.append(ev.run(method_factory(), backbone, device=device, seed=seed).preds)
        n = t0 - first
        a, b = runs[0][:n], runs[1][:n]
        finite = bool(np.isfinite(b).all())
        equal = bool(np.array_equal(a, b))
        bad = np.flatnonzero(~(a == b).reshape(n, -1).all(1))
        report.append({"t0": t0, "finite": finite, "bit_identical": equal,
                       "first_affected_origin": int(first + bad[0]) if len(bad) else None})
        ok &= finite and equal
    return ok, report
