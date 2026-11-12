"""Paired significance tests with multiple-comparison correction.

* Diebold-Mariano test on per-step loss differentials, Newey-West (Bartlett) HAC variance with lag h-1
  (h = forecast horizon; overlapping multi-step forecasts are autocorrelated) and the Harvey, Leybourne &
  Newbold (1997) small-sample correction; p-values from Student-t with n-1 df.
* Wilcoxon signed-rank test across paired cells (dataset x horizon x delay x seed, or similar).
* Holm (1979) step-down adjustment for a family of p-values.
* Circular moving-block bootstrap of the mean paired loss differential (audit procedure; CI + p-value).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def newey_west_var(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = len(d)
    gamma0 = float(d @ d) / n
    v = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        v += 2.0 * w * float(d[k:] @ d[:-k]) / n
    return max(v, 1e-300)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1, hln: bool = True) -> dict:
    """H0: equal expected loss. Negative statistic => method A has lower loss than B."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = max(h - 1, 0)
    var = newey_west_var(d, lag) / n
    stat = d.mean() / np.sqrt(var)
    if hln:
        k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if n > 2 * h else 1.0
        stat = stat * k
    p = 2 * sps.t.sf(abs(stat), df=n - 1)
    return {"stat": float(stat), "p": float(p), "mean_diff": float(d.mean()), "n": int(n), "lag": lag}


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, b):
        return {"stat": float("nan"), "p": 1.0, "n": int(len(a)), "median_diff": float(np.median(a - b))}
    r = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n": int(len(a)),
            "median_diff": float(np.median(a - b)), "wins_a": int(np.sum(a < b)), "wins_b": int(np.sum(b < a))}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def block_bootstrap_mean_diff(loss_a: np.ndarray, loss_b: np.ndarray, block: int, B: int = 1000,
                              seed: int = 0, alpha: float = 0.05) -> dict:
    """Circular moving-block bootstrap of the mean paired loss differential d = a - b.

    Block length should cover the dependence (e.g. 2x the seasonal period, as in the delay-aware audit).
    Returns the mean difference, a percentile CI, and a two-sided p-value for H0: E[d] = 0 obtained by
    bootstrapping the centred series.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n] % n
    boot = d[idx].mean(1)
    centred = boot - d.mean()
    p = float(np.mean(np.abs(centred) >= abs(d.mean())))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p": max(p, 1.0 / B),
            "B": B, "block": block, "n": n}
