"""EXPLORATORY (added after the confirmation results; not part of the pre-registered analysis).

Causal per-horizon switch between VINE-S and last-value persistence ("VINE-S+P"). For every horizon h, at test origin
t the switch uses the forecaster whose discounted squared error on FULLY MATURED origins (origin <= t - d) is lower;
the discount is VINE-S's own fixed memory (gamma = 0.9998, nothing tuned), and VINE-S is used until the first label
has matured. Squared errors are the stored per-origin, per-horizon channel-averaged errors of the online grid, so the
switch is replayed exactly, without re-running any model. Statistics follow vine_report.py (seed-averaged per-origin
losses, block bootstrap CI, DM p-value, significance = Holm-adjusted p < 0.05 and CI excluding zero), with Holm over
this exploratory table only.

usage: python vine_fallback.py --root results/online_grid --out fallback.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsl.stats import block_bootstrap_mean_diff, diebold_mariano, holm  # noqa: E402

SEASON = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "Weather": 144, "Exchange": 7}
DEVELOPMENT = {"ETTh1", "ETTh2", "ETTm1"}
SEEDS = (0, 1, 2)
GAMMA = 0.9998
A, P = "vine_s", "persist_last"
VERSUS = ["vine_s", "persist_last", "dric_l8"]


def switch(org: np.ndarray, se_a: np.ndarray, se_p: np.ndarray, d: int, gamma: float = GAMMA):
    """Per-origin loss of the causal switch and the share of (origin, horizon) pairs that used persistence."""
    n, H = se_a.shape
    diff = se_a - se_p                                    # > 0: persistence was better at that origin/horizon
    S = np.zeros((n, H))
    for k in range(n):                                    # discounted running sum in calendar time
        S[k] = diff[k] if k == 0 else gamma ** (org[k] - org[k - 1]) * S[k - 1] + diff[k]
    last = np.searchsorted(org, org - d, side="right") - 1    # last fully matured origin index at each t
    use_p = np.zeros((n, H), bool)
    ok = last >= 0
    use_p[ok] = S[last[ok]] > 0
    se = np.where(use_p, se_p, se_a)
    return se.mean(axis=1), use_p.mean()


def load(root, ds, bb, H, s, m, d):
    f = root / ds / bb / f"H{H}" / f"seed{s}" / f"steps_{m}_d{d}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["origins"], z["se"].astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results/online_grid")
    ap.add_argument("--backbone", default="DLinear")
    ap.add_argument("--out", type=Path, default=Path("fallback.csv"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    rows = []
    for ds in sorted(SEASON):
        for H in (24, 96):
            for d in (H, 2 * H, 4 * H, 8 * H):
                per_seed, ok = [], True
                for s in SEEDS:
                    la, lp = load(a.root, ds, a.backbone, H, s, A, d), load(a.root, ds, a.backbone, H, s, P, d)
                    others = {m: load(a.root, ds, a.backbone, H, s, m, d) for m in VERSUS}
                    if la is None or lp is None or any(v is None for v in others.values()):
                        ok = False
                        break
                    if not np.array_equal(la[0], lp[0]):
                        raise RuntimeError(f"origin mismatch {ds} H{H} d{d} s{s}")
                    loss, share = switch(la[0], la[1], lp[1], d)
                    per_seed.append((la[0], loss, share, {m: v[1].mean(axis=1) for m, v in others.items()}))
                if not ok:
                    continue
                org = per_seed[0][0]
                x = np.mean([p[1] for p in per_seed], axis=0)
                share = float(np.mean([p[2] for p in per_seed]))
                block = max(2 * SEASON[ds], 2 * H)
                for m in VERSUS:
                    y = np.mean([p[3][m] for p in per_seed], axis=0)
                    bs = block_bootstrap_mean_diff(x, y, block=block, B=a.boot, seed=0)
                    dm = diebold_mariano(x, y, h=H)
                    rows.append({"dataset": ds, "H": H, "delay": d, "lag": d - H, "a": "vine_s_p", "b": m,
                                 "mse_a": x.mean(), "mse_b": y.mean(), "rel": x.mean() / y.mean() - 1.0,
                                 "rel_lo": bs["ci_low"] / y.mean(), "rel_hi": bs["ci_high"] / y.mean(),
                                 "p_dm": dm["p"], "share_persist": share, "n": len(org),
                                 "confirmation": ds not in DEVELOPMENT})
                print(f"{ds} H{H} d{d}: persistence share {share:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_dm.values)
    df["sig"] = (df.p_holm < 0.05) & ((df.rel_lo > 0) | (df.rel_hi < 0))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} contrasts -> {a.out}")


if __name__ == "__main__":
    main()
