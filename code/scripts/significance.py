"""Holm-corrected paired significance tests between methods.

Online results (results/<exp>/<dataset>/<segment>/H*/d*/<method>/seed*/steps.npz):
  * per cell (dataset, segment, H, delay, seed): Diebold-Mariano on per-origin losses (MSE averaged over
    horizon and channels), HAC lag H-1, plus a circular moving-block bootstrap (--block, --boot);
  * across cells: Wilcoxon signed-rank on cell-level mean MSE.
Continual results (results/<exp>/summary.csv with a `taskification` column):
  * Wilcoxon across (dataset, taskification, seed) cells on a chosen metric.
Holm correction is applied over the whole family of comparisons in each table.

Examples
  python scripts/significance.py results/smoke_online --reference frozen
  python scripts/significance.py results/smoke_online --all-pairs --metric ae
  python scripts/significance.py results/smoke_cl --metric final_avg_mse --reference finetune
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsl.stats import block_bootstrap_mean_diff, diebold_mariano, holm, wilcoxon_paired  # noqa: E402


def online_tables(root: Path, reference: str | None, all_pairs: bool, metric: str, alpha: float,
                  block: int = 48, B: int = 1000):
    recs = []
    for f in root.glob("*/*/H*/d*/*/seed*/steps.npz"):   # <dataset>/<segment>/H*/d*_<mode>/<method>/seed*/
        seed, method, dd, hh, seg, ds = (f.parts[-2], f.parts[-3], f.parts[-4], f.parts[-5], f.parts[-6],
                                         f.parts[-7])
        recs.append({"dataset": ds, "segment": seg, "H": int(hh[1:]), "delay": dd, "method": method,
                     "seed": seed, "path": f})
    if not recs:
        raise SystemExit(f"no steps.npz under {root}")
    df = pd.DataFrame(recs)
    methods = sorted(df.method.unique())
    pairs = list(itertools.combinations(methods, 2)) if all_pairs or not reference else \
        [(m, reference) for m in methods if m != reference]
    cell_cols = ["dataset", "segment", "H", "delay", "seed"]
    dm_rows, cell_means = [], {}
    for cell, g in df.groupby(cell_cols):
        losses = {}
        for _, r in g.iterrows():
            z = np.load(r.path)
            losses[r.method] = z[metric].mean(1)
            cell_means[(cell, r.method)] = float(z[metric].mean())
        for a, b in pairs:
            if a in losses and b in losses:
                res = diebold_mariano(losses[a], losses[b], h=int(cell[2]))
                bb = block_bootstrap_mean_diff(losses[a], losses[b], block=block, B=B)
                dm_rows.append({**dict(zip(cell_cols, cell)), "method_a": a, "method_b": b, **res,
                                "p_boot": bb["p"], "boot_ci_low": bb["ci_low"], "boot_ci_high": bb["ci_high"]})
    dm = pd.DataFrame(dm_rows)
    if len(dm):
        dm["p_holm"] = holm(dm.p.values)
        dm["p_boot_holm"] = holm(dm.p_boot.values)
        dm["significant"] = dm.p_holm < alpha
        dm["better"] = np.where(dm.mean_diff < 0, dm.method_a, dm.method_b)
    cells = sorted({c for c, _ in cell_means})
    wx_rows = []
    for a, b in pairs:
        common = [c for c in cells if (c, a) in cell_means and (c, b) in cell_means]
        va = np.array([cell_means[(c, a)] for c in common])
        vb = np.array([cell_means[(c, b)] for c in common])
        if len(common):
            wx_rows.append({"method_a": a, "method_b": b, "mean_a": va.mean(), "mean_b": vb.mean(),
                            "rel_change_a_vs_b_%": 100 * (va.mean() / vb.mean() - 1), **wilcoxon_paired(va, vb)})
    wx = pd.DataFrame(wx_rows)
    if len(wx):
        wx["p_holm"] = holm(wx.p.values)
        wx["significant"] = wx.p_holm < alpha
    return dm, wx


def cl_table(root: Path, reference: str | None, all_pairs: bool, metric: str, alpha: float):
    df = pd.read_csv(root / "summary.csv")
    methods = sorted(df.learner.unique())
    pairs = list(itertools.combinations(methods, 2)) if all_pairs or not reference else \
        [(m, reference) for m in methods if m != reference]
    piv = df.pivot_table(index=["dataset", "taskification", "seed"], columns="learner", values=metric)
    rows = []
    for a, b in pairs:
        sub = piv[[a, b]].dropna()
        rows.append({"method_a": a, "method_b": b, "metric": metric, "mean_a": sub[a].mean(), "mean_b": sub[b].mean(),
                     **wilcoxon_paired(sub[a].values, sub[b].values)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p.values)
        out["significant"] = out.p_holm < alpha
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--reference", default=None, help="compare every method against this one")
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--metric", default=None, help="online: se|ae (default se); continual: summary column")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=48, help="bootstrap block length (e.g. 2 x seasonal period)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    is_cl = (args.results / "summary.csv").exists() and "taskification" in open(args.results / "summary.csv").readline()
    if is_cl:
        t = cl_table(args.results, args.reference, args.all_pairs, args.metric or "final_avg_mse", args.alpha)
        print(t.to_string(index=False))
        t.to_csv(args.results / "significance_cl.csv", index=False)
    else:
        dm, wx = online_tables(args.results, args.reference, args.all_pairs, args.metric or "se", args.alpha,
                               args.block, args.boot)
        print("== Diebold-Mariano per cell (Holm over all rows) ==")
        print(dm.drop(columns=[c for c in ["lag"] if c in dm]).to_string(index=False) if len(dm) else "(none)")
        print("\n== Wilcoxon across cells (Holm) ==")
        print(wx.to_string(index=False) if len(wx) else "(none)")
        dm.to_csv(args.results / "significance_dm.csv", index=False)
        wx.to_csv(args.results / "significance_wilcoxon.csv", index=False)
    print(f"\nwritten to {args.results}")


if __name__ == "__main__":
    main()
