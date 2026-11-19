"""Pre-registered analysis of the Paper 2 online grid (PREREGISTRATION.md, Paper 2 and its amendments).

Reads results/online_grid/<dataset>/<backbone>/H<H>/seed<k>/{test.csv, steps_<method>_d<d>.npz} and writes
<out>/{means.csv, contrasts.csv, hypotheses.json, cost.csv, coverage.csv}.

Statistics (fixed before the confirmation results; see the amendment table):
  * a setting is (dataset, H, d); a contrast A vs B in a setting uses only seeds where both exist, and is reported
    only when all 3 seeds exist. Per-origin loss = MSE over the horizon (channels are already averaged by the
    harness), averaged over seeds at the same origin (seeds are backbone seeds; origins are identical);
  * paired differential d_t = loss_A - loss_B; circular moving-block bootstrap (B = 1000, block = max(2 x season, 2H))
    for the 95% CI, and Diebold-Mariano with HAC bandwidth H - 1 for the p-value. A contrast is significant when its
    Holm-adjusted DM p-value is below 0.05 AND its bootstrap CI excludes zero. (Amendment 2026-11-16: the first full
    run used max(p_boot, p_DM); a bootstrap p-value cannot fall below 1/B, so under Holm with m > 50 tests that rule
    can never reject. The bootstrap is kept as the CI and as a required second check.)
  * Holm over the whole primary table (both co-primary methods, both primary contrasts, every setting), which is
    stricter than Holm within each setting plus across the two methods; secondary contrasts are Holm-corrected
    within their own table;
  * across settings: per-dataset median relative gains, Wilcoxon over datasets (dependent cells collapsed).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsl.stats import block_bootstrap_mean_diff, diebold_mariano, holm, wilcoxon_paired  # noqa: E402

SEASON = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "Weather": 144, "Exchange": 7}
CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21, "Exchange": 8}
DEVELOPMENT = {"ETTh1", "ETTh2", "ETTm1"}
CO_PRIMARY = ["vine_s", "vine_lam"]
CONTROL = {"vine_s": "vine_s_lo", "vine_lam": "mc_ls_pr"}
EXTERNAL = ["rls_audit", "elf", "epoc", "dric_g", "dric_l8", "dsof", "proceed"]
H1_FIELD = ["rls_audit", "elf", "dric_l8", "dric_g", "epoc", "mc_ls", "mc_ls_pr", "persist_last", "persist_seasonal"]
SEEDS = (0, 1, 2)


def load_rows(root: Path, backbone: str) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob(f"*/{backbone}/H*/seed*/test.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    df = pd.DataFrame(rows)
    for c in ["H", "seed", "delay", "lag", "n", "tuned_at_dfac"]:
        df[c] = df[c].astype(int)
    for c in ["mse", "mae", "step_us_med", "step_us_p95", "step_us_mean", "state_learned", "state_pending"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates(["dataset", "H", "seed", "delay", "method"], keep="last")


class Losses:
    def __init__(self, root: Path, backbone: str):
        self.root, self.bb, self.cache = root, backbone, {}

    def get(self, ds, H, d, method):
        """Seed-averaged per-origin loss, or None unless all seeds exist."""
        key = (ds, H, d, method)
        if key not in self.cache:
            arrs, org = [], None
            for s in SEEDS:
                f = self.root / ds / self.bb / f"H{H}" / f"seed{s}" / f"steps_{method}_d{d}.npz"
                if not f.exists():
                    self.cache[key] = None
                    return None
                z = np.load(f)
                if org is None:
                    org = z["origins"]
                elif not np.array_equal(org, z["origins"]):
                    raise RuntimeError(f"origin mismatch {f}")
                arrs.append(z["se"].astype(np.float64).mean(axis=1))
            self.cache[key] = (org, np.mean(arrs, axis=0))
        return self.cache[key]


def contrast(L: Losses, ds, H, d, a, b, B):
    la, lb = L.get(ds, H, d, a), L.get(ds, H, d, b)
    if la is None or lb is None:
        return None
    if not np.array_equal(la[0], lb[0]):
        raise RuntimeError(f"origin mismatch {ds} H{H} d{d} {a} {b}")
    xa, xb = la[1], lb[1]
    block = max(2 * SEASON[ds], 2 * H)
    bs = block_bootstrap_mean_diff(xa, xb, block=block, B=B, seed=0)
    dm = diebold_mariano(xa, xb, h=H)
    base = xb.mean()
    return {"dataset": ds, "H": H, "delay": d, "lag": d - H, "a": a, "b": b, "mse_a": xa.mean(), "mse_b": base,
            "rel": xa.mean() / base - 1.0, "rel_lo": bs["ci_low"] / base, "rel_hi": bs["ci_high"] / base,
            "p_boot": bs["p"], "p_dm": dm["p"], "p": dm["p"], "block": block, "n": len(xa),
            "confirmation": ds not in DEVELOPMENT}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results/online_grid")
    ap.add_argument("--backbone", default="DLinear")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--harm", action="store_true", help="also test every method against frozen (harm rate)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    df = load_rows(a.root, a.backbone)
    L = Losses(a.root, a.backbone)

    # ---- coverage and seed-averaged means (complete settings only: all seeds present)
    g = df.groupby(["dataset", "H", "delay", "method"])
    means = g.agg(mse=("mse", "mean"), mae=("mae", "mean"), mse_sd=("mse", "std"), n_seeds=("seed", "nunique"))
    means = means.reset_index()
    means["lag_over_H"] = (means["delay"] - means["H"]) / means["H"]
    means.to_csv(a.out / "means.csv", index=False)
    full = means[means.n_seeds == len(SEEDS)]
    settings = sorted({(r.dataset, r.H, r.delay) for r in full.itertuples()})
    cov = full.groupby(["dataset", "H"]).agg(methods=("method", "nunique"), delays=("delay", "nunique")).reset_index()
    cov.to_csv(a.out / "coverage.csv", index=False)

    # ---- primary contrasts: each co-primary vs dric_l8 and vs its label-only control; one Holm family
    prim = []
    for ds, H, d in settings:
        for m in CO_PRIMARY:
            for b, kind in [("dric_l8", "vs_dric_l8"), (CONTROL[m], "vs_control")]:
                r = contrast(L, ds, H, d, m, b, a.boot)
                if r:
                    prim.append({**r, "family": "primary", "kind": kind})
    if prim:
        for r, p in zip(prim, holm([r["p"] for r in prim])):
            r["p_holm"] = float(p)
            r["sig"] = bool(p < 0.05 and (r["rel_hi"] < 0 or r["rel_lo"] > 0))
    # ---- secondary: co-primary vs each external method and vs DSOF/Proceed; Holm within this table
    sec = []
    for ds, H, d in settings:
        for m in CO_PRIMARY:
            for b in EXTERNAL:
                if b == "dric_l8":
                    continue
                r = contrast(L, ds, H, d, m, b, a.boot)
                if r:
                    sec.append({**r, "family": "secondary", "kind": f"vs_{b}"})
    if sec:
        for r, p in zip(sec, holm([r["p"] for r in sec])):
            r["p_holm"] = float(p)
            r["sig"] = bool(p < 0.05 and (r["rel_hi"] < 0 or r["rel_lo"] > 0))
    # ---- ablation / sensitivity rows for vine_s (reported, never selected); Holm within table
    abl = []
    for ds, H, d in settings:
        for b in ["vine_s_lo", "vine_s_pure", "vine_lam_lm", "vine_lam", "vine_s_g9995", "vine_s_g9999"]:
            r = contrast(L, ds, H, d, "vine_s", b, a.boot)
            if r:
                abl.append({**r, "family": "ablation", "kind": f"vs_{b}"})
        r = contrast(L, ds, H, d, "vine_teacher", "vine", a.boot)
        if r:
            abl.append({**r, "family": "ablation", "kind": "teacher_vs_vine"})
    if abl:
        for r, p in zip(abl, holm([r["p"] for r in abl])):
            r["p_holm"] = float(p)
            r["sig"] = bool(p < 0.05 and (r["rel_hi"] < 0 or r["rel_lo"] > 0))
    # ---- harm: every method vs frozen in every setting; Holm within this table; harm = significantly worse
    harm = []
    if a.harm:
        methods = sorted(set(full.method) - {"frozen"})
        for ds, H, d in settings:
            for m in methods:
                r = contrast(L, ds, H, d, m, "frozen", a.boot)
                if r:
                    harm.append({**r, "family": "harm", "kind": "vs_frozen"})
        if harm:
            for r, p in zip(harm, holm([r["p"] for r in harm])):
                r["p_holm"] = float(p)
                r["sig"] = bool(p < 0.05 and (r["rel_hi"] < 0 or r["rel_lo"] > 0))
    con = pd.DataFrame(prim + sec + abl + harm)
    con.to_csv(a.out / "contrasts.csv", index=False)

    # ---- hypotheses (as written; evaluated per co-primary method)
    piv = full.pivot_table(index=["dataset", "H", "delay"], columns="method", values="mse")
    hyp = {"settings": len(settings), "datasets": sorted({s[0] for s in settings})}
    for m in CO_PRIMARY + ["safe_mix"]:
        if m not in piv:
            continue
        field = [c for c in H1_FIELD if c in piv]
        best = piv[field].min(axis=1)
        rel = piv[m] / best - 1.0
        cells = pd.DataFrame({"rel": rel}).reset_index()
        cells["lag_ge_H"] = cells.delay >= 2 * cells.H
        hi = cells[cells.lag_ge_H].dropna()
        lo = cells[~cells.lag_ge_H].dropna()
        per_ds = hi.groupby("dataset").rel.median()
        w = wilcoxon_paired(per_ds.values, np.zeros(len(per_ds))) if len(per_ds) >= 2 else {"p": float("nan")}
        hyp[f"H1_{m}"] = {"frac_wins_lag_ge_H": float((hi.rel < 0).mean()) if len(hi) else None,
                          "n_lag_ge_H": int(len(hi)), "median_rel_lag_ge_H": float(hi.rel.median()) if len(hi) else None,
                          "per_dataset_median_rel": per_ds.round(5).to_dict(), "wilcoxon_over_datasets_p": w["p"],
                          "lag0_max_rel_vs_best": float(lo.rel.max()) if len(lo) else None,
                          "lag0_frac_no_worse": float((lo.rel <= 0).mean()) if len(lo) else None,
                          "pass_60pct": bool(len(hi) and (hi.rel < 0).mean() >= 0.6)}
    # H2: ETTh2 safe_mix non-inferior to frozen (+1% margin, one-sided 95%: upper end of the 90% two-sided CI)
    h2 = []
    for (ds, H, d) in settings:
        if ds != "ETTh2":
            continue
        for m in ["safe_mix", "vine_s", "vine_lam"]:
            la, lb = L.get(ds, H, d, m), L.get(ds, H, d, "frozen")
            if la is None or lb is None:
                continue
            bs = block_bootstrap_mean_diff(la[1], lb[1], block=max(2 * SEASON[ds], 2 * H), B=a.boot, seed=0, alpha=0.10)
            h2.append({"method": m, "H": H, "delay": d, "rel": la[1].mean() / lb[1].mean() - 1,
                       "rel_upper95": bs["ci_high"] / lb[1].mean()})
    hyp["H2"] = {"rows": h2, "pass": {m: all(r["rel_upper95"] < 0.01 for r in h2 if r["method"] == m)
                                      for m in {r["method"] for r in h2}}}
    # H4: teacher target worse than vine in >= 70% of settings
    if {"vine_teacher", "vine"} <= set(piv):
        t = (piv["vine_teacher"] > piv["vine"]).dropna()
        hyp["H4"] = {"frac_teacher_worse": float(t.mean()), "n": int(len(t)), "pass": bool(t.mean() >= 0.7)}
    # H5: slope of MSE against d/H (per dataset x H, least squares over the delays) at most half of MC-LS's
    h5 = []
    for (ds, H), sub in full.groupby(["dataset", "H"]):
        p = sub.pivot_table(index="delay", columns="method", values="mse")
        if "mc_ls" not in p or len(p) < 3:
            continue
        x = p.index.values / H
        sl = {m: float(np.polyfit(x, p[m].values, 1)[0]) for m in ["mc_ls", "vine", "vine_lam", "vine_s", "dric_l8"]
              if m in p and p[m].notna().all()}
        h5.append({"dataset": ds, "H": H, **{f"slope_{k}": v for k, v in sl.items()}})
    hyp["H5"] = h5
    for m in ["vine", "vine_lam", "vine_s"]:
        ok = [r[f"slope_{m}"] <= 0.5 * r["slope_mc_ls"] for r in h5 if f"slope_{m}" in r and r["slope_mc_ls"] > 0]
        hyp[f"H5_pass_frac_{m}"] = float(np.mean(ok)) if ok else None
    # H7: cost at H = 24
    cost = df.groupby(["dataset", "H", "method"]).agg(step_us_med=("step_us_med", "median"),
                                                      step_us_p95=("step_us_p95", "median"),
                                                      state_learned=("state_learned", "median"),
                                                      state_pending=("state_pending", "median")).reset_index()
    cost["state_learned_per_channel"] = cost.state_learned / cost.dataset.map(CHANNELS)
    cost.to_csv(a.out / "cost.csv", index=False)
    h7 = []
    for ds, sub in cost[cost.H == 24].groupby("dataset"):
        s = sub.set_index("method")
        for m in CO_PRIMARY:
            if m in s.index and "rls_audit" in s.index:
                h7.append({"dataset": ds, "method": m, "latency_ratio_vs_rls": s.step_us_med[m] / s.step_us_med["rls_audit"],
                           "state_kB_per_channel": s.state_learned_per_channel[m] / 1e3})
    hyp["H7"] = h7
    # summary of primary contrasts
    if prim:
        P = pd.DataFrame(prim)
        hyp["primary_summary"] = (P.assign(win=P.rel < 0, sig_win=(P.rel < 0) & P.sig, sig_loss=(P.rel > 0) & P.sig)
                                  .groupby(["a", "kind", "confirmation"])
                                  .agg(n=("rel", "size"), wins=("win", "sum"), sig_wins=("sig_win", "sum"),
                                       sig_losses=("sig_loss", "sum"), median_rel=("rel", "median"))
                                  .reset_index().to_dict("records"))
    if harm:
        Hm = pd.DataFrame(harm)
        hyp["harm_rate"] = (Hm.assign(harm=(Hm.rel > 0) & Hm.sig, help=(Hm.rel < 0) & Hm.sig)
                            .groupby("a").agg(n=("rel", "size"), harm=("harm", "mean"), help=("help", "mean"))
                            .reset_index().to_dict("records"))
    (a.out / "hypotheses.json").write_text(json.dumps(hyp, indent=1, default=float))
    print(f"settings {len(settings)}; contrasts {len(con)}; written to {a.out}")


if __name__ == "__main__":
    main()
