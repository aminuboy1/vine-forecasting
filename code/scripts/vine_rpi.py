"""Revision Predictability Index (RPI) and nesting-defect diagnostics for Paper 2 (pre-registered H3, H6; Fig. 4).

Definitions (methodology/paper2_online_tsf_methodology.md Sec. 7.5; PREREGISTRATION.md H3/H6; amendment 2026-11-14 (b)):
  * For any forecaster P that re-issues an (H, C) forecast every step, the revision of target s+h between origins
    s and s+1 is  rho^P_{s+1,h} = P_{s+1,h-1} - P_{s,h},  h = 2..H.
  * RPI_h(P) is the OUT-OF-SAMPLE R^2 of a per-channel ridge regression of rho^P_{s+1,h} on the features phi_s, on
    the validation stream: fitted on its first half, scored on its second half (the same split as tuning; the
    denominator uses the first-half mean, so RPI can be negative). Pooled over channels; RPI = mean over h. The ridge
    penalty is chosen from {1, 10, 100, 1000} (standardised features) by blocked 5-fold CV inside the first half.
  * phi_s is the feature map of the vine_s top layer (VINEFeatures with that cell's tuned feature settings), built
    from the forecasts the top layer actually receives: the DRiC-L8 forecasts F.
  * Three forecasters per (cell, delay):  B = frozen backbone,  F = DRiC-L8 (vine_s's base; pre-correction for the
    stack),  G = vine_s (post-correction).  H3 correlates RPI(F) with test gains; H6 checks RPI(G) <= 0.25 RPI(F).
  * Nesting defect (Sec. 4, nu_h = ||b_h||_Sigma): with the matured regression of F's error e_{s,h} on phi_s fitted
    on the first half and eps its residual, NU_h is the out-of-sample R^2 of eps_{s+1,h-1} on phi_s, i.e. the share
    of the residual variance explained by the defect coefficient b_h (0 under weak nesting). Labels are used after
    the fact: this is a diagnostic, never an input to any method.
  * EXPLORATORY, added 2026-11-16 after seeing development-stream RPIs (amendment 2026-11-16b): the online RPI
    (orpi_*), a prequential R^2 on the same second-half rows with an expanding-window refit (see online_r2).

Configurations are the frozen validation-tuned ones (tuning.json; tuned per delay, 8H uses the 4H configuration), so
the replay reproduces the forecasts issued on the validation stream during tuning.

    python scripts/vine_rpi.py --dataset ETTh2 --H 24 --seed 0
Output: results/online_grid/<dataset>/<backbone>/H<H>/seed<k>/rpi.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsl.data import load_dataset, prepare_split            # noqa: E402
from tsl.models import pretrain_backbone                    # noqa: E402
from tsl.online.evaluator import OnlineEvaluator            # noqa: E402
from tsl.online.methods import build_method                 # noqa: E402
from tsl.online.methods.vine import VINEFeatures            # noqa: E402
from tsl.utils import CODE_ROOT, set_seed, setup_runtime    # noqa: E402
from online_grid import resolve                             # noqa: E402

FEAT_KEYS = ("k_f", "k_r", "rates", "use_level", "use_revision_feat", "use_ewma", "k_lb", "n_lb", "use_last",
             "k_phase", "season")


def features(top: dict, H: int, C: int, L: int, season: int, xs: np.ndarray, P: np.ndarray, origins) -> np.ndarray:
    """phi_s for every origin (N, C, p), from input windows xs (N, L, C) and the forecasts P (N, H, C)."""
    t = {k: top[k] for k in FEAT_KEYS if k in top}
    fm = VINEFeatures(H, C, t.get("k_f", 3), t.get("k_r", 3), tuple(t.get("rates", (0.5, 0.95))),
                      int(t.get("season") or season), t.get("use_level", True), t.get("use_revision_feat", True),
                      t.get("use_ewma", True), t.get("k_lb", 0), t.get("n_lb", 2), L, t.get("use_last", False),
                      t.get("k_phase", 0))
    out = np.empty((len(origins), C, fm.p))
    for i, s in enumerate(origins):
        out[i] = fm(xs[i].astype(np.float64), P[i].astype(np.float64), None if i == 0 else P[i - 1].astype(np.float64),
                    int(s))
    return out


RIDGES = (1.0, 10.0, 100.0, 1000.0)


def _design(X: np.ndarray, c: int, n_fit: int) -> np.ndarray:
    """Channel c's features without the intercept, standardised with the statistics of rows < n_fit."""
    Xc = X[:, c, 1:]
    mu, sd = Xc[:n_fit].mean(0), Xc[:n_fit].std(0)
    keep = sd > 1e-10
    return (Xc[:, keep] - mu[keep]) / sd[keep]


def _sse(X: np.ndarray, Y: np.ndarray, fit: np.ndarray, ev: np.ndarray, ridge: float, n_std: int):
    """Per-channel ridge fitted on rows `fit`, evaluated on rows `ev`: (SSE, SST, residuals on ev), pooled."""
    k, C = Y.shape[1], Y.shape[2]
    sse, sst = np.zeros(k), np.zeros(k)
    res = np.empty((len(ev), k, C))
    for c in range(C):
        Z = _design(X, c, n_std)
        Yc = Y[:, :, c]
        ym = Yc[fit].mean(0)
        A = Z[fit]
        W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ (Yc[fit] - ym))
        r = Yc[ev] - ym - Z[ev] @ W
        res[:, :, c] = r
        sse += (r ** 2).sum(0)
        sst += ((Yc[ev] - ym) ** 2).sum(0)
    return sse, sst, res


def choose_ridge(X: np.ndarray, Y: np.ndarray, n_fit: int, folds: int = 5) -> float:
    """Ridge penalty by blocked (contiguous) K-fold cross-validation INSIDE the fit half only, pooled over channels
    and columns. Fixed rule (PREREGISTRATION amendment 2026-11-16b); the second half is never used to choose it."""
    edges = np.linspace(0, n_fit, folds + 1).astype(int)
    idx = np.arange(n_fit)
    best, best_err = RIDGES[0], np.inf
    for lam in RIDGES:
        err = 0.0
        for i in range(folds):
            ev = idx[edges[i]:edges[i + 1]]
            err += _sse(X, Y, np.setdiff1d(idx, ev), ev, lam, n_fit)[0].sum()
        if err < best_err:
            best, best_err = lam, err
    return best


def oos_r2(X: np.ndarray, Y: np.ndarray, n_fit: int, ridge: float | None = None):
    """Per-channel ridge of Y (n, k, C) on X (n, C, p), fitted on rows < n_fit, R^2 per column k on rows >= n_fit,
    pooled over channels (denominator: deviations from the fit-half mean). Returns (R^2 per column, ridge)."""
    lam = choose_ridge(X, Y, n_fit) if ridge is None else ridge
    sse, sst, _ = _sse(X, Y, np.arange(n_fit), np.arange(n_fit, len(Y)), lam, n_fit)
    return 1.0 - sse / np.maximum(sst, 1e-300), lam


def fit_resid(X: np.ndarray, Y: np.ndarray, n_fit: int) -> np.ndarray:
    """Residual (all rows) of the per-channel ridge of Y on X fitted on rows < n_fit (CV-chosen penalty)."""
    lam = choose_ridge(X, Y, n_fit)
    return _sse(X, Y, np.arange(n_fit), np.arange(len(Y)), lam, n_fit)[2]


def online_r2(X: np.ndarray, Y: np.ndarray, n_fit: int, ridge: float) -> np.ndarray:
    """EXPLORATORY (not pre-registered): prequential R^2 on the same rows >= n_fit. Row t is predicted by the ridge
    fitted on ALL rows < t (expanding window, unpenalised intercept, the CV penalty of the static RPI); the
    denominator uses the running mean of rows < t. Measures what an online learner can exploit when the relation
    drifts, which the fixed-split RPI cannot."""
    n, k, C = Y.shape
    sse, sst = np.zeros(k), np.zeros(k)
    for c in range(C):
        Z = np.c_[np.ones(n), _design(X, c, n_fit)]
        Yc = Y[:, :, c]
        pen = ridge * np.eye(Z.shape[1])
        pen[0, 0] = 1e-8
        G = Z[:n_fit].T @ Z[:n_fit]
        b = Z[:n_fit].T @ Yc[:n_fit]
        sy, m = Yc[:n_fit].sum(0), n_fit
        for t in range(n_fit, n):
            W = np.linalg.solve(G + pen, b)
            sse += (Yc[t] - Z[t] @ W) ** 2
            sst += (Yc[t] - sy / m) ** 2
            G += np.outer(Z[t], Z[t])
            b += np.outer(Z[t], Yc[t])
            sy += Yc[t]
            m += 1
    return 1.0 - sse / np.maximum(sst, 1e-300)


def rpi(P: np.ndarray, Phi: np.ndarray, n_fit: int):
    """RPI_h, h = 2..H: revisions rho_{s+1,h} = P_{s+1,h-1} - P_{s,h} regressed on phi_s."""
    R = P[1:, :-1, :] - P[:-1, 1:, :]                      # row i = revision at origin i+1 (targets h = 2..H)
    return oos_r2(Phi[:-1], R.astype(np.float64), n_fit)            # (R^2_h, ridge)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--backbone", default="DLinear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dfac", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--tune-dfac", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--base", default="dric_l8")
    ap.add_argument("--top", default="vine_s")
    ap.add_argument("--out", type=Path, default=CODE_ROOT / "results" / "online_grid")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dump", type=Path, default=None, help="save forecasts and features (debugging)")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    device = setup_runtime(a.device, a.threads)
    set_seed(a.seed, True)
    data = load_dataset(a.dataset)
    prep = prepare_split(data, {"scheme": "fsnet"})
    y = x = prep.z
    L, H, C = a.L, a.H, prep.z.shape[1]
    pre = {"epochs": a.epochs, "lr": 1e-3, "batch_size": 32, "patience": 3}
    model, pinfo = pretrain_backbone(prep, {"name": a.backbone}, L, H, pre, device, a.seed,
                                     a.out.parent / "_checkpoints", data.name, "fsnet", verbose=False)
    if not pinfo.get("cached"):
        raise RuntimeError("backbone checkpoint not found: the replay must use the backbone of the main grid")
    season = int(data.meta.get("steps_per_day", 24))
    pinfo = {**pinfo, "steps_per_day": season, "dataset": data.name}
    cell = a.out / data.name / a.backbone / f"H{H}" / f"seed{a.seed}"
    tuning = json.loads((cell / "tuning.json").read_text())

    vfirst, vlast = max(prep.borders["train_end"] - 1, L - 1), prep.borders["val_end"] - H - 1
    ev0 = OnlineEvaluator(y, vfirst, vlast, L, H, delay=H, x=x)
    arch = ev0.make_archive(model, device, threads=a.threads)
    origins = np.arange(vfirst, vlast + 1)
    n = len(origins)
    n_fit = n // 2                                         # same split as tuning (score on the 2nd half)
    xs = np.stack([x[s - L + 1: s + 1] for s in origins])
    B = np.stack([arch.get(int(s), guard=False) for s in origins]).astype(np.float64)
    Y = np.stack([y[s + 1: s + 1 + H] for s in origins]).astype(np.float64)

    out = {"dataset": data.name, "backbone": a.backbone, "H": H, "seed": a.seed, "n_val_origins": int(n),
           "n_fit": int(n_fit), "base": a.base, "top": a.top, "rows": []}
    for f in a.dfac:
        d = f * H
        fd = max([g for g in a.tune_dfac if g <= f] or [min(a.tune_dfac)])
        specs = {k: resolve(tuning[m]["best_by_dfac"][str(fd)], tuning, fd) for k, m in (("F", a.base), ("G", a.top))}
        P = {"B": B}
        for k, spec in specs.items():
            ev = OnlineEvaluator(y, vfirst, vlast, L, H, delay=d, x=x, keep_preds=True)
            set_seed(a.seed, True)
            torch.set_num_threads(1)
            r = ev.run(build_method(dict(spec)), model, device=device, seed=a.seed, archive=arch, pretrain_info=pinfo)
            P[k] = r.preds.astype(np.float64)
        Phi = features(specs["G"]["top"], H, C, L, season, xs, P["F"], origins)
        if a.dump is not None:
            a.dump.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.dump / f"{data.name}_H{H}_s{a.seed}_d{d}.npz", Phi=Phi, Y=Y, **P)
        row = {"delay": d, "tuned_at_dfac": fd}
        for k in ("B", "F", "G"):
            r2, lam = rpi(P[k], Phi, n_fit - 1)
            row[f"rpi_{k}"] = float(r2.mean())
            row[f"ridge_{k}"] = lam
            R = (P[k][1:, :-1, :] - P[k][:-1, 1:, :]).astype(np.float64)
            row[f"orpi_{k}"] = float(online_r2(Phi[:-1], R, n_fit - 1, lam).mean())   # exploratory
            row[f"rpi_{k}_h"] = [float(v) for v in r2]
            row[f"val_mse_{k}"] = float(((P[k] - Y)[n_fit:] ** 2).mean())
        E = Y - P["F"]                                     # F's errors e_{s,h} (after the fact; diagnostic)
        eps = fit_resid(Phi, E, n_fit)
        nu, _ = oos_r2(Phi[:-1], eps[1:, :-1, :], n_fit - 1)  # eps_{s+1,h-1} on phi_s, h = 2..H
        row["nu_F"] = float(nu.mean())
        row["nu_F_h"] = [float(v) for v in nu]
        out["rows"].append(row)
        print(f"[{data.name} H={H} s{a.seed} d={d}] RPI B {row['rpi_B']:+.3f} F {row['rpi_F']:+.3f} "
              f"G {row['rpi_G']:+.3f}  online B {row['orpi_B']:+.3f} F {row['orpi_F']:+.3f} G {row['orpi_G']:+.3f}  "
              f"NU(F) {row['nu_F']:+.3f}  val MSE F {row['val_mse_F']:.4f} "
              f"G {row['val_mse_G']:.4f}", flush=True)
    (cell / "rpi.json").write_text(json.dumps(out, indent=1))
    print(f"[done] {cell / 'rpi.json'} in {time.perf_counter() - t0:.0f}s", flush=True)


def collect(root: Path, out: Path) -> None:
    """Flatten every rpi.json under root (results/online_grid) into one CSV (one row per cell, seed and delay)."""
    import csv
    keys = ["dataset", "H", "seed", "delay", "rpi_B", "rpi_F", "rpi_G", "orpi_B", "orpi_F", "orpi_G", "nu_F",
            "ridge_B", "ridge_F", "ridge_G", "val_mse_B", "val_mse_F", "val_mse_G", "n_val_origins"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for f in sorted(root.glob("*/*/H*/seed*/rpi.json")):
            j = json.loads(f.read_text())
            for r in j["rows"]:
                w.writerow({**{k: j[k] for k in ("dataset", "H", "seed", "n_val_origins")}, **r})


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "collect":       # python vine_rpi.py collect <root> <out.csv>
        collect(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
