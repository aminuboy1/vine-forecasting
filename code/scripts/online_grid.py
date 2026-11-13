"""Paper 2 main grid, one cell = (dataset, H, backbone, seed). Pre-registered protocol (PREREGISTRATION.md):

  1. pretrain (or load the cached) backbone on the train split;
  2. TUNE every corrector on the 5% validation segment run as a mini online stream under the same protocol:
     the published/default config + random draws, <= `--budget` configs per method, the same RNG seed for every
     method; score = mean validation MSE over d in {H, 2H, 4H}; the chosen configs are written to disk BEFORE
  3. the test segment is run once per method and delay d in {H, 2H, 4H, 8H} (per-origin losses saved for tests).

    python scripts/online_grid.py --dataset ETTh2 --H 24 --backbone DLinear --seed 0
Outputs: results/online_grid/<dataset>/<backbone>/H<H>/seed<k>/{tuning.json, test.csv, steps_<method>_d<d>.npz}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import copy
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsl.data import load_dataset, prepare_split          # noqa: E402
from tsl.models import pretrain_backbone                  # noqa: E402
from tsl.online import OnlineEvaluator, build_method      # noqa: E402
from tsl.utils import CODE_ROOT, environment_info, set_seed, setup_runtime   # noqa: E402

# ---------------------------------------------------------------------------------------- search spaces
# First entry of every method = its published / default configuration (always evaluated).
METHODS = {
    "frozen": ({"type": "frozen"}, {}),
    "persist_last": ({"type": "persistence", "kind": "last"}, {}),
    "persist_seasonal": ({"type": "persistence", "kind": "seasonal"}, {}),
    "rls_audit": ({"type": "rls_audit"}, {}),                                   # published bank: no free knobs
    "elf": ({"type": "elf"}, {"ridge": [2.0, 20.0, 200.0], "update_every": [50, 100, 200, 400],
                             "eta": [0.1, 0.5, 2.0], "alpha": [0.7, 0.9, 1.0]}),
    "dric_l8": ({"type": "dric", "variant": "l8", "phases": "all"},
                {"ridge": [0.1, 1.0, 10.0], "half_life": [32.0, 64.0, 128.0, 256.0, 512.0], "tau": [5.0, 10.0, 20.0]}),
    "dric_g": ({"type": "dric", "variant": "g", "phases": "all"},
               {"ridge": [0.1, 1.0, 10.0], "half_life": [32.0, 64.0, 128.0, 256.0, 512.0], "tau": [5.0, 10.0, 20.0]}),
    "epoc": ({"type": "epoc", "phases": "all"},
             {"ridge": [0.1, 1.0, 10.0], "half_life": [32.0, 64.0, 128.0, 256.0, 512.0]}),
    "mc_ls": ({"type": "mc_ls"}, {"gamma_m": [0.995, 0.999, 0.9995, 0.9998], "ridge": [1.0, 10.0, 100.0],
                                  "gate": [0, 32], "k_lb": [0, 4], "clip": [3.0, 5.0]}),
    "mc_ls_pr": ({"type": "mc_ls_pr"}, {"gamma_m": [0.995, 0.999, 0.9995, 0.9998], "ridge": [1.0, 10.0, 100.0],
                                        "gate": [0, 32], "k_lb": [0, 4], "clip": [3.0, 5.0]}),
    "rev_regress": ({"type": "mc_ls_pr", "k_f": 0, "use_level": False, "use_ewma": False},
                    {"gamma_m": [0.995, 0.999, 0.9995, 0.9998], "ridge": [1.0, 10.0, 100.0], "gate": [0, 32]}),
    "vine_teacher": ({"type": "vine", "target": "teacher"},
                     {"gamma": [0.995, 0.999, 0.9995, 0.9998], "gamma_a": [0.995, 0.999, 0.9998],
                      "ridge": [1.0, 10.0, 100.0], "gate": [0, 32], "k_lb": [0, 4], "clip": [3.0, 5.0]}),
    "vine": ({"type": "vine"}, {"gamma": [0.995, 0.999, 0.9995, 0.9998], "gamma_a": [0.995, 0.999, 0.9998],
                                "ridge": [1.0, 10.0, 100.0], "gate": [0, 32], "k_lb": [0, 4], "clip": [3.0, 5.0],
                                "kappa": [0.8, 0.9, 0.95, 1.0]}),
    "vine_lam": ({"type": "vine_lam"}, {"lam": [0.25, 0.5, 0.75, 0.9], "gamma": [0.995, 0.999, 0.9998],
                                        "ridge": [1.0, 10.0, 100.0], "gate": [0, 32], "k_lb": [0, 4]}),
    "safe_mix": ({"type": "safe_mix"}, {"eta": [0.05, 0.2, 0.5, 2.0], "share": [1e-3, 1e-2]}),
    # combiner over LABEL-ONLY experts (isolates what the VINE experts add to the mixture; reviewer M3)
    "safe_mix_lo": ({"type": "safe_mix", "experts": [{"type": "frozen"}, {"type": "mc_ls_pr"},
                                                     {"type": "mc_ls_pr", "gamma_m": 0.999}, {"type": "rls_audit"}]},
                    {"eta": [0.05, 0.2, 0.5, 2.0], "share": [1e-3, 1e-2]}),
    "dric_l8_single": ({"type": "dric", "variant": "l8", "phases": "single"}, {}),   # as published (stride-H grid)
    "ogd_head": ({"type": "ogd", "scope": "head", "lr": 1e-4, "batch": 1}, {"lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]}),
    "periodic": ({"type": "periodic", "period_frac": 0.1, "epochs": 1, "lr": 1e-4, "max_steps": 500},
                 {"lr": [3e-5, 1e-4, 3e-4]}),
    # ---- amendment 2026-11-14 (frozen before any confirmation-stream result): revision learning as a plug-in layer.
    # Memory of the VINE family and of its same-feature controls is FIXED long (gamma = 0.9998, half-life ~3466 steps
    # ~ DRiC's published 128-block half-life at H = 24): the 5% validation stream cannot identify long memories.
    # Headline: VINE-lambda stacked on the validation-tuned DRiC-L8.
    "vine_s": ({"type": "stack", "base": "@dric_l8",
                "top": {"type": "vine_lam", "gamma": 0.9998, "gamma_a": 0.9998}},
               {"top.lam": [0.25, 0.5, 0.75, 0.9], "top.ridge": [1.0, 10.0, 100.0], "top.gate": [0, 32],
                "top.k_lb": [0, 4]}),
    # control: identical stack, base, features and memory, label-only top layer (isolates revision learning)
    "vine_s_lo": ({"type": "stack", "base": "@dric_l8", "top": {"type": "mc_ls_pr", "gamma_m": 0.9998}},
                  {"top.ridge": [1.0, 10.0, 100.0], "top.gate": [0, 32], "top.k_lb": [0, 4], "top.clip": [3.0, 5.0]}),
    # ablation: pure revision fixed point on top (lambda = 1, kappa tuned)
    "vine_s_pure": ({"type": "stack", "base": "@dric_l8", "top": {"type": "vine", "gamma": 0.9998, "gamma_a": 0.9998}},
                    {"top.ridge": [1.0, 10.0, 100.0], "top.gate": [0, 32], "top.k_lb": [0, 4],
                     "top.kappa": [0.8, 0.9, 0.95, 1.0]}),
    # ablation: the same top layer without the base (does stacking matter?)
    "vine_lam_lm": ({"type": "vine_lam", "gamma": 0.9998, "gamma_a": 0.9998},
                    {"lam": [0.25, 0.5, 0.75, 0.9], "ridge": [1.0, 10.0, 100.0], "gate": [0, 32], "k_lb": [0, 4]}),
}
CODE_VERSION = "2026-11-13c"      # bump when a method's code changes, to invalidate cached tuning
WIDE_SKIP = {"dric_l8", "ogd_head", "periodic"}           # H copies of per-channel learners: too heavy for C > 300 (DRiC-G is kept)


def nest(cfg: dict) -> dict:
    """Dotted keys ("top.lam": 0.5) -> nested specs (for stacked methods)."""
    out = copy.deepcopy({k: v for k, v in cfg.items() if "." not in k})
    for k, v in cfg.items():
        if "." in k:
            node = out
            *path, leaf = k.split(".")
            for q in path:
                node = node.setdefault(q, {})
            node[leaf] = v
    return out


def space_hash(name: str, a) -> str:
    """Hash of a method's search space (+ the hashes of the methods it references)."""
    default, space = METHODS[name]
    rs = sorted(set(refs(default)) | set(refs(space)))
    return hashlib.sha256(json.dumps([default, space, a.budget, a.tune_dfac, CODE_VERSION]
                                     + [space_hash(r, a) for r in rs], sort_keys=True).encode()).hexdigest()[:16]


def refs(obj) -> list[str]:
    """Names referenced as "@name" (= that method's validation-tuned configuration at the same delay)."""
    if isinstance(obj, str):
        return [obj[1:]] if obj.startswith("@") else []
    if isinstance(obj, dict):
        return [r for v in obj.values() for r in refs(v)]
    if isinstance(obj, list):
        return [r for v in obj for r in refs(v)]
    return []


def resolve(obj, tuning: dict, fd: int):
    if isinstance(obj, str) and obj.startswith("@"):
        return resolve(copy.deepcopy(tuning[obj[1:]]["best_by_dfac"][str(fd)]), tuning, fd)
    if isinstance(obj, dict):
        return {k: resolve(v, tuning, fd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, tuning, fd) for v in obj]
    return obj


def sample_configs(default: dict, space: dict, budget: int, seed: int) -> list[dict]:
    out = [dict(default)]
    if not space:
        return out
    keys = sorted(space)
    grid = list(itertools.product(*[space[k] for k in keys]))
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(grid)):
        if len(out) >= budget:
            break
        cfg = nest({**default, **dict(zip(keys, grid[i]))})
        if cfg not in out:
            out.append(cfg)
    return out


def run_one(y, x, first, last, L, H, d, spec, model, device, pinfo, archive, seed):
    ev = OnlineEvaluator(y, first, last, L, H, delay=d, x=x)
    set_seed(seed, True)
    torch.set_num_threads(1)
    return ev.run(build_method(dict(spec)), model, device=device, seed=seed, archive=archive, pretrain_info=pinfo)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--backbone", default="DLinear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--test-dfac", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--tune-dfac", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--out", type=Path, default=CODE_ROOT / "results" / "online_grid")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-warm-val", dest="warm_val", action="store_false",
                    help="start the test run cold at the test segment (default: run through validation, score test)")
    a = ap.parse_args(argv)

    device = setup_runtime(a.device, a.threads)
    set_seed(a.seed, True)
    data = load_dataset(a.dataset)
    prep = prepare_split(data, {"scheme": "fsnet"})
    y = x = prep.z
    L, H = a.L, a.H
    bb = {"name": a.backbone}
    pre = {"epochs": a.epochs, "lr": 1e-3, "batch_size": 32, "patience": 3}
    cell = a.out / data.name / a.backbone / f"H{H}" / f"seed{a.seed}"
    cell.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.set_num_threads(a.threads)
    model, pinfo = pretrain_backbone(prep, bb, L, H, pre, device, a.seed, a.out.parent / "_checkpoints", data.name,
                                     "fsnet", verbose=False)
    pinfo = {**pinfo, "steps_per_day": data.meta.get("steps_per_day", 24)}
    skip = WIDE_SKIP if data.values.shape[1] > 300 else set()
    names = list(a.methods or [m for m in METHODS if m not in skip])
    for m in list(names):        # a referenced method is tuned first, and dropped together with its dependants
        rs = refs(METHODS[m][0]) + refs(METHODS[m][1])
        if any(r in skip for r in rs):
            names.remove(m)
            continue
        for r in rs:
            if r not in names:
                names.insert(names.index(m), r)
    print(f"[{data.name} {a.backbone} H={H} seed={a.seed}] backbone {time.perf_counter() - t0:.0f}s; methods {names}",
          flush=True)

    # ------------------------------------------------------------------ 1) tuning on the validation stream
    tfile = cell / "tuning.json"
    tuning = json.loads(tfile.read_text()) if tfile.exists() else {}
    vfirst, vlast = max(prep.borders["train_end"] - 1, L - 1), prep.borders["val_end"] - H - 1
    varch = OnlineEvaluator(y, vfirst, vlast, L, H, delay=H, x=x).make_archive(model, device, threads=a.threads)
    burn = (vlast - vfirst + 1) // 2                     # score only the 2nd half of the validation stream (M1)
    for name in names:
        default, space = METHODS[name]
        rs = sorted(set(refs(default)) | set(refs(space)))
        stale = [r for r in rs if r not in tuning or tuning[r].get("space_hash") != space_hash(r, a)]
        if stale:
            raise RuntimeError(f"{name} references methods without current tuning {stale}")
        h = space_hash(name, a)
        if name in tuning and tuning[name].get("space_hash") == h:
            continue
        cands = sample_configs(default, space, a.budget, seed=12345)
        per_d = {f: [] for f in a.tune_dfac}
        for cfg in cands:
            for f in a.tune_dfac:
                spec_f = resolve(cfg, tuning, f)                 # outside the try: reference errors must surface
                try:
                    r = run_one(y, x, vfirst, vlast, L, H, f * H, spec_f, model, device, pinfo, varch, a.seed)
                    per_d[f].append(float(r.se[burn:].mean()))
                except Exception as e:                       # a config that crashes is scored as invalid
                    per_d[f].append(float("inf"))
                    print(f"   tuning {name} {cfg} failed: {e!r}", flush=True)
        best = {str(f): int(np.argmin(v)) for f, v in per_d.items()}          # tuned PER DELAY (M2)
        tuning[name] = {"space_hash": h, "best_by_dfac": {f: cands[i] for f, i in best.items()},
                        "best_val_mse_by_dfac": {f: per_d[int(f)][i] for f, i in best.items()},
                        "default_val_mse_by_dfac": {str(f): v[0] for f, v in per_d.items()},
                        "n_configs": len(cands), "burn_in_origins": burn,
                        "all": [{"cfg": c, **{f"val_mse_d{f}": per_d[f][k] for f in a.tune_dfac}}
                                for k, c in enumerate(cands)]}
        tfile.write_text(json.dumps(tuning, indent=1))
        print(f"   tuned {name:>16s}: " + " ".join(f"d{f}H {per_d[f][best[str(f)]]:.4f}/{per_d[f][0]:.4f}"
                                                   for f in a.tune_dfac) + f" (best/default, {len(cands)} cfgs)",
              flush=True)

    # ------------------------------------------------------------------ 2) test stream (configs frozen above)
    # every method starts at the beginning of the validation segment (deployed with the backbone) and is scored on
    # the test origins only (identical for all methods; avoids cold-start artefacts; test never used for tuning)
    tfirst = max(prep.borders["val_end"] - 1, L - 1)
    first = vfirst if a.warm_val else tfirst
    last = prep.borders["test_end"] - H - 1
    tarch = OnlineEvaluator(y, first, last, L, H, delay=H, x=x).make_archive(model, device, threads=a.threads)
    csv_path = cell / "test.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done = {(r["method"], int(r["delay"])) for r in csv.DictReader(f)}
    fields = ["dataset", "backbone", "H", "seed", "delay", "lag", "method", "mse", "mae", "n", "upd_us_med",
              "pred_us_mean", "step_us_mean", "step_us_med", "step_us_p95", "state_learned", "state_pending",
              "wall_s", "tuned_at_dfac", "config"]
    for f in a.test_dfac:
        d = f * H
        for name in names:
            if (name, d) in done:
                continue
            fd = max([g for g in a.tune_dfac if g <= f] or [min(a.tune_dfac)])
            spec = resolve(tuning[name]["best_by_dfac"][str(fd)], tuning, fd)
            r = run_one(y, x, first, last, L, H, d, spec, model, device, pinfo, tarch, a.seed)
            keep = r.origins >= tfirst
            se, ae, origins = r.se[keep], r.ae[keep], r.origins[keep]
            s = {**r.summary(), "mse": float(se.mean()), "mae": float(ae.mean())}
            tm = r.timing
            n = len(r.origins)
            upd_tot = tm.get("update", {}).get("total_s", 0.0)
            prd_tot = tm.get("predict", {}).get("total_s", 0.0)
            cats = r.state_bytes.get("categories", {})
            row = {"dataset": data.name, "backbone": a.backbone, "H": H, "seed": a.seed, "delay": d, "lag": d - H,
                   "method": name, "mse": s["mse"], "mae": s["mae"], "n": int(keep.sum()),
                   "upd_us_med": s.get("update_latency_us_median"), "pred_us_mean": s.get("predict_latency_us_mean"),
                   "step_us_mean": 1e6 * (upd_tot + prd_tot) / max(n, 1),
                   "step_us_med": tm.get("step", {}).get("median_us"), "step_us_p95": tm.get("step", {}).get("p95_us"),
                   "tuned_at_dfac": fd, "state_learned": cats.get("learned"),
                   "state_pending": cats.get("pending"), "wall_s": r.extra.get("wall_s"), "config": json.dumps(spec)}
            new = not csv_path.exists()
            with open(csv_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if new:
                    w.writeheader()
                w.writerow(row)
            np.savez_compressed(cell / f"steps_{name}_d{d}.npz", origins=origins, se=se.astype(np.float32),
                                ae=ae.astype(np.float32))
            print(f"   test d={d:4d} {name:>16s} MSE {s['mse']:.4f} MAE {s['mae']:.4f} "
                  f"step {row['step_us_mean']:.0f}us", flush=True)
    (cell / "env.json").write_text(json.dumps({"env": environment_info(), "pretrain": {k: v for k, v in pinfo.items()
                                                                                     if not hasattr(v, "shape")}},
                                              indent=1, default=str))
    print(f"[done] {cell} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
