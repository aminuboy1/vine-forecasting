"""Experiment runner.

    python run.py --config configs/online/smoke_etth2.yaml
    python run.py --config configs/online/etth2_main.yaml --seeds 0 1 2 --set runtime.threads=4
    python run.py --config configs/continual/smoke_etth2.yaml --set L=48 --set H=12

The YAML must set `harness: online | continual`. Results go to <output.dir>/<name>/ (default
code/results/<name>/): JSON per run, summary.csv, and (online) per-step losses in steps.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsl.utils import CODE_ROOT, apply_overrides, environment_info, load_config, save_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="overrides cfg.seeds")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value (YAML)")
    ap.add_argument("--out", type=Path, default=None, help="results root (default cfg.output.dir or code/results)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    out = args.out or Path(cfg.get("output", {}).get("dir", CODE_ROOT / "results"))
    if not out.is_absolute():
        out = CODE_ROOT / out
    harness = cfg.get("harness")
    t0 = time.perf_counter()
    run_dir = out / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "seeds": seeds, "argv": sys.argv, "env": environment_info()}, run_dir / "run_config.json")
    per_seed = {}
    for s in seeds:
        ts = time.perf_counter()
        if harness == "online":
            from tsl.online.experiment import run_online
            run_online(cfg, s, out, verbose=not args.quiet)
        elif harness == "continual":
            from tsl.continual.experiment import run_cl
            run_cl(cfg, s, out, verbose=not args.quiet)
        else:
            raise SystemExit(f"config must set harness: online|continual (got {harness!r})")
        per_seed[s] = time.perf_counter() - ts
        print(f"== seed {s} done in {per_seed[s]:.1f}s", flush=True)
    if harness == "continual" and len(seeds) >= 1:
        _aggregate_bps(run_dir)
    wall = time.perf_counter() - t0
    save_json({"wall_seconds_total": wall, "wall_seconds_per_seed": per_seed}, run_dir / "timing.json")
    print(f"== all done in {wall:.1f}s -> {run_dir}")


def _aggregate_bps(run_dir: Path):
    """Seed-averaged BPS-style analysis per dataset."""
    import numpy as np
    from tsl.continual.metrics import boundary_profile_sensitivity
    for ds_dir in [p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]:
        acc: dict = {}
        for f in ds_dir.glob("*/*/seed*/result.json"):
            r = json.loads(f.read_text())
            tk, lr = f.parts[-4], f.parts[-3]
            acc.setdefault(tk, {}).setdefault(lr, []).append(r["metrics_mse"])
        if len(acc) < 2:
            continue
        avg = {}
        for tk, d in acc.items():
            avg[tk] = {}
            for lr, ms in d.items():
                keys = [k for k in ms[0] if isinstance(ms[0][k], (int, float))]
                avg[tk][lr] = {k: float(np.mean([m[k] for m in ms if k in m])) for k in keys}
                avg[tk][lr]["avg_seen_curve"] = np.mean([m["avg_seen_curve"] for m in ms], 0).tolist()
        save_json(boundary_profile_sensitivity(avg), ds_dir / "bps_seed_mean.json")


if __name__ == "__main__":
    main()
