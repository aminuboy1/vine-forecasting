"""Experiment runner.

    python run.py --config configs/online/smoke_etth2.yaml
    python run.py --config configs/online/etth2_main.yaml --seeds 0 1 2 --set runtime.threads=4

The YAML must set `harness: online`. Results go to <output.dir>/<name>/ (default
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
        else:
            raise SystemExit(f"config must set harness: online (got {harness!r}); the continual harness of the "
                             "companion SOMA paper lives in its own repository")
        per_seed[s] = time.perf_counter() - ts
        print(f"== seed {s} done in {per_seed[s]:.1f}s", flush=True)
    wall = time.perf_counter() - t0
    save_json({"wall_seconds_total": wall, "wall_seconds_per_seed": per_seed}, run_dir / "timing.json")
    print(f"== all done in {wall:.1f}s -> {run_dir}")


if __name__ == "__main__":
    main()
