"""Config-driven online experiment: pretrain backbone on the train split, then stream the val or test split.

Protocol keys (cfg["protocol"]):
  mode: strict | prefix | leaky          delays: [null, ...] (null -> H)  or  delay_factors: [1, 2, 4]
  prefix_lag: null (-> d - H) or int     allow_leaky / check_leaks / trace_memory / max_forecasts
  segment: test | val                    (val = the 5% validation segment as a mini online stream for tuning)
  provisional: {phi: 0.9, sd_frac: 0.3, seed: 12345}   (two-channel stress test; inputs != labels)
Runtime keys (cfg["runtime"]): device, threads (pretraining/batched inference), online_threads (per-step
loop; default 1 = single-core latency as in the methodology), deterministic.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from ..data import load_dataset, prepare_split
from ..models import pretrain_backbone
from ..utils import CODE_ROOT, environment_info, save_json, set_seed, setup_runtime
from .evaluator import OnlineEvaluator
from .methods import build_method
from .provisional import make_provisional

SUMMARY_FIELDS = ["experiment", "dataset", "backbone", "segment", "H", "delay", "mode", "prefix_lag", "method",
                  "seed", "mse", "mae", "n_forecasts", "update_latency_us_median", "update_latency_us_mean",
                  "update_latency_us_p95", "predict_latency_us_mean", "total_update_s", "state_bytes_final",
                  "state_learned_bytes", "state_pending_bytes", "state_raw_bytes", "peak_rss_mb", "wall_s",
                  "provisional", "leaky"]


def _append_csv(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _delays(cfg_proto: dict, H: int) -> list[int]:
    if cfg_proto.get("delay_factors"):
        return [int(round(f * H)) for f in cfg_proto["delay_factors"]]
    ds = cfg_proto.get("delays", [None])
    return [H if d is None else int(d) for d in ds]


def run_online(cfg: dict, seed: int, out_root: Path | None = None, verbose: bool = True) -> list[dict]:
    rt = cfg.get("runtime", {})
    device = setup_runtime(rt.get("device", "auto"), rt.get("threads"))
    batch_threads = torch.get_num_threads()
    online_threads = rt.get("online_threads", 1)
    set_seed(seed, rt.get("deterministic", True))
    out_root = Path(out_root or cfg.get("output", {}).get("dir", CODE_ROOT / "results")) / cfg["name"]
    dcfg = dict(cfg["dataset"])
    data = load_dataset(dcfg.pop("name"), channels=dcfg.pop("channels", None), max_rows=dcfg.pop("max_rows", None),
                        path=dcfg.pop("path", None), **dcfg)
    split = cfg.get("split", {"scheme": "fsnet"})
    prep = prepare_split(data, split)
    L = int(cfg["L"])
    proto = cfg.get("protocol", {})
    y = prep.z
    x = y
    if proto.get("provisional"):
        pv = proto["provisional"]
        x = make_provisional(y, float(pv.get("phi", 0.9)), float(pv.get("sd_frac", 0.3)), int(pv.get("seed", 12345)),
                             ref_rows=slice(0, prep.borders["train_end"]))
    segment = proto.get("segment", "test")
    rows = []
    for H in cfg["horizons"]:
        H = int(H)
        t0 = time.perf_counter()
        bb_cfg = cfg["backbone"]
        model, pinfo = pretrain_backbone(prep, bb_cfg, L, H, cfg.get("pretrain", {}), device, seed,
                                         out_root.parent / "_checkpoints", data.name, split.get("scheme", "?"),
                                         verbose=verbose)
        pinfo = {**pinfo, "steps_per_day": data.meta.get("steps_per_day", 24)}
        if verbose:
            print(f"[{data.name} H={H} seed={seed}] backbone ready in {time.perf_counter() - t0:.1f}s "
                  f"(cached={pinfo.get('cached')})", flush=True)
        if segment == "test":
            first, seg_end = prep.borders["val_end"] - 1, prep.borders["test_end"]
        elif segment == "val":
            first, seg_end = prep.borders["train_end"] - 1, prep.borders["val_end"]
        else:
            raise ValueError("protocol.segment must be test|val")
        first = max(first, L - 1)
        last = seg_end - H - 1
        if proto.get("max_forecasts"):
            last = min(last, first + int(proto["max_forecasts"]) - 1)
        for d in _delays(proto, H):
            ev = OnlineEvaluator(y, first, last, L, H, delay=d, mode=proto.get("mode", "strict"),
                                 allow_leaky=bool(proto.get("allow_leaky", False)),
                                 check_leaks=bool(proto.get("check_leaks", True)),
                                 trace_memory=bool(proto.get("trace_memory", False)), x=x,
                                 prefix_lag=proto.get("prefix_lag"))
            archive = ev.make_archive(model, device, threads=batch_threads,
                                      max_cache_bytes=float(rt.get("forecast_cache_bytes", 1.5e9)))
            for mspec in cfg["methods"]:
                method = build_method(mspec)
                set_seed(seed, rt.get("deterministic", True))
                torch.set_num_threads(int(online_threads) if online_threads else batch_threads)
                res = ev.run(method, model, device=device, seed=seed, archive=archive, pretrain_info=pinfo,
                             verbose=verbose)
                torch.set_num_threads(batch_threads)
                summ = res.summary()
                tag = ev.mode + (f"_lag{ev.lag}" if ev.mode == "prefix" else "") + \
                    ("_LEAKY" if ev.mode == "leaky" else "") + ("_provisional" if x is not y else "")
                odir = out_root / data.name / segment / f"H{H}" / f"d{d}_{tag}" / res.method / f"seed{seed}"
                save_json({"summary": summ, "timing": res.timing, "pretrain": pinfo, "config": cfg,
                           "method_spec": mspec, "seed": seed, "env": environment_info()}, odir / "metrics.json")
                if cfg.get("output", {}).get("save_steps", True):
                    np.savez_compressed(odir / "steps.npz", origins=res.origins, se=res.se.astype(np.float32),
                                        ae=res.ae.astype(np.float32))
                cats = res.state_bytes.get("categories", {})
                row = {"experiment": cfg["name"], "dataset": data.name, "backbone": bb_cfg["name"],
                       "segment": segment, "H": H, "delay": d, "mode": ev.mode, "prefix_lag": ev.lag,
                       "method": res.method, "seed": seed, **{k: summ.get(k) for k in SUMMARY_FIELDS if k in summ},
                       "state_learned_bytes": cats.get("learned"), "state_pending_bytes": cats.get("pending"),
                       "state_raw_bytes": cats.get("raw"), "wall_s": res.extra.get("wall_s"),
                       "provisional": x is not y, "leaky": ev.mode == "leaky"}
                _append_csv(out_root / "summary.csv", row)
                rows.append(row)
                if verbose:
                    print(f"  d={d:4d} {res.method:>14s}  MSE {summ['mse']:.4f}  MAE {summ['mae']:.4f}  "
                          f"upd(med) {summ['update_latency_us_median']:8.1f}us  "
                          f"state {summ['state_bytes_final'] / 1024:8.1f}KiB  wall {res.extra.get('wall_s', 0):.1f}s",
                          flush=True)
    return rows
