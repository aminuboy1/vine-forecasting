"""Offline (pre)training on the train split with validation early stopping.

Checkpoints are cached by a hash of (dataset, split, model config, L, H, seed, train config) so that
every online method in a sweep uses the *same* frozen backbone (the audit found backbone recipe can
change frozen online error by up to 25x, so the recipe is logged with every result).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.windows import iterate_batches, make_windows
from ..utils import config_hash, count_params, save_json
from .backbones import build_backbone


@torch.no_grad()
def evaluate_mse(model, X, Y, batch_size, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in iterate_batches(X, Y, batch_size, False, None, device):
        tot += float(((model(xb) - yb) ** 2).sum())
        n += yb.numel()
    return tot / max(n, 1)


def fit(model: nn.Module, X, Y, Xv=None, Yv=None, *, epochs=10, lr=1e-3, batch_size=32, patience=3,
        weight_decay=0.0, device="cpu", seed=0, max_steps_per_epoch=None, grad_clip=None, verbose=True,
        lr_decay=0.5) -> dict:
    """Plain MSE training loop (Adam), early stopping on validation MSE, returns history."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        t0, tl, nb = time.perf_counter(), 0.0, 0
        for xb, yb in iterate_batches(X, Y, batch_size, True, rng, device):
            opt.zero_grad(set_to_none=True)
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tl += float(loss.detach())
            nb += 1
            if max_steps_per_epoch and nb >= max_steps_per_epoch:
                break
        vl = evaluate_mse(model, Xv, Yv, 256, device) if Xv is not None and len(Xv) else tl / max(nb, 1)
        hist.append({"epoch": ep, "train_mse": tl / max(nb, 1), "val_mse": vl, "sec": time.perf_counter() - t0})
        if verbose:
            print(f"  epoch {ep:3d} train {tl / max(nb, 1):.4f} val {vl:.4f} ({hist[-1]['sec']:.1f}s)", flush=True)
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            for g in opt.param_groups:  # TSLib 'type1'-like decay on plateau
                g["lr"] *= lr_decay
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": hist, "best_val_mse": best}


def pretrain_backbone(prep, model_cfg: dict, L: int, H: int, train_cfg: dict, device, seed: int,
                      ckpt_dir: Path, dataset_name: str, split_scheme: str, verbose: bool = True):
    """Train (or load cached) backbone on the train split; returns (model, info)."""
    key = {"dataset": dataset_name, "split": split_scheme, "model": model_cfg, "L": L, "H": H,
           "train": train_cfg, "seed": seed, "T": int(prep.z.shape[0])}
    h = config_hash(key)
    ckpt = Path(ckpt_dir) / f"{dataset_name}_{model_cfg['name']}_L{L}_H{H}_s{seed}_{h}.pt"
    torch.manual_seed(seed)
    model = build_backbone(model_cfg["name"], L, H, **{k: v for k, v in model_cfg.items() if k != "name"}).to(device)
    if ckpt.exists() and not train_cfg.get("force", False):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        info = {"checkpoint": str(ckpt), "cached": True}
        if ckpt.with_suffix(".json").exists():
            import json
            info.update(json.loads(ckpt.with_suffix(".json").read_text()))
        info["cached"] = True
        return model, info
    tr = prep.segment("train", L)
    va = prep.segment("val", L)
    X, Y, _ = make_windows(prep.z, *tr, L, H, stride=int(train_cfg.get("stride", 1)))
    Xv, Yv, _ = make_windows(prep.z, *va, L, H)
    t0 = time.perf_counter()
    if model_cfg["name"] == "Naive" or int(train_cfg.get("epochs", 10)) == 0:
        res = {"history": [], "best_val_mse": evaluate_mse(model, Xv, Yv, 256, device) if len(Xv) else None}
    else:
        res = fit(model, X, Y, Xv, Yv, epochs=int(train_cfg.get("epochs", 10)), lr=float(train_cfg.get("lr", 1e-3)),
                  batch_size=int(train_cfg.get("batch_size", 32)), patience=int(train_cfg.get("patience", 3)),
                  device=device, seed=seed, max_steps_per_epoch=train_cfg.get("max_steps_per_epoch"),
                  verbose=verbose)
    info = {"checkpoint": str(ckpt), "cached": False, "train_windows": int(len(X)), "val_windows": int(len(Xv)),
            "pretrain_seconds": time.perf_counter() - t0, "n_params": count_params(model), **res, "key": key}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    save_json(info, ckpt.with_suffix(".json"))
    return model, info
