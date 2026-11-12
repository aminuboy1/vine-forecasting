"""Seeding, device/thread setup, config handling, timing and memory measurement."""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- seeding / device
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001
            pass


def physical_cores() -> int:
    n = os.cpu_count() or 1
    # Hyper-threading: logical/2 is a good default for dense BLAS on Intel laptops.
    return max(1, n // 2)


def setup_runtime(device: str = "auto", threads: int | None = None) -> torch.device:
    """Pick the device and set CPU threads. `threads=None` -> physical cores (or $TSL_THREADS)."""
    if threads is None:
        threads = int(os.environ.get("TSL_THREADS", physical_cores()))
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(max(1, min(2, int(threads))))
    except RuntimeError:
        pass  # can only be set once per process
    torch.set_default_dtype(torch.float32)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


# --------------------------------------------------------------------------- config
# keys whose dict value REPLACES the inherited one instead of being merged (a grid is a set, not a patch)
REPLACE_KEYS = {"taskifications", "variables", "periods", "provisional"}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS:
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load YAML; supports `_base_: other.yaml` (relative path) for inheritance."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        bases = base if isinstance(base, list) else [base]
        merged: dict = {}
        for b in bases:
            merged = deep_update(merged, load_config(path.parent / b))
        cfg = deep_update(merged, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """CLI overrides `a.b.c=value` (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- provenance
def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


# --------------------------------------------------------------------------- timing / memory
class Timer:
    """Accumulating wall-clock timer; `with t.section("update"): ...`."""

    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(name, []).append(time.perf_counter() - t0)

    def summary(self) -> dict:
        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v)
            out[k] = {"n": int(a.size), "total_s": float(a.sum()), "mean_us": float(a.mean() * 1e6),
                      "median_us": float(np.median(a) * 1e6), "p95_us": float(np.percentile(a, 95) * 1e6)}
        return out


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process (bytes), cross-platform, no psutil needed."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            return int(pmc.PeakWorkingSetSize)
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)
    except Exception:  # noqa: BLE001
        return None


def current_rss_bytes() -> int | None:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if sys.platform == "win32":
            return peak_rss_bytes()  # best effort
    except Exception:  # noqa: BLE001
        return None
    return None


class ByteCount(int):
    """Explicit byte count for state that is not held as an array (e.g. retained raw history rows)."""


def nbytes_of(obj: Any) -> int:
    """Recursively count bytes of numpy arrays / tensors inside containers."""
    if obj is None:
        return 0
    if isinstance(obj, ByteCount):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, torch.nn.Module):
        return sum(nbytes_of(t) for t in obj.state_dict().values())
    if isinstance(obj, dict):
        return sum(nbytes_of(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes_of(v) for v in obj)
    if isinstance(obj, (int, float)):
        return 8
    return 0


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)
