"""Chronological train/val/test borders and train-only standardisation.

Split schemes (`SplitSpec.scheme`):
  * "standard"  : Autoformer/TSLib convention. ETT*: 12/4/4 months; others 70/10/20 (val = remainder).
  * "fsnet"     : 20:5:75 (FSNet / OneNet / Proceed online protocol) as ratios of the full length,
                  i.e. num_train=int(0.2T), num_test=int(0.75T), num_val=T-train-test.
  * "fsnet_ett_months": FSNet-style 4/1/15 months for ETT (first 20 months only). Provided because some
                  public codebases use month borders for ETT; check which one the paper you compare against uses.
  * "audit"     : 25:5:70 (delay-aware audit, arXiv 2610.12232) for direct comparability with its numbers.
  * "ratio"     : arbitrary (train, val, test) ratios.

The scaler is ALWAYS fitted on the training rows only (rows [0, train_end)).
`border` semantics follow TSLib: segment k covers rows [b1[k], b2[k]) where b1 for val/test is moved back by
`lookback` rows so that the first window of the segment has a full input (inputs may come from the previous
segment, targets never do).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import TimeSeriesData


@dataclass
class SplitSpec:
    scheme: str = "standard"
    ratios: tuple[float, float, float] | None = None  # for scheme="ratio"


def compute_borders(T: int, spec: SplitSpec, is_ett: bool = False, steps_per_day: int = 24) -> dict:
    """Return dict with train_end, val_end, test_end (exclusive row indices)."""
    s = spec.scheme
    if s == "standard":
        if is_ett:
            m = 30 * steps_per_day
            tr, va, te = 12 * m, 16 * m, 20 * m
        else:
            n_tr, n_te = int(T * 0.7), int(T * 0.2)
            tr, va, te = n_tr, T - n_te, T
    elif s == "fsnet":
        n_tr, n_te = int(T * 0.2), int(T * 0.75)
        tr, va, te = n_tr, T - n_te, T
    elif s == "audit":
        n_tr, n_te = int(T * 0.25), int(T * 0.70)
        tr, va, te = n_tr, T - n_te, T
    elif s == "fsnet_ett_months":
        if not is_ett:
            raise ValueError("fsnet_ett_months only defined for ETT datasets")
        m = 30 * steps_per_day
        tr, va, te = 4 * m, 5 * m, 20 * m
    elif s == "ratio":
        r = np.asarray(spec.ratios, dtype=float)
        r = r / r.sum()
        tr = int(T * r[0])
        te_len = int(T * r[2])
        va = T - te_len
        te = T
    else:
        raise ValueError(f"unknown split scheme {s}")
    te = min(te, T)
    assert 0 < tr < va < te <= T, (tr, va, te, T)
    return {"train_end": int(tr), "val_end": int(va), "test_end": int(te)}


class StandardScaler:
    """Per-channel z-score; fitted on training rows only. float32 throughout."""

    def __init__(self, eps: float = 1e-8):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean = x.mean(0).astype(np.float32)
        std = x.std(0).astype(np.float32)
        self.std = np.where(std < self.eps, 1.0, std).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class PreparedSplit:
    data: TimeSeriesData
    z: np.ndarray              # (T', C) standardized, float32 (T' = test_end)
    borders: dict
    scaler: StandardScaler

    def segment(self, which: str, lookback: int) -> tuple[int, int]:
        """Row range [start, end) for windows of a segment, including `lookback` history rows."""
        b = self.borders
        if which == "train":
            return 0, b["train_end"]
        if which == "val":
            return max(0, b["train_end"] - lookback), b["val_end"]
        if which == "test":
            return max(0, b["val_end"] - lookback), b["test_end"]
        raise KeyError(which)


def prepare_split(data: TimeSeriesData, spec: SplitSpec | dict | str) -> PreparedSplit:
    if isinstance(spec, str):
        spec = SplitSpec(scheme=spec)
    elif isinstance(spec, dict):
        spec = SplitSpec(**spec)
    borders = compute_borders(data.T, spec, is_ett=bool(data.meta.get("ett")),
                              steps_per_day=int(data.meta.get("steps_per_day", 24)))
    raw = data.values[: borders["test_end"]]
    scaler = StandardScaler().fit(raw[: borders["train_end"]])  # train rows only
    z = scaler.transform(raw)
    return PreparedSplit(data, z, borders, scaler)
