"""Dataset registry and CSV loading.

All loaders return a `TimeSeriesData` with float32 values (T, C), a timestamp index and channel names.
Data root: $TSL_DATA or <code>/data. Standard files live under <root>/tslib/... (see scripts/download_tslib.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import CODE_ROOT

# name -> metadata. `ett` marks the ETT family (month-based standard borders).
DATASETS: dict[str, dict] = {
    "ETTh1": {"path": "tslib/ETT-small/ETTh1.csv", "freq": "h", "ett": True, "steps_per_day": 24},
    "ETTh2": {"path": "tslib/ETT-small/ETTh2.csv", "freq": "h", "ett": True, "steps_per_day": 24},
    "ETTm1": {"path": "tslib/ETT-small/ETTm1.csv", "freq": "15min", "ett": True, "steps_per_day": 96},
    "ETTm2": {"path": "tslib/ETT-small/ETTm2.csv", "freq": "15min", "ett": True, "steps_per_day": 96},
    "ECL": {"path": "tslib/electricity/electricity.csv", "freq": "h", "steps_per_day": 24},
    "Traffic": {"path": "tslib/traffic/traffic.csv", "freq": "h", "steps_per_day": 24},
    "Weather": {"path": "tslib/weather/weather.csv", "freq": "10min", "steps_per_day": 144},
    "Exchange": {"path": "tslib/exchange_rate/exchange_rate.csv", "freq": "D", "steps_per_day": 1},
}
ALIASES = {"electricity": "ECL", "traffic": "Traffic", "weather": "Weather", "exchange_rate": "Exchange",
           "exchange": "Exchange", "ecl": "ECL"}


def data_root() -> Path:
    return Path(os.environ.get("TSL_DATA", CODE_ROOT / "data"))


def canonical(name: str) -> str:
    if name in DATASETS:
        return name
    return ALIASES.get(name, ALIASES.get(name.lower(), name))


def dataset_path(name: str) -> Path:
    name = canonical(name)
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {list(DATASETS)}")
    return data_root() / DATASETS[name]["path"]


@dataclass
class TimeSeriesData:
    name: str
    values: np.ndarray                 # (T, C) float32, raw (unscaled)
    timestamps: pd.DatetimeIndex
    columns: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.values.shape[0]

    @property
    def C(self) -> int:
        return self.values.shape[1]


def load_csv(path: Path, name: str, date_col: str = "date", meta: dict | None = None,
             channels: list[str] | None = None, max_rows: int | None = None) -> TimeSeriesData:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_tslib.py --only {name}` (or set $TSL_DATA).")
    df = pd.read_csv(path, nrows=max_rows)
    ts = pd.to_datetime(df[date_col]) if date_col in df.columns else pd.RangeIndex(len(df))
    df = df.drop(columns=[date_col]) if date_col in df.columns else df
    if channels is not None:
        df = df[channels]
    vals = df.to_numpy(dtype=np.float32)
    if np.isnan(vals).any():
        # forward fill then zero; standard collection has no NaNs but custom data may
        vals = pd.DataFrame(vals).ffill().fillna(0.0).to_numpy(np.float32)
    return TimeSeriesData(name, vals, pd.DatetimeIndex(ts) if not isinstance(ts, pd.RangeIndex) else ts,
                          list(df.columns), dict(meta or {}))


def load_dataset(name: str, channels: list[str] | None = None, max_rows: int | None = None,
                 path: str | None = None, **kwargs) -> TimeSeriesData:
    """Load a registered dataset, a special dataset (see tsl/data/loaders.py), or any CSV via `path=`."""
    if path is not None:
        return load_csv(Path(path), name, meta={"freq": None}, channels=channels, max_rows=max_rows)
    cname = canonical(name)
    if cname not in DATASETS:
        from .loaders import SPECIAL, load_special
        if cname not in SPECIAL:
            raise KeyError(f"unknown dataset {name!r}; known: {list(DATASETS) + SPECIAL}")
        d = load_special(cname, **kwargs)
        if channels is not None:
            idx = [d.columns.index(c) if isinstance(c, str) else int(c) for c in channels]
            d.values, d.columns = d.values[:, idx], [d.columns[i] for i in idx]
            for k in ("avail", "regime"):
                if k in d.meta:
                    d.meta[k] = d.meta[k][:, idx]
        if max_rows:
            d.values, d.timestamps = d.values[:max_rows], d.timestamps[:max_rows]
            for k in ("avail", "regime"):
                if k in d.meta:
                    d.meta[k] = d.meta[k][:max_rows]
        return d
    meta = DATASETS[cname]
    return load_csv(dataset_path(cname), cname, meta=meta, channels=channels, max_rows=max_rows)


def synthetic_dataset(T: int = 2000, C: int = 3, seed: int = 0, drift_at: int | None = None) -> TimeSeriesData:
    """Deterministic sine+noise series (used by unit tests; no download needed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    vals = np.stack([np.sin(2 * np.pi * t / (24 + 5 * c)) + 0.1 * rng.standard_normal(T) + 0.01 * c * t / T
                     for c in range(C)], 1).astype(np.float32)
    if drift_at is not None:
        vals[drift_at:] += 1.5
    ts = pd.date_range("2020-01-01", periods=T, freq="h")
    return TimeSeriesData("synthetic", vals, ts, [f"c{c}" for c in range(C)], {"freq": "h", "steps_per_day": 24})
