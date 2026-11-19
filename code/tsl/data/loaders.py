"""Loaders for the non-TSLib datasets (OPSD, an optional extra stream; not used in the paper's main grid).

Every loader returns a `TimeSeriesData` whose `meta` may contain
  * "avail": (T, C) bool availability mask (variable births/deaths, missing values) -- open-variable streams
  * "periods": list of (start, end) row ranges given by the source (e.g. -Stream years)
  * "adj": list of adjacency matrices per period (-Stream sets), optional
Missing values are stored as 0 in `values` and flagged False in `avail`.

Datasets
  OPSD-DE / OPSD-IT ........ Open Power System Data time_series 2020-10-06 (hourly load, MW)  [paper 2]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .registry import TimeSeriesData, data_root

# ------------------------------------------------------------------------------------------- OPSD
OPSD_FILE = "opsd/time_series_60min_singleindex.csv"
OPSD_COLUMNS = {
    "DE": ["DE_load_actual_entsoe_transparency", "DE_50hertz_load_actual_entsoe_transparency",
           "DE_amprion_load_actual_entsoe_transparency", "DE_tennet_load_actual_entsoe_transparency",
           "DE_transnetbw_load_actual_entsoe_transparency"],
    "IT": ["IT_load_actual_entsoe_transparency"],
}
OPSD_FORECAST_COLUMNS = {"DE": "DE_load_forecast_entsoe_transparency", "IT": "IT_load_forecast_entsoe_transparency"}


def load_opsd(country: str = "DE", with_tso_forecast: bool = False, start: str = "2015-01-01",
              end: str | None = None) -> TimeSeriesData:
    path = data_root() / OPSD_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python scripts/download_opsd.py`")
    cols = list(OPSD_COLUMNS[country])
    if with_tso_forecast:
        cols.append(OPSD_FORECAST_COLUMNS[country])
    df = pd.read_csv(path, usecols=["utc_timestamp"] + cols, parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp")
    df = df.loc[pd.Timestamp(start, tz="UTC"):] if start else df
    if end:
        df = df.loc[:pd.Timestamp(end, tz="UTC")]
    first_valid = df[cols[0]].first_valid_index()
    last_valid = df[cols[0]].last_valid_index()
    df = df.loc[first_valid:last_valid]
    avail = df.notna().to_numpy()
    vals = df.ffill().bfill().to_numpy(np.float32)   # isolated gaps (<0.1%) filled; flagged in avail
    return TimeSeriesData(f"OPSD-{country}", vals, pd.DatetimeIndex(df.index.tz_convert(None)), cols,
                          {"freq": "h", "steps_per_day": 24, "avail": avail,
                           "source": "OPSD time_series v2020-10-06 (ENTSO-E Transparency); licence: verify"})


def load_special(name: str, **kw) -> TimeSeriesData:
    if name in ("OPSD-DE", "OPSD-IT"):
        return load_opsd(name.split("-")[1], **kw)
    raise KeyError(name)


SPECIAL = ["OPSD-DE", "OPSD-IT"]
_ = Path  # keep import for type hints in user code
