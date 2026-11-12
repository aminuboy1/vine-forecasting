"""Loaders for the non-TSLib datasets used by the two papers.

Every loader returns a `TimeSeriesData` whose `meta` may contain
  * "avail": (T, C) bool availability mask (variable births/deaths, missing values) -- open-variable streams
  * "periods": list of (start, end) row ranges given by the source (e.g. -Stream years)
  * "adj": list of adjacency matrices per period (-Stream sets), optional
Missing values are stored as 0 in `values` and flagged False in `avail`.

Datasets
  OPSD-DE / OPSD-IT ........ Open Power System Data time_series 2020-10-06 (hourly load, MW)  [paper 2]
  ECL-Churn ................ UCI ElectricityLoadDiagrams2011-2014, raw 370 clients, hourly; births detected
                             as the first non-zero value, deaths as trailing all-zero runs        [paper 1]
  Energy-Stream / PEMS-Stream / Air-Stream / CA-Stream .. EAC / STBP npz format (one file per period,
                             append-only node expansion, key "x" of shape (T, N_period))       [paper 1]
  CESNET-<level> ........... CESNET-TimeSeries24 (institutions | institution_subnets), 10 min / 1 h / 1 day,
                             one metric (default n_bytes); missing id_time rows -> unavailable     [paper 1]
  SynthRegimes ............. generator in tsl/data/synth.py                                       [paper 1]
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


# ------------------------------------------------------------------------------------------- UCI ECL-Churn
def load_ecl_churn(freq: str = "1h", cache: bool = True) -> TimeSeriesData:
    """Raw UCI LD2011_2014 (370 clients, 15-min kW). Aggregated by mean to `freq`.

    Births: rows before a client's first non-zero value are unavailable (UCI: "Some clients were created
    after 2011. In these cases consumption were considered zero"). Deaths: a trailing run of zeros longer
    than one day is treated as retirement. Both are stored in meta["avail"]; audit with
    `churn_summary(data)` before using (methodology D3 audit).
    """
    src = data_root() / "uci_ecl" / "LD2011_2014.txt"
    cpath = data_root() / "uci_ecl" / f"cache_{freq}.npz"
    if cache and cpath.exists():
        z = np.load(cpath, allow_pickle=False)
        ts = pd.DatetimeIndex(z["ts"].astype("datetime64[ns]"))
        return TimeSeriesData("ECL-Churn", z["values"], ts, list(z["columns"]),
                              {"freq": freq, "steps_per_day": 24 if freq == "1h" else 96, "avail": z["avail"]})
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run `python scripts/download_uci_ecl.py`")
    df = pd.read_csv(src, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.astype(np.float32)
    if freq != "15min":
        # timestamps mark the END of each 15-min interval; label hourly bins by their end as well
        df = df.resample(freq, label="right", closed="right").mean()
    vals = df.to_numpy(np.float32)
    T, C = vals.shape
    nz = vals != 0
    first = np.where(nz.any(0), nz.argmax(0), T)
    last = np.where(nz.any(0), T - 1 - nz[::-1].argmax(0), -1)
    spd = 24 if freq == "1h" else 96
    avail = np.zeros((T, C), bool)
    for c in range(C):
        end = T if (T - 1 - last[c]) <= spd else last[c] + 1
        avail[first[c]:end, c] = True
    ts = pd.DatetimeIndex(df.index)
    if cache:
        np.savez(cpath, values=vals, avail=avail, ts=ts.values.astype("datetime64[ns]"),
                 columns=np.array(df.columns, dtype=str))
    return TimeSeriesData("ECL-Churn", vals, ts, list(df.columns), {"freq": freq, "steps_per_day": spd,
                                                                    "avail": avail})


def churn_summary(data: TimeSeriesData) -> dict:
    a = data.meta["avail"]
    born = a.argmax(0)
    died = np.array([np.flatnonzero(a[:, c])[-1] + 1 if a[:, c].any() else 0 for c in range(a.shape[1])])
    return {"n_vars": int(a.shape[1]), "born_after_start": int((born > 0).sum()),
            "retired_before_end": int((died < a.shape[0]).sum()), "never_active": int((~a.any(0)).sum()),
            "active_fraction": float(a.mean())}


# ------------------------------------------------------------------------------------------- -Stream (EAC / STBP)
STREAMS = {
    "Energy-Stream": {"dir": "eac/ENERGY-Wind", "periods": ["0", "1", "2", "3"], "freq": "10min", "spd": 144},
    "PEMS-Stream": {"dir": "eac/PEMS", "periods": [str(y) for y in range(2011, 2018)], "freq": "5min", "spd": 288},
    "Air-Stream": {"dir": "eac/AIR", "periods": [str(y) for y in range(2016, 2020)], "freq": "1h", "spd": 24},
    "CA-Stream": {"dir": "stbp_ca", "periods": ["0", "1", "2", "3"], "freq": "5min", "spd": 288},
}


def load_stream(name: str, periods: list[str] | None = None, with_adj: bool = False,
                max_rows_per_period: int | None = None) -> TimeSeriesData:
    """Concatenate the per-period npz files into one open-variable stream.

    Node expansion is append-only (period p+1's first N_p nodes are period p's nodes), so node j of any
    period is column j of the stream; a node is available from the first period that contains it.
    Sampling intervals are inferred from row counts (not stored in the files); timestamps are synthetic.
    """
    spec = STREAMS[name]
    root = data_root() / spec["dir"]
    periods = periods or spec["periods"]
    blocks, Ns = [], []
    for p in periods:
        f = root / "RawData" / f"{p}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; run `python scripts/download_streams.py --only "
                                    f"{name.split('-')[0].lower()}`")
        x = np.load(f, allow_pickle=False)["x"].astype(np.float32)
        if max_rows_per_period:
            x = x[:max_rows_per_period]
        blocks.append(x)
        Ns.append(x.shape[1])
    C = max(Ns)
    T = sum(b.shape[0] for b in blocks)
    vals = np.zeros((T, C), np.float32)
    avail = np.zeros((T, C), bool)
    bounds, r = [], 0
    for b in blocks:
        vals[r:r + len(b), :b.shape[1]] = b
        avail[r:r + len(b), :b.shape[1]] = True
        bounds.append((r, r + len(b)))
        r += len(b)
    meta = {"freq": spec["freq"], "steps_per_day": spec["spd"], "avail": avail, "periods": bounds,
            "period_names": periods, "nodes_per_period": Ns}
    if with_adj:
        meta["adj"] = [np.load(root / "graph" / f"{p}_adj.npz", allow_pickle=False)["x"] for p in periods]
    ts = pd.date_range("2000-01-01", periods=T, freq=spec["freq"])  # synthetic, contiguous per period
    return TimeSeriesData(name, vals, ts, [f"node{j}" for j in range(C)], meta)


# ------------------------------------------------------------------------------------------- CESNET
def load_cesnet(level: str = "institutions", agg: str = "1_hour", metric: str = "n_bytes",
                max_series: int | None = None, cache: bool = True) -> TimeSeriesData:
    root = data_root() / "cesnet_ts24" / "extracted"
    d = root / level / f"agg_{agg}"
    if not d.exists():
        raise FileNotFoundError(f"{d} missing; run `python scripts/download_cesnet.py --extract`")
    cpath = root.parent / f"cache_{level}_{agg}_{metric}.npz"
    times = pd.read_csv(root / "times" / f"times_{agg}.csv")
    ts = pd.DatetimeIndex(pd.to_datetime(times["time"], utc=True).dt.tz_convert(None))
    if cache and cpath.exists():
        z = np.load(cpath, allow_pickle=False)
        vals, avail, cols = z["values"], z["avail"], list(z["columns"])
    else:
        files = sorted(d.glob("*.csv"), key=lambda p: int(p.stem))
        T = len(times)
        vals = np.zeros((T, len(files)), np.float32)
        avail = np.zeros((T, len(files)), bool)
        for j, f in enumerate(files):
            df = pd.read_csv(f, usecols=["id_time", metric])
            idx = df["id_time"].to_numpy()
            v = df[metric].to_numpy(np.float64)
            ok = np.isfinite(v)
            vals[idx[ok], j] = v[ok]
            avail[idx[ok], j] = True
        cols = [f.stem for f in files]
        if cache:
            np.savez(cpath, values=vals, avail=avail, columns=np.array(cols, dtype=str))
    if max_series:
        vals, avail, cols = vals[:, :max_series], avail[:, :max_series], cols[:max_series]
    spd = {"10_minutes": 144, "1_hour": 24, "1_day": 1}[agg]
    hol = root / "weekends_and_holidays.csv"
    meta = {"freq": agg, "steps_per_day": spd, "avail": avail, "metric": metric,
            "holidays": str(hol) if hol.exists() else None}
    return TimeSeriesData(f"CESNET-{level}", vals.astype(np.float32), ts, cols, meta)


def load_special(name: str, **kw) -> TimeSeriesData:
    if name in ("OPSD-DE", "OPSD-IT"):
        return load_opsd(name.split("-")[1], **kw)
    if name == "ECL-Churn":
        return load_ecl_churn(**kw)
    if name in STREAMS:
        return load_stream(name, **kw)
    if name.startswith("CESNET"):
        level = name.split("-", 1)[1] if "-" in name else "institutions"
        return load_cesnet(level=level, **kw)
    if name == "SynthRegimes":
        from .synth import synth_regimes
        return synth_regimes(**kw)
    raise KeyError(name)


SPECIAL = ["OPSD-DE", "OPSD-IT", "ECL-Churn", *STREAMS, "CESNET-institutions", "CESNET-institution_subnets",
           "SynthRegimes"]
_ = Path  # keep import for type hints in user code
