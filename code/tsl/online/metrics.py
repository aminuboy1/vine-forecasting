"""Online metrics: cumulative / per-horizon / per-step errors, latency, state and memory."""
from __future__ import annotations

import numpy as np


def cumulative_curve(per_step: np.ndarray, n_points: int = 50) -> dict:
    """Running mean of a per-step loss sampled at ~n_points positions (for plots)."""
    c = np.cumsum(per_step, dtype=np.float64) / np.arange(1, len(per_step) + 1)
    idx = np.unique(np.linspace(0, len(per_step) - 1, min(n_points, len(per_step))).astype(int))
    return {"step": idx.tolist(), "value": c[idx].tolist()}


def summarize(res) -> dict:
    se, ae = res.se.astype(np.float64), res.ae.astype(np.float64)
    t = res.timing
    upd = t.get("update", {})
    prd = t.get("predict", {})
    n = len(res.origins)
    return {
        "method": res.method,
        "mse": float(se.mean()),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(se.mean())),
        "mse_per_horizon": se.mean(0).tolist(),
        "mae_per_horizon": ae.mean(0).tolist(),
        "cum_mse_curve": cumulative_curve(se.mean(1)),
        "n_forecasts": int(n),
        "update_latency_us_mean": upd.get("mean_us", 0.0),
        "update_latency_us_median": upd.get("median_us", 0.0),
        "update_latency_us_p95": upd.get("p95_us", 0.0),
        "predict_latency_us_mean": prd.get("mean_us", 0.0),
        "total_update_s": upd.get("total_s", 0.0),
        "total_predict_s": prd.get("total_s", 0.0),
        "setup_s": t.get("setup", {}).get("total_s", 0.0),
        "state_bytes_final": res.state_bytes["final"],
        "state_bytes_max": res.state_bytes["max"],
        "state_breakdown": res.state_bytes.get("breakdown", {}),
        "peak_rss_mb": None if res.peak_rss_bytes is None else res.peak_rss_bytes / 2 ** 20,
        "protocol": res.protocol,
        **{f"extra_{k}": v for k, v in res.extra.items() if np.isscalar(v)},
    }
