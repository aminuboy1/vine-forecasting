"""Streaming evaluator for online forecasting under truthful delayed labels."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from ..utils import Timer, peak_rss_bytes
from .base import ForecastArchive, OnlineContext, OnlineMethod
from .protocol import HistoryView, LeakChecker, MaturedLabel, PrefixLabel, RevealedInfo, canonical_mode


@dataclass
class OnlineResult:
    method: str
    origins: np.ndarray            # (N,)
    se: np.ndarray                 # (N, H) squared error averaged over channels
    ae: np.ndarray                 # (N, H) absolute error averaged over channels
    timing: dict
    state_bytes: dict
    peak_rss_bytes: int | None
    protocol: dict
    extra: dict = field(default_factory=dict)
    preds: np.ndarray | None = None

    def summary(self) -> dict:
        from .metrics import summarize
        return summarize(self)


class OnlineEvaluator:
    """Runs one method over a stream with label delay `delay`.

    y : (T, C) settled labels (targets, scoring).   x : (T, C) real-time inputs (default: x = y).
    Forecast origins s = first_origin .. last_origin (inclusive); every origin with a complete target is
    scored (drop_last=False). Errors are computed by the evaluator from y; methods never see a target
    before its release time.
    """

    def __init__(self, y: np.ndarray, first_origin: int, last_origin: int | None, L: int, H: int,
                 delay: int | None = None, mode: str = "strict", allow_leaky: bool = False,
                 check_leaks: bool = True, keep_preds: bool = False, state_every: int = 500,
                 trace_memory: bool = False, x: np.ndarray | None = None, prefix_lag: int | None = None):
        y = np.asarray(y, dtype=np.float32)
        x = y if x is None else np.asarray(x, dtype=np.float32)
        if x.shape != y.shape:
            raise ValueError("inputs and labels must have the same shape")
        delay = H if delay is None else int(delay)
        mode = canonical_mode(mode)
        if mode == "leaky" and not allow_leaky:
            raise ValueError("mode='leaky' reproduces the FSNet/OneNet information leak and must be "
                             "enabled explicitly with allow_leaky=True (appendix comparisons only).")
        if mode != "leaky" and delay < H:
            raise ValueError(f"delay d={delay} < H={H} would release unobserved targets; use d >= H "
                             f"(or mode='leaky' with allow_leaky=True).")
        if first_origin < L - 1:
            raise ValueError("first_origin must leave a full lookback window")
        max_last = y.shape[0] - H - 1
        last_origin = max_last if last_origin is None else min(int(last_origin), max_last)
        self.y, self.x, self.L, self.H, self.d, self.mode = y, x, L, H, delay, mode
        self.lag = (delay - H) if prefix_lag is None else int(prefix_lag)
        self.first, self.last = int(first_origin), int(last_origin)
        self.check_leaks, self.keep_preds = check_leaks, keep_preds
        self.state_every, self.trace_memory = state_every, trace_memory

    def make_archive(self, backbone, device="cpu", **kw) -> ForecastArchive:
        return ForecastArchive(backbone, self.x, self.first, self.last, self.L, device, **kw)

    def _xw(self, s: int) -> np.ndarray:
        return self.x[s - self.L + 1: s + 1].copy()

    def _yb(self, s: int, k: int | None = None) -> np.ndarray:
        k = self.H if k is None else k
        return self.y[s + 1: s + 1 + k].copy()

    def run(self, method: OnlineMethod, backbone: torch.nn.Module, device="cpu", seed: int = 0,
            archive: ForecastArchive | None = None, pretrain_info: dict | None = None,
            verbose: bool = False, base_preds: np.ndarray | None = None) -> OnlineResult:
        L, H, d = self.L, self.H, self.d
        origins = np.arange(self.first, self.last + 1)
        N, C = len(origins), self.y.shape[1]
        checker = LeakChecker(H, d, self.mode, enabled=self.check_leaks and self.mode != "leaky",
                              prefix_lag=self.lag)
        hist = HistoryView(self.y, self.x, checker)
        hist.now, hist.phase = self.first - 1, "init"
        backbone = backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        if archive is None:
            archive = self.make_archive(backbone, device)
        archive.now = self.first - 1

        def bp_of(s):
            if base_preds is not None:
                return base_preds[s - self.first]
            return archive.get(s, guard=False)

        ctx = OnlineContext(backbone=backbone, L=L, H=H, C=C, delay=d, mode=self.mode, first_origin=self.first,
                            history=hist, device=torch.device(device), seed=seed, pretrain_info=pretrain_info,
                            prefix_lag=self.lag, archive=archive, extra={"n_forecasts": int(N)})
        torch.manual_seed(seed)
        np.random.seed(seed)
        timer = Timer()
        with timer.section("setup"):
            method.setup(ctx)

        se = np.empty((N, H), np.float32)
        ae = np.empty((N, H), np.float32)
        preds = np.empty((N, H, C), np.float32) if self.keep_preds else None
        own_pred: dict[int, np.ndarray] = {}     # forecasts awaiting their labels (evaluator bookkeeping)
        state_trace = []
        if self.trace_memory:
            import tracemalloc
            tracemalloc.start()
        t_start = time.perf_counter()

        for i, s in enumerate(origins):
            s = int(s)
            archive.now = s
            # ---------------- 1) release labels at time s, then update
            hist.now, hist.phase = s, "update"
            if self.mode != "leaky":
                matured = []
                o = s - d
                if o >= self.first:
                    matured.append(MaturedLabel(o, self._xw(o), self._yb(o), bp_of(o), own_pred.pop(o, None), s))
                info = RevealedInfo(now=s, matured=matured)
                if self.mode == "prefix" and method.uses_prefix:
                    info._prefix_fn = self._make_prefix_fn(s, bp_of, own_pred)
                checker.check(info)
                if matured or info._prefix_fn is not None:
                    with timer.section("update"):
                        method.update(info)

            # ---------------- 2) forecast at s
            hist.phase = "predict"
            xw = self._xw(s)
            bp = bp_of(s) if method.needs_base_pred else None
            with timer.section("predict"):
                yhat = np.asarray(method.predict(xw, bp, s), dtype=np.float32)
            if yhat.shape != (H, C):
                raise ValueError(f"{method.name}.predict returned {yhat.shape}, expected {(H, C)}")
            yt = self._yb(s)
            err = yhat - yt
            se[i] = (err ** 2).mean(1)
            ae[i] = np.abs(err).mean(1)
            if preds is not None:
                preds[i] = yhat
            if self.mode != "leaky":
                own_pred[s] = yhat
            else:
                # FSNet/OneNet: the full label of the forecast just made is used immediately.
                hist.phase = "update"
                leak = RevealedInfo(now=s, matured=[MaturedLabel(s, xw, yt, bp, yhat, s)])
                checker.check(leak)   # does not raise in leaky mode, records the violation size
                with timer.section("update"):
                    method.update(leak)

            if self.state_every and (i % self.state_every == 0 or i == N - 1):
                state_trace.append((s, method.state_bytes()))
            if verbose and i % 5000 == 0:
                print(f"    [{method.name}] step {i}/{N} ({time.perf_counter() - t_start:.1f}s)", flush=True)

        wall = time.perf_counter() - t_start
        traced_peak = None
        if self.trace_memory:
            import tracemalloc
            traced_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        if self.mode == "leaky":
            warnings.warn("LEAKY protocol used: results are NOT valid for headline comparisons.")
        sb = [b for _, b in state_trace]
        protocol = {"mode": self.mode, "H": H, "delay": d, "prefix_lag": self.lag, "L": L,
                    "first_origin": self.first, "last_origin": self.last, "n_forecasts": int(N),
                    "leaky": self.mode == "leaky", "leak_checks": checker.n_checks,
                    "max_leak_steps": checker.max_violation, "history_max_label_accessed": hist.max_label_accessed,
                    "two_channel": self.x is not self.y}
        return OnlineResult(method=method.name, origins=origins, se=se, ae=ae, timing=timer.summary(),
                            state_bytes={"final": int(sb[-1]) if sb else 0, "max": int(max(sb)) if sb else 0,
                                         "trace": state_trace, "categories": method.state_categories(),
                                         "breakdown": _breakdown_bytes(method)},
                            peak_rss_bytes=peak_rss_bytes(), protocol=protocol,
                            extra={**method.extra_metrics(), "wall_s": wall,
                                   "traced_peak_mb": None if traced_peak is None else traced_peak / 2 ** 20},
                            preds=preds)

    def _make_prefix_fn(self, now, bp_of, own_pred):
        def fn():
            out = []
            for o in range(max(self.first, now - self.d + 1), now):
                k = min(self.H, now - self.lag - o)
                if k >= 1:
                    out.append(PrefixLabel(o, self._xw(o), self._yb(o, k), bp_of(o), own_pred.get(o)))
            return out
        return fn


def _breakdown_bytes(method) -> dict:
    from ..utils import nbytes_of
    return {k: int(nbytes_of(v)) for k, v in method.state_breakdown().items()}
