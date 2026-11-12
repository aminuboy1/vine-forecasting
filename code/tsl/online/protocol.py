"""Truthful delayed-label protocol for online forecasting (two-channel stream).

Two channels (numerically identical in public benchmarks, different in the provisional-data stress test):
  * inputs  x_t : observed in real time; usable as FEATURES at any time >= t;
  * labels  y_t : settled values; usable as TRAINING TARGETS only once released.

Time convention (row indices):
  * At time s the input x[s] has just been observed.
  * The forecaster issues  Yhat_s = f(x[s-L+1 .. s])  for targets  Y_s = y[s+1 .. s+H].
  * P-strict (mode "strict", alias "matured"): the whole label block Y_s is released at s + d (d >= H).
    A value y_tau first appears in a released block at tau - H + d, so the label limit at time t is
        label_limit(t) = t - d + H            (== t when d == H).
  * P-prefix (mode "prefix"): additionally, y_tau may be used as a target once tau <= t - l, where
    l = prefix_lag (default d - H, the audit's "prefix reduced accordingly" rule). For every pending origin
    s' (t - d < s' < t) the evaluator exposes the settled prefix y[s'+1 .. min(s'+H, t-l)].
  * P-leaky (mode "leaky", appendix only): FSNet/OneNet style. The full label Y_s is handed to the learner
    right after forecasting at s, so updates touch y[s+2 .. s+H] which are not yet observed. Requires
    allow_leaky=True; results are tagged LEAKY.

`LeakChecker` raises `LeakageError` whenever an update would touch a label beyond the limit, both for the
labels handed over by the evaluator and for any access through `HistoryView` / `ForecastArchive`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MODES = ("strict", "prefix", "leaky")
ALIASES = {"matured": "strict", "p-strict": "strict", "p-prefix": "prefix", "p-leaky": "leaky"}


def canonical_mode(mode: str) -> str:
    m = ALIASES.get(mode.lower(), mode.lower())
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES} (or aliases {list(ALIASES)}), got {mode!r}")
    return m


class LeakageError(AssertionError):
    """Raised when an online update would use not-yet-revealed values."""


@dataclass
class MaturedLabel:
    origin: int                    # forecast origin s'
    x: np.ndarray                  # (L, C) input window used at s'
    y: np.ndarray                  # (H, C) full settled target y[s'+1 .. s'+H]
    base_pred: np.ndarray | None   # frozen backbone forecast issued at s' (H, C)
    pred: np.ndarray | None        # this method's own forecast issued at s' (H, C)
    release_time: int

    @property
    def max_target_time(self) -> int:
        return self.origin + self.y.shape[0]


@dataclass
class PrefixLabel:
    origin: int
    x: np.ndarray
    y_prefix: np.ndarray           # (k, C) = y[s'+1 .. s'+k], k <= H
    base_pred: np.ndarray | None
    pred: np.ndarray | None

    @property
    def k(self) -> int:
        return self.y_prefix.shape[0]

    @property
    def max_target_time(self) -> int:
        return self.origin + self.k


@dataclass
class RevealedInfo:
    now: int
    matured: list[MaturedLabel] = field(default_factory=list)
    _prefix_fn: Callable[[], list[PrefixLabel]] | None = None
    _prefix_cache: list[PrefixLabel] | None = None

    @property
    def prefixes(self) -> list[PrefixLabel]:
        """Lazily built (only methods with `uses_prefix=True` pay the cost)."""
        if self._prefix_cache is None:
            self._prefix_cache = self._prefix_fn() if self._prefix_fn is not None else []
        return self._prefix_cache


class LeakChecker:
    def __init__(self, H: int, delay: int, mode: str = "strict", enabled: bool = True,
                 prefix_lag: int | None = None):
        self.mode = canonical_mode(mode)
        self.H, self.d, self.enabled = H, delay, enabled
        self.lag = (delay - H) if prefix_lag is None else int(prefix_lag)
        if self.lag < 0:
            raise ValueError("prefix_lag must be >= 0")
        self.n_checks = 0
        self.max_violation = 0  # tracked (not raised) in leaky mode

    def honest_label_limit(self, now: int) -> int:
        if self.mode == "prefix":
            return now - self.lag
        return now - self.d + self.H

    def label_limit(self, now: int) -> int:
        """Largest row whose LABEL an update may use as a target at time `now`."""
        if self.mode == "leaky":
            return now + self.H
        return self.honest_label_limit(now)

    # backwards-compatible name
    visible_limit = label_limit

    def assert_visible(self, t_max: int, now: int, what: str = "access") -> None:
        self.n_checks += 1
        hl = self.honest_label_limit(now)   # for leaky mode: the truthful limit it violates
        if t_max > hl:
            self.max_violation = max(self.max_violation, t_max - hl)
        if self.enabled and t_max > self.label_limit(now):
            raise LeakageError(f"{what}: uses label y[{t_max}] but only y[..{self.label_limit(now)}] is released "
                               f"at time {now} (mode={self.mode}, H={self.H}, d={self.d}, lag={self.lag})")

    def check(self, info: RevealedInfo) -> None:
        now = info.now
        for m in info.matured:
            if self.enabled and m.origin + self.d > now:
                raise LeakageError(f"label of origin {m.origin} released at {now} < origin+d={m.origin + self.d}")
            if self.enabled and m.release_time > now:
                raise LeakageError(f"label released in the future ({m.release_time} > {now})")
            self.assert_visible(m.max_target_time, now, f"matured label (origin {m.origin})")
        if info._prefix_cache is not None or info._prefix_fn is not None:
            if self.mode == "strict" and self.enabled:
                raise LeakageError("prefix labels are not allowed under P-strict")
            for p in info.prefixes:
                self.assert_visible(p.max_target_time, now, f"prefix label (origin {p.origin})")


class HistoryView:
    """Guarded read access to the two channels for methods (e.g. retraining windows, warm starts).

    * `inputs(start, stop)`  : x rows, allowed up to the current time `now`;
    * `labels(start, stop)`  : y rows usable as targets, allowed up to `label_limit(now)`;
    * `get` is an alias of `labels` (targets are what leak).
    Returned arrays are copies, so the underlying arrays cannot be reached through `.base`.
    """

    def __init__(self, y: np.ndarray, x: np.ndarray, checker: LeakChecker):
        self._y, self._x, self._checker = y, x, checker
        self.now: int = -1
        self.phase: str = "init"   # "init" | "update" | "predict"
        self.max_label_accessed = -1

    @property
    def C(self) -> int:
        return self._y.shape[1]

    def label_limit(self) -> int:
        return self._checker.label_limit(self.now)

    def observed_until(self) -> int:
        """Last row whose label may be used as a target right now."""
        return self.label_limit()

    def inputs(self, start: int, stop: int) -> np.ndarray:
        start, stop = max(0, int(start)), int(stop)
        if stop - 1 > self.now:
            raise LeakageError(f"input access x[{stop - 1}] at time {self.now} (future input)")
        return np.array(self._x[start:stop], dtype=np.float32, copy=True)

    def labels(self, start: int, stop: int) -> np.ndarray:
        start, stop = max(0, int(start)), int(stop)
        if self.phase == "predict" and stop - 1 > self.label_limit():
            raise LeakageError(f"predict-time label access y[{stop - 1}] at time {self.now}")
        if stop - 1 > self.label_limit() or self._checker.mode == "leaky":
            self._checker.assert_visible(stop - 1, self.now, "HistoryView.labels")
        self.max_label_accessed = max(self.max_label_accessed, stop - 1)
        return np.array(self._y[start:stop], dtype=np.float32, copy=True)

    get = labels
