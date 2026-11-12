"""Plug-in interface for online forecasting methods (correctors / adapters).

A new method subclasses `OnlineMethod` and implements `predict` and `update`:

    class MyCorrector(OnlineMethod):
        name = "my_corrector"
        def setup(self, ctx): ...                       # called once before the stream starts
        def predict(self, x, base_pred, t): ...         # -> (H, C) forecast for y[t+1..t+H]
        def update(self, info: RevealedInfo): ...       # labels released at time info.now
        def state_breakdown(self): return {"W": self.W} # arrays retained between steps (state bytes)

What a method may use (everything else raises LeakageError or is simply not reachable):
  * predict(x, base_pred, t): the input window x = x[t-L+1..t] and the frozen backbone forecast for it;
  * update(info): labels released at `info.now` (info.matured; info.prefixes in P-prefix mode);
  * ctx.history.inputs(a, b) / ctx.history.labels(a, b): guarded access to the two channels;
  * ctx.base_forecast(s): the frozen backbone's forecast issued at any earlier origin s <= now
    (label-free; e.g. for forecast REVISIONS  yhat_{s+1,h-1} - yhat_{s,h}).
The evaluator calls update(info_t) BEFORE predict(x_t), so labels released at t are usable at t.

`state_breakdown` keys are grouped into categories by prefix for reporting:
  "pending_*" -> pending queue, "raw_*" -> raw data retained, everything else -> learned state.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..utils import nbytes_of
from .protocol import HistoryView, LeakageError, RevealedInfo


@contextmanager
def torch_threads(n: int | None):
    if n is None:
        yield
        return
    old = torch.get_num_threads()
    torch.set_num_threads(int(n))
    try:
        yield
    finally:
        torch.set_num_threads(old)


class ForecastArchive:
    """Frozen-backbone forecasts for every origin, computed lazily in chunks from the INPUT channel.

    Narrow streams are cached entirely; wide streams (ECL/Traffic at large H would need GBs) keep only an
    LRU of chunks and recompute evicted chunks on demand (deterministic, so results are identical).
    `get(s)` is guarded: methods may only ask for origins s <= now (forecasts issued so far).
    """

    def __init__(self, model: torch.nn.Module, x: np.ndarray, first: int, last: int, L: int, device="cpu",
                 chunk: int = 1024, max_cache_bytes: float = 1.5e9, dtype=np.float32, threads: int | None = None):
        self.model, self.x, self.first, self.last, self.L = model.eval(), x, first, last, L
        self.device, self.chunk, self.dtype, self.threads = device, chunk, dtype, threads
        self.now = last            # evaluator moves this; default unguarded for offline use
        with torch.no_grad():
            probe = model(torch.from_numpy(np.ascontiguousarray(x[first - L + 1:first + 1][None], np.float32))
                          .to(device))
        self.H, self.C = probe.shape[1], probe.shape[2]
        n = last - first + 1
        bytes_all = n * self.H * self.C * np.dtype(dtype).itemsize
        self.max_chunks = max(4, int(max_cache_bytes // max(chunk * self.H * self.C * np.dtype(dtype).itemsize, 1)))
        self.full = bytes_all <= max_cache_bytes
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.n_computed_chunks = 0

    def _chunk(self, k: int) -> np.ndarray:
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        o0 = self.first + k * self.chunk
        o1 = min(self.last, o0 + self.chunk - 1)
        origins = np.arange(o0, o1 + 1)
        idx = origins[:, None] + np.arange(-self.L + 1, 1)[None, :]
        out = []
        with torch.no_grad(), torch_threads(self.threads):
            for i in range(0, len(origins), 256):
                xb = torch.from_numpy(np.ascontiguousarray(self.x[idx[i:i + 256]], dtype=np.float32)).to(self.device)
                out.append(self.model(xb).float().cpu().numpy().astype(self.dtype))
        arr = np.concatenate(out, 0)
        self.cache[k] = arr
        self.n_computed_chunks += 1
        if not self.full:
            while len(self.cache) > self.max_chunks:
                self.cache.popitem(last=False)
        return arr

    def get(self, s: int, guard: bool = True) -> np.ndarray:
        s = int(s)
        if guard and s > self.now:
            raise LeakageError(f"backbone forecast for origin {s} requested at time {self.now}")
        if s < self.first or s > self.last:
            raise IndexError(f"origin {s} outside [{self.first}, {self.last}]")
        k = (s - self.first) // self.chunk
        return self._chunk(k)[(s - self.first) % self.chunk].astype(np.float32)

    def all(self) -> np.ndarray:
        """Materialise every forecast (offline analysis only; can be large)."""
        nk = (self.last - self.first) // self.chunk + 1
        return np.concatenate([self._chunk(k) for k in range(nk)], 0).astype(np.float32)


@dataclass
class OnlineContext:
    backbone: torch.nn.Module      # frozen, eval-mode backbone (do not modify in place; deepcopy if training)
    L: int
    H: int
    C: int
    delay: int
    mode: str                      # "strict" | "prefix" | "leaky"
    first_origin: int              # first forecast origin of the online segment
    history: HistoryView           # guarded access to inputs / released labels
    device: torch.device
    seed: int = 0
    pretrain_info: dict | None = None
    prefix_lag: int = 0
    archive: ForecastArchive | None = None
    extra: dict = field(default_factory=dict)

    def base_forecast(self, s: int) -> np.ndarray:
        """Frozen backbone forecast issued at origin s (s <= now). Shape (H, C)."""
        if self.archive is None:
            raise RuntimeError("no forecast archive attached")
        return self.archive.get(s)

    def label_limit(self) -> int:
        return self.history.label_limit()


class OnlineMethod:
    name: str = "base"
    uses_prefix: bool = False      # set True to receive prefix labels in P-prefix mode
    needs_base_pred: bool = True   # if False the evaluator passes base_pred=None to predict (saves time)

    def __init__(self, **cfg):
        self.cfg = cfg

    def setup(self, ctx: OnlineContext) -> None:
        self.ctx = ctx

    def predict(self, x: np.ndarray, base_pred: np.ndarray | None, t: int) -> np.ndarray:
        raise NotImplementedError

    def update(self, info: RevealedInfo) -> None:
        pass

    def state_breakdown(self) -> dict:
        """Arrays/tensors retained across steps, by name (see module docstring for categories)."""
        return {}

    def state_bytes(self) -> int:
        return int(sum(nbytes_of(v) for v in self.state_breakdown().values()))

    def state_categories(self) -> dict:
        cats = {"learned": 0, "pending": 0, "raw": 0}
        for k, v in self.state_breakdown().items():
            c = "pending" if k.startswith("pending") else "raw" if k.startswith("raw") else "learned"
            cats[c] += nbytes_of(v)
        return cats

    def extra_metrics(self) -> dict:
        return {}


def batched_forecast(model: torch.nn.Module, z: np.ndarray, origins: np.ndarray, L: int, device,
                     batch_size: int = 512) -> np.ndarray:
    """Frozen-model forecasts for many origins at once; only input windows z[s-L+1..s] are read."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(origins), batch_size):
            o = origins[i:i + batch_size]
            idx = o[:, None] + np.arange(-L + 1, 1)[None, :]
            xb = torch.from_numpy(np.ascontiguousarray(z[idx], dtype=np.float32)).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0,), np.float32)


MethodFactory = Callable[[], OnlineMethod]
