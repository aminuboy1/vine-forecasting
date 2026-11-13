"""Stacked correction: a top corrector learns on the forecasts issued by a base corrector.

`Stack(base, top)` ("stack"): at origin t the base corrector maps the frozen forecast B_t to F_t; the top corrector
then treats F_t as its "backbone" forecast. All of VINE's machinery (revisions, telescoping, the point-lag anchor,
the fixed point) is defined for ANY forecaster that re-issues forecasts every step, so VINE on top of a label-only
corrector learns from the revisions of the corrected forecasts: rho_{t,h} = F_{t,h-1} - F_{t-1,h}.

Causality: the base is a causal online method; the top sees only F_s for s <= t and labels through the guarded
history. When a label block matures, the top receives it with `base_pred` replaced by the base's issued forecast F_s
(so gates and block-matured statistics measure the residual of what the top actually corrected).
Prefix labels (P-prefix mode) are not forwarded: `uses_prefix` is False, and neither DRiC nor VINE consumes them.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..base import OnlineContext, OnlineMethod
from ..protocol import RevealedInfo


class Stack(OnlineMethod):
    name = "stack"

    def __init__(self, base: dict | None = None, top: dict | None = None, **kw):
        super().__init__(**kw)
        self.base_spec = dict(base or {"type": "dric", "variant": "l8", "phases": "all"})
        self.top_spec = dict(top or {"type": "vine"})

    def setup(self, ctx: OnlineContext) -> None:
        super().setup(ctx)
        from . import build_method
        self.base = build_method(dict(self.base_spec))
        self.top = build_method(dict(self.top_spec))
        self.base.setup(ctx)
        self.top.setup(ctx)
        self.pending_base: dict[int, np.ndarray] = {}

    def predict(self, x: np.ndarray, base_pred: np.ndarray, t: int) -> np.ndarray:
        f = np.asarray(self.base.predict(x, base_pred, t), np.float32)
        self.pending_base[t] = f
        return self.top.predict(x, f, t)

    def update(self, info: RevealedInfo) -> None:
        self.base.update(info)
        matured = []
        for m in info.matured:
            f = self.pending_base.pop(m.origin, None)
            if f is None:
                raise RuntimeError(f"stack: no base forecast stored for matured origin {m.origin}")
            matured.append(replace(m, base_pred=f, pred=None))
        self.top.update(RevealedInfo(now=info.now, matured=matured, _prefix_fn=info._prefix_fn))

    def state_breakdown(self) -> dict:
        out = {"pending_base_forecasts": list(self.pending_base.values())}
        for tag, m in (("base", self.base), ("top", self.top)):
            for k, v in m.state_breakdown().items():
                out[f"{k}_{tag}" if k.startswith(("pending", "raw")) else f"{tag}_{k}"] = v
        return out

    def extra_metrics(self) -> dict:
        return {"base": self.base.extra_metrics(), "top": self.top.extra_metrics()}
