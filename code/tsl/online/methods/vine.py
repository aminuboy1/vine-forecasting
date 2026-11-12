"""STUB for VINE (paper 2: vintage-consistent revision learning). Not implemented here on purpose.

The harness already provides everything VINE needs (see methodology/paper2_online_tsf_methodology.md):

  * P-strict / P-prefix two-channel stream ........ OnlineEvaluator(mode="strict"|"prefix", x=inputs, y=labels,
                                                     delay=d >= H, prefix_lag=l (default d-H))
  * the backbone's own past forecasts ............. ctx.base_forecast(s) for any origin s <= now (guarded)
  * label-free revisions .......................... revision(ctx, t) below: yhat_{t,1:H-1} - yhat_{t-1,2:H}
  * anchor labels released at s+d ................. info.matured[*] (origin s, y block, base_pred, own pred)
  * prefix anchor (P-prefix) ...................... info.prefixes[*] with y[s'+1 .. now-l] (uses_prefix=True)
  * inputs as features (never as targets) ......... the x window passed to predict / ctx.history.inputs()
  * leak test ..................................... tsl.online.leaktest.future_perturbation_test
  * state categories (learned / pending / raw) .... state_breakdown() keys "pending_*" / "raw_*"

Fill in `setup`, `predict` and `update`, then register it in tsl/online/methods/__init__.py (already
registered under "vine"; it raises until implemented so it can never silently produce numbers).
"""
from __future__ import annotations

import numpy as np

from ..base import OnlineContext, OnlineMethod
from ..protocol import RevealedInfo


def revision(ctx: OnlineContext, t: int) -> np.ndarray | None:
    """rho_{t,h} = yhat_{t,h-1} - yhat_{t-1,h} for h = 2..H, as an (H-1, C) array (label-free).

    Row k (0-based) is the revision of target t+k between origins t-1 and t. None at the first origin.
    """
    if t - 1 < ctx.first_origin:
        return None
    cur, prev = ctx.base_forecast(t), ctx.base_forecast(t - 1)
    return cur[:-1] - prev[1:]


class VINE(OnlineMethod):
    name = "vine"
    uses_prefix = False   # set True for the P-prefix anchor variant

    def __init__(self, gamma: float = 0.995, gamma_a: float = 0.999, ridge: float = 0.1, k_f: int = 3,
                 k_r: int = 3, rho_max: float = 0.98, solve_every: int = 1, **kw):
        super().__init__(**kw)
        self.hp = dict(gamma=gamma, gamma_a=gamma_a, ridge=ridge, k_f=k_f, k_r=k_r, rho_max=rho_max,
                       solve_every=solve_every)

    def setup(self, ctx: OnlineContext) -> None:
        raise NotImplementedError("VINE is a stub: implement it in tsl/online/methods/vine.py "
                                  "(the harness interfaces it needs are documented at the top of the file).")

    def predict(self, x: np.ndarray, base_pred: np.ndarray | None, t: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def update(self, info: RevealedInfo) -> None:  # pragma: no cover
        raise NotImplementedError
