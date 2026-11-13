"""Reference baselines: frozen backbone, delayed online gradient descent, periodic retraining."""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch

from ...data.windows import make_windows
from ...models.train import fit
from ...utils import ByteCount
from ..base import OnlineMethod
from ..protocol import RevealedInfo


class Frozen(OnlineMethod):
    """No update: the pretrained backbone forecast is used as-is."""
    name = "frozen"

    def predict(self, x, base_pred, t):
        return base_pred


class Persistence(OnlineMethod):
    """Backbone-free naive forecasts from the input window: `last` repeats x_t; `seasonal` repeats the last season
    (x_{t+h-S} for the season S = steps per day, read through the guarded input history when S > L)."""
    name = "persistence"
    needs_base_pred = False

    def __init__(self, kind: str = "last", season: int | None = None, **kw):
        super().__init__(**kw)
        if kind not in ("last", "seasonal"):
            raise ValueError("kind must be last|seasonal")
        self.kind, self.season = kind, season

    def setup(self, ctx):
        super().setup(ctx)
        self.S = int(self.season or (ctx.pretrain_info or {}).get("steps_per_day") or 24)

    def predict(self, x, base_pred, t):
        H = self.ctx.H
        if self.kind == "last":
            return np.repeat(x[-1:], H, 0)
        S = self.S
        hist = x if S <= x.shape[0] else self.ctx.history.inputs(t - S + 1, t + 1)
        idx = (np.arange(1, H + 1) - 1) % S          # x_{t+h-S} = hist[-S + (h-1) % S]
        return hist[-S:][idx]


class _TorchModelMethod(OnlineMethod):
    needs_base_pred = False

    def _init_model(self, ctx):
        self.model = copy.deepcopy(ctx.backbone).to(ctx.device)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x, base_pred, t):
        self.model.eval()
        xb = torch.from_numpy(x[None]).to(self.ctx.device)
        return self.model(xb)[0].float().cpu().numpy()


class DelayedOGD(_TorchModelMethod):
    """Delayed online gradient descent ("DGrad"): when the label of origin s-d matures at time s, take
    `steps` gradient steps on a mini-batch made of the `batch` most recently matured pairs.

    scope: "full" (all parameters) or "head" (last projection layer only).
    In prefix mode with use_prefix=True, revealed prefixes contribute a masked loss (prefix positions only).
    """
    name = "ogd"

    def __init__(self, scope: str = "full", lr: float = 1e-4, optimizer: str = "adam", batch: int = 1,
                 steps: int = 1, use_prefix: bool = False, grad_clip: float | None = 1.0, **kw):
        super().__init__(**kw)
        self.scope, self.lr, self.opt_name = scope, lr, optimizer
        self.batch, self.steps, self.use_prefix_flag, self.grad_clip = batch, steps, use_prefix, grad_clip
        self.uses_prefix = use_prefix
        self.name = f"ogd_{scope}"

    def setup(self, ctx):
        super().setup(ctx)
        self._init_model(ctx)
        if self.scope == "head":
            for p in self.model.parameters():
                p.requires_grad_(False)
            params = self.model.head_parameters()
            for p in params:
                p.requires_grad_(True)
        else:
            params = list(self.model.parameters())
        self.params = [p for p in params if p.requires_grad]
        Opt = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[self.opt_name]
        self.opt = Opt(self.params, lr=self.lr)
        self.buf = deque(maxlen=self.batch)
        self.n_updates = 0

    def update(self, info: RevealedInfo):
        for m in info.matured:
            self.buf.append((m.x, m.y))
        pre = info.prefixes if (self.use_prefix_flag and self.ctx.mode == "prefix") else []
        if not self.buf and not pre:
            return
        dev = self.ctx.device
        self.model.train()
        for _ in range(self.steps):
            self.opt.zero_grad(set_to_none=True)
            loss = 0.0
            if self.buf:
                xb = torch.from_numpy(np.stack([b[0] for b in self.buf])).to(dev)
                yb = torch.from_numpy(np.stack([b[1] for b in self.buf])).to(dev)
                loss = ((self.model(xb) - yb) ** 2).mean()
            if pre:
                H = self.ctx.H
                xp = torch.from_numpy(np.stack([p.x for p in pre])).to(dev)
                yp = np.zeros((len(pre), H, self.ctx.C), np.float32)
                mk = np.zeros((len(pre), H, 1), np.float32)
                for i, p in enumerate(pre):
                    yp[i, :p.k] = p.y_prefix
                    mk[i, :p.k] = 1.0
                yp, mk = torch.from_numpy(yp).to(dev), torch.from_numpy(mk).to(dev)
                lp = (((self.model(xp) - yp) ** 2) * mk).sum() / (mk.sum() * self.ctx.C).clamp_min(1)
                loss = loss + lp
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            self.opt.step()
        self.n_updates += 1
        self.model.eval()

    def state_breakdown(self):
        return {"params": self.model, "optimizer": [v for s in self.opt.state.values() for v in s.values()
                                                    if torch.is_tensor(v)],
                "raw_buffer": list(self.buf)}

    def extra_metrics(self):
        return {"n_updates": self.n_updates}


class PeriodicRetrain(_TorchModelMethod):
    """Every `period` steps, re-fit the backbone on all rows revealed so far (or a rolling `window`).

    from_scratch=False fine-tunes the current weights (warm start); True re-initialises from the
    pretrained checkpoint. Retraining cost shows up in the update latency (spiky by design).
    """
    name = "periodic"

    def __init__(self, period: int = 500, window: int | None = None, epochs: int = 1, lr: float = 1e-4,
                 batch_size: int = 32, max_steps: int | None = 200, from_scratch: bool = False,
                 val_frac: float = 0.0, period_frac: float | None = None, **kw):
        super().__init__(**kw)
        self.period, self.window, self.epochs, self.lr = period, window, epochs, lr
        self.period_frac = period_frac   # e.g. 0.1 = retrain every 10% of the online segment
        self.bs, self.max_steps, self.from_scratch, self.val_frac = batch_size, max_steps, from_scratch, val_frac
        self.name = f"periodic_{period}"

    def setup(self, ctx):
        super().setup(ctx)
        self._init_model(ctx)
        if self.period_frac:
            self.period = max(1, int(self.period_frac * ctx.extra.get("n_forecasts", 10 * self.period)))
            self.name = f"periodic_{self.period_frac:g}"
        self.init_state = copy.deepcopy(ctx.backbone.state_dict()) if self.from_scratch else None
        self.last = ctx.first_origin
        self.n_retrains = 0

    def update(self, info: RevealedInfo):
        now = info.now
        if now - self.last < self.period:
            return
        # (labels of the most recent d-H rows are not yet released; observed_until() enforces that)
        self.last = now
        L, H = self.ctx.L, self.ctx.H
        end = self.ctx.history.observed_until() + 1
        start = 0 if self.window is None else max(0, end - self.window)
        z = self.ctx.history.get(start, end)
        X, Y, _ = make_windows(z, 0, len(z), L, H)
        if len(X) == 0:
            return
        if self.init_state is not None:
            self.model.load_state_dict(self.init_state)
        seed = self.ctx.seed + self.n_retrains
        fit(self.model, X, Y, None, None, epochs=self.epochs, lr=self.lr, batch_size=self.bs, patience=10 ** 9,
            device=self.ctx.device, seed=seed, max_steps_per_epoch=self.max_steps, verbose=False)
        self.model.eval()
        self.n_retrains += 1

    def state_breakdown(self):
        # retraining needs the raw history it refits on: count it as retained state
        h = self.ctx.history
        rows = h.observed_until() + 1 if self.window is None else min(self.window, h.observed_until() + 1)
        return {"params": self.model, "raw_retained_history": ByteCount(max(rows, 0) * h.C * 4)}

    def extra_metrics(self):
        return {"n_retrains": self.n_retrains}
