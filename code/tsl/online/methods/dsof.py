"""DSOF (Lau et al., ICLR 2025, "Fast and Slow Streams for Online Time Series Forecasting Without Information
Leakage"): port of the official code (github.com/yyalau/iclr2025_dsof, files src/trainers/trainerBaseERTD.py and
src/trainers/w_student/{trainerBaseFramework,trainerFramework_ERTDRes}.py, configs config/trainer/w_student/residual/
dsof.yaml, config/optimizer/w_student/DLinear_MLP/*.yaml, config/model/MLP.yaml) to the truthful delayed-label protocol.

Model (ERTDRes, "residual" student):
    main  = the backbone (trainable copy);   student = MLP on [x ; main(x).detach()] along time, per channel
    forecast = main(x).detach() + student([x ; main(x).detach()])          (mlp width 16, depth 3, Identity act.)
Slow stream (experience replay, `er`): every step on which a COMPLETE label block arrives, the sample enters a FIFO
    buffer (300); once it holds >= 32 samples, `num_ERepochs` = 1 update on 32 random buffer samples:
    MSE(main, y) -> main (AdamW, amsgrad)  and  MSE(forecast, y) -> student (AdamW, amsgrad).
Fast stream (temporal difference, `td`, k in td_k = [1]): the forecast issued k steps before the newest released
    value is regressed on the target [observed y_{s+1..s+k} ; teacher forecast] with weights
    [1]*(k-1) + [gamma^0, gamma^1, ...] (gamma = `discounted` = 0.9); the teacher is the CURRENT main model's forecast
    from the origin right after the observed prefix (td4update_pseudo uses predT, the main model only). Only the
    student receives gradient (the main forecast enters the student detached).
Order per step (DSOF test_step): ER, then TD, then forecast.

Truthful-delay adaptation (the original assumes d = H, i.e. y_t is usable at t):
    * with point lag l = d - H the newest released value is y_{t-l} (`history.label_limit()`), so the TD origin is
      s = t - l - k and the teacher origin is t - l (identical to the original when d = H);
    * a label block enters the replay buffer when it is released (s + d), not at s + H.
Offline phase: DSOF trains main and student jointly on the training split (main on MSE(main, y), student on
    MSE(forecast, y)). Our main is the shared pretrained backbone (trained on the same objective), so only the
    student is trained offline here, on the training windows released before the stream starts, AdamW(amsgrad) at
    the published `opt_student.train` rate, <= 10 epochs, early stopping (patience 3) on the last 10% of windows.
Learning rates: the published per-dataset DLinear_MLP optimiser files; datasets without a published file use the
    ETTh2 file (hourly) or ETTm1 file (15-min). `lr_scale` multiplies the three online rates (tuned on validation).
    The main model's online-TD optimiser of the original never receives gradient (the main forecast is detached in
    the TD loss), so it is omitted.
"""
from __future__ import annotations

import copy
import hashlib
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from ..base import OnlineMethod
from ..protocol import RevealedInfo

# published optimiser files config/optimizer/w_student/DLinear_MLP/<dataset>.yaml (lr values, AdamW/Adam, amsgrad)
PRESETS = {
    "ETTh2": dict(main_batch=0.00028889418723619334, student_batch=0.00015686974622312612,
                  student_online=0.00036288062287989935, student_train=0.001026953060677239),
    "ETTm1": dict(main_batch=0.0001226758892445545, student_batch=0.0002742142681649708,
                  student_online=0.00030370016586033394, student_train=0.009611051935254944),
    "Weather": dict(main_batch=0.000708135485251135, student_batch=7.249430683445887e-05,
                    student_online=0.00022886838382246013, student_train=0.0019221120023831346),
    "Exchange": dict(main_batch=0.00010112274178326875, student_batch=0.000783882718834249,
                     student_online=0.0008776466424276737, student_train=0.0011932670228165677),
}
FALLBACK = {"ETTh1": "ETTh2", "ETTm2": "ETTm1"}
_OFFLINE_CACHE: dict = {}


class StudentMLP(nn.Module):
    """config/model/MLP.yaml + src/models/MLP.py: per-channel MLP over time, (B, T_in, C) -> (B, H, C)."""

    def __init__(self, n_in: int, n_out: int, width: int = 16, depth: int = 3, act: str = "Identity",
                 dropout: float = 0.0):
        super().__init__()
        self.input = nn.Linear(n_in, width)
        self.hiddens = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 2)])
        self.output = nn.Linear(width, n_out)
        self.act = getattr(nn, act)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        h = self.act(self.dropout(self.input(z.permute(0, 2, 1))))
        for lay in self.hiddens:
            h = self.act(self.dropout(lay(h)))
        return self.output(h).permute(0, 2, 1)


def _released_windows(ctx, L: int, H: int):
    """(X, Y) windows whose label block is fully released at setup time (inputs from x, targets from y)."""
    lim = ctx.history.label_limit()
    if lim - H < L - 1:
        C = ctx.C
        return np.zeros((0, L, C), np.float32), np.zeros((0, H, C), np.float32)
    xs = ctx.history.inputs(0, lim + 1)
    ys = ctx.history.labels(0, lim + 1)
    origins = np.arange(L - 1, lim - H + 1)
    X = np.stack([xs[s - L + 1:s + 1] for s in origins]).astype(np.float32)
    Y = np.stack([ys[s + 1:s + H + 1] for s in origins]).astype(np.float32)
    return X, Y


class DSOF(OnlineMethod):
    name = "dsof"
    needs_base_pred = False

    def __init__(self, preset: str = "auto", lr_scale: float = 1.0, discounted: float = 0.9, td_k=(1,),
                 buffer: int = 300, batch_replay: int = 32, num_er_epochs: int = 1, er: bool = True, td: bool = True,
                 width: int = 16, depth: int = 3, act: str = "Identity", train_epochs: int = 10,
                 patience: int = 3, train_batch: int = 32, **kw):
        super().__init__(**kw)
        self.preset, self.lr_scale, self.discounted = preset, float(lr_scale), float(discounted)
        self.td_k = [int(k) for k in (td_k if isinstance(td_k, (list, tuple)) else [td_k])]
        self.buffer_size, self.batch_replay, self.num_er_epochs = int(buffer), int(batch_replay), int(num_er_epochs)
        self.use_er, self.use_td = bool(er), bool(td)
        self.width, self.depth, self.act = int(width), int(depth), act
        self.train_epochs, self.patience, self.train_batch = int(train_epochs), int(patience), int(train_batch)

    # ------------------------------------------------------------------ setup
    def _lrs(self, ctx):
        ds = (ctx.pretrain_info or {}).get("dataset", "ETTh2")
        key = self.preset if self.preset != "auto" else ds
        key = key if key in PRESETS else FALLBACK.get(key, "ETTm1" if (ctx.pretrain_info or {}).get(
            "steps_per_day", 24) > 24 else "ETTh2")
        return key, PRESETS[key]

    def setup(self, ctx):
        super().setup(ctx)
        L, H, C, dev = ctx.L, ctx.H, ctx.C, ctx.device
        self.preset_used, lr = self._lrs(ctx)
        self.main = copy.deepcopy(ctx.backbone).to(dev)
        for p in self.main.parameters():
            p.requires_grad_(True)
        self.main.eval()
        self.student = StudentMLP(L + H, H, self.width, self.depth, self.act).to(dev)
        self._offline_student(ctx, lr["student_train"])
        s = self.lr_scale
        adamw = lambda ps, r: torch.optim.AdamW(ps, lr=r, amsgrad=True)     # noqa: E731
        self.opt_main_batch = adamw(self.main.parameters(), lr["main_batch"] * s)
        self.opt_student_batch = adamw(self.student.parameters(), lr["student_batch"] * s)
        self.opt_student_online = adamw(self.student.parameters(), lr["student_online"] * s)
        self.buf: deque = deque(maxlen=self.buffer_size)
        self.rng = np.random.default_rng(ctx.seed)
        self.lag = ctx.delay - H
        ks = [k for k in self.td_k if k <= H]
        self.ks = ks
        self.td_w = torch.cat([torch.cat([torch.ones(k - 1), torch.tensor([self.discounted ** i
                                                                           for i in range(H - k + 1)])])[None, :, None]
                               for k in ks], 0).float().to(dev) if ks else None
        self.n_er = self.n_td = 0

    def _offline_student(self, ctx, lr_train: float):
        X, Y = _released_windows(ctx, ctx.L, ctx.H)
        key = ("dsof", id(ctx.backbone), ctx.L, ctx.H, ctx.C, lr_train, self.width, self.depth, self.act,
               self.train_epochs, self.patience, self.train_batch, ctx.seed,
               hashlib.sha1(X.tobytes() + Y.tobytes()).hexdigest())
        if key in _OFFLINE_CACHE:
            self.student.load_state_dict(_OFFLINE_CACHE[key])
            return
        # (the student keeps PyTorch's default nn.Linear init, drawn after the evaluator seeds torch)
        if len(X) >= 20:
            dev = ctx.device
            nv = max(1, len(X) // 10)
            Xt, Yt, Xv, Yv = X[:-nv], Y[:-nv], X[-nv:], Y[-nv:]
            opt = torch.optim.AdamW(self.student.parameters(), lr=lr_train, amsgrad=True)
            rng = np.random.default_rng(ctx.seed)
            best, best_state, bad = float("inf"), copy.deepcopy(self.student.state_dict()), 0
            with torch.no_grad():
                Mt = self._main_np(Xt)
                Mv = self._main_np(Xv)
            for _ in range(self.train_epochs):
                self.student.train()
                for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // self.train_batch)):
                    xb, mb = torch.from_numpy(Xt[b]).to(dev), torch.from_numpy(Mt[b]).to(dev)
                    yb = torch.from_numpy(Yt[b]).to(dev)
                    opt.zero_grad(set_to_none=True)
                    loss = ((mb + self.student(torch.cat([xb, mb], 1)) - yb) ** 2).mean()
                    loss.backward()
                    opt.step()
                self.student.eval()
                with torch.no_grad():
                    xv, mv, yv = (torch.from_numpy(a).to(dev) for a in (Xv, Mv, Yv))
                    v = float(((mv + self.student(torch.cat([xv, mv], 1)) - yv) ** 2).mean())
                if v < best - 1e-12:
                    best, best_state, bad = v, copy.deepcopy(self.student.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            self.student.load_state_dict(best_state)
        self.student.zero_grad(set_to_none=True)
        self.student.eval()
        _OFFLINE_CACHE[key] = copy.deepcopy(self.student.state_dict())

    def _main_np(self, X: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(X), 512):
            out.append(self.main(torch.from_numpy(X[i:i + 512]).to(self.ctx.device)).float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, self.ctx.H, self.ctx.C), np.float32)

    # ------------------------------------------------------------------ online
    def _forward(self, xb: torch.Tensor):
        """(predT with graph to main, combined forecast with graph to the student only)."""
        pt = self.main(xb)
        md = pt.detach()
        return pt, md + self.student(torch.cat([xb, md], 1))

    def update(self, info: RevealedInfo):
        dev = self.ctx.device
        for m in info.matured:
            self.buf.append((np.asarray(m.x, np.float32), np.asarray(m.y, np.float32)))
        # ---- slow stream: experience replay on complete labels (only on steps where a block was released)
        self.main.zero_grad(set_to_none=True)
        self.student.zero_grad(set_to_none=True)
        if self.use_er and info.matured and len(self.buf) >= self.batch_replay:
            self.main.train()
            self.student.train()
            for _ in range(self.num_er_epochs):
                idx = self.rng.choice(len(self.buf), self.batch_replay, replace=False)
                xb = torch.from_numpy(np.stack([self.buf[i][0] for i in idx])).to(dev)
                yb = torch.from_numpy(np.stack([self.buf[i][1] for i in idx])).to(dev)
                pt, pred = self._forward(xb)
                ((pt - yb) ** 2).mean().backward()
                ((pred - yb) ** 2).mean().backward()
                self.opt_main_batch.step()
                self.opt_student_batch.step()
                self.opt_main_batch.zero_grad(set_to_none=True)
                self.opt_student_batch.zero_grad(set_to_none=True)
            self.n_er += 1
            self.main.eval()
            self.student.eval()
        # ---- fast stream: TD on the newest released values + teacher (current main model) forecast
        if self.use_td and self.ks:
            L, H = self.ctx.L, self.ctx.H
            lim = self.ctx.history.label_limit()            # newest released value y_{t-l}
            if lim - max(self.ks) - L + 1 >= 0:
                hist = self.ctx.history
                x_t = torch.from_numpy(hist.inputs(lim - L + 1, lim + 1)[None]).to(dev)
                with torch.no_grad():
                    teacher = self.main(x_t)                  # predT at origin t - l (td4update_pseudo)
                xs, tg = [], []
                for k in self.ks:
                    s = lim - k
                    xs.append(hist.inputs(s - L + 1, s + 1))
                    yk = torch.from_numpy(hist.labels(s + 1, s + k + 1)[None]).to(dev)
                    tg.append(torch.cat([yk, teacher[:, :H - k]], 1))
                xb = torch.from_numpy(np.stack(xs)).to(dev)
                self.student.train()
                _, pred = self._forward(xb)
                loss = (((pred - torch.cat(tg, 0)) ** 2) * self.td_w).mean()
                loss.backward()
                self.opt_student_online.step()
                self.opt_student_online.zero_grad(set_to_none=True)
                self.main.zero_grad(set_to_none=True)
                self.student.eval()
                self.n_td += 1

    @torch.no_grad()
    def predict(self, x, base_pred, t):
        xb = torch.from_numpy(np.asarray(x, np.float32)[None]).to(self.ctx.device)
        _, pred = self._forward(xb)
        return pred[0].float().cpu().numpy()

    def state_breakdown(self):
        opt_state = [v for o in (self.opt_main_batch, self.opt_student_batch, self.opt_student_online)
                     for s in o.state.values() for v in s.values() if torch.is_tensor(v)]
        return {"main": self.main, "student": self.student, "optimizer": opt_state, "raw_buffer": list(self.buf)}

    def extra_metrics(self):
        return {"n_er_updates": self.n_er, "n_td_updates": self.n_td, "preset": self.preset_used}
