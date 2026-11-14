"""Proceed (Zhao & Shen, KDD 2025, "Proactive Model Adaptation Against Concept Drift for Online Time Series
Forecasting"): port of the official code (github.com/SJTU-DMTai/OnlineTSF, files adapter/proceed.py,
adapter/module/{generator,down_up}.py, exp/{exp_online,exp_proceed,exp_basic}.py, run.py defaults and
scripts/online/*/Proceed*) to the truthful delayed-label protocol.

Model: every nn.Linear of the backbone becomes a Down_Up adapter layer
        out = ((x * scale2) W^T) * scale + (shift + b),   scale = 1 + a_out, scale2 = 1 + a_in, shift = a_shift,
with a = [a_out, a_in, a_shift] produced per sample by a generator from the estimated concept drift
        drift = clip_{||.||<=1}( mean_c mlp1(x)  -  mean_{c,batch} mlp2([x ; y]_recent) ),
mlp1: L -> d_c -> d_c (GELU), mlp2: L+H -> d_c -> d_c, applied along time per channel; generator per adapted layer:
sigmoid(W0 drift + b0) (bottleneck r) -> W1 (zero init) + b1 (zero init), so the model starts at the backbone.
`recent` is the newest sample whose label is complete.

Offline phase (Exp_Main.train with the Proceed framework, freeze=False / "fulltune"): backbone, mlps and generator
    trained jointly with MSE and Adam at `lr` on the training split; after each batch the batch becomes `recent`
    (so the generator learns from batch-to-batch drift). Early stopping (patience 3, <= 10 epochs) on the last 10% of
    the released training windows (the original early-stops on an online pass over the validation split, which in
    our protocol is part of the stream).
Online phase (Exp_Proceed.online -> Exp_Online.online, recent_num = 1): freeze_adapter(True) freezes mlp1, mlp2 and
    the generator weights; at each step the newest complete sample (origin t - d; t - H in the original, which
    assumes d = H) is used for ONE Adam step at `online_lr` on the backbone and the generator's output biases b1
    (the optimiser state of the offline phase is kept, --save_opt), then becomes `recent`; the current window is
    forecast with the adaptation generated from its drift against `recent`.
Not ported: the original's offline `update_valid` pass over the validation split (it needs all validation labels
    before the test stream starts; here the validation segment is streamed online, as for every other method);
    ema = 0 and individual_generator = False (the published defaults).
Defaults: concept_dim 50, bottleneck 32 (scripts/online/TCN_RevIN/Proceed/ETTh2.sh), lr 1e-3, online_lr 3e-5;
    tuned on validation.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OnlineMethod
from ..protocol import RevealedInfo
from .dsof import _released_windows

_OFFLINE_CACHE: dict = {}


class AdaptedLinear(nn.Module):
    """adapter/module/down_up.Linear (merge_weights path is numerically the same as the per-sample _ssf path)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.adaptation = None

    @property
    def out_dim(self):
        return self.out_features + self.in_features + (self.out_features if self.bias is not None else 0)

    def forward(self, x):
        a = self.adaptation
        if a is None:
            return F.linear(x, self.weight, self.bias)
        o, i = self.out_features, self.in_features
        shape = (a.shape[0],) + (1,) * (x.dim() - 2) + (-1,)        # per sample, broadcast over middle dims
        scale = (a[:, :o] + 1).reshape(shape)
        scale2 = (a[:, o:o + i] + 1).reshape(shape)
        res = F.linear(x * scale2, self.weight) * scale
        if self.bias is not None:
            res = res + (a[:, -o:].reshape(shape) + self.bias)
        return res


class Bottleneck(nn.Module):
    """adapter/module/generator.Bottleneck with shared=True, n_layers=1, need_bias=True, activation=Sigmoid."""

    def __init__(self, in_dim: int, out_dim: int, r: int):
        super().__init__()
        self.w0 = nn.Linear(in_dim, r, bias=False)
        self.b0 = nn.Parameter(torch.empty(1, r))
        bound = 1 / math.sqrt(in_dim)
        nn.init.uniform_(self.b0, -bound, bound)
        self.w1 = nn.Parameter(torch.zeros(out_dim, r))
        self.b1 = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, z):
        return torch.sigmoid(self.w0(z) + self.b0) @ self.w1.T + self.b1


def _replace_linears(module: nn.Module, prefix="") -> list[tuple[str, AdaptedLinear]]:
    out = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new = AdaptedLinear(child)
            setattr(module, name, new)
            out.append((prefix + name, new))
        else:
            out += _replace_linears(child, prefix + name + ".")
    return out


class ProceedNet(nn.Module):
    def __init__(self, backbone: nn.Module, L: int, H: int, C: int, concept_dim: int, bottleneck: int):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        layers = _replace_linears(self.backbone)
        self.names = [n for n, _ in layers]
        self.layers = [m for _, m in layers]
        self.gen = nn.ModuleList([Bottleneck(concept_dim, m.out_dim, bottleneck) for m in self.layers])
        self.mlp1 = nn.Sequential(nn.Linear(L, concept_dim), nn.GELU(), nn.Linear(concept_dim, concept_dim))
        self.mlp2 = nn.Sequential(nn.Linear(L + H, concept_dim), nn.GELU(), nn.Linear(concept_dim, concept_dim))
        self.recent = torch.zeros(1, L + H, C)          # newest complete sample(s); not part of state_dict

    def forward(self, x):
        concept = self.mlp1(x.transpose(1, 2)).mean(-2)                       # (B, d_c)
        recent = self.mlp2(self.recent.transpose(1, 2)).mean(-2).mean(0)       # (d_c,)
        drift = concept - recent
        nrm = torch.norm(drift, dim=-1, keepdim=True)
        drift = drift * torch.clip(1.0 / nrm, max=1)                          # clip(x, max_norm=1)
        for g, m in zip(self.gen, self.layers):
            m.adaptation = g(drift)
        try:
            return self.backbone(x)
        finally:
            for m in self.layers:
                m.adaptation = None

    def freeze_adapter(self, freeze: bool = True):
        for mod in (self.mlp1, self.mlp2):
            mod.requires_grad_(not freeze)
        for g in self.gen:
            g.w0.requires_grad_(not freeze)
            g.b0.requires_grad_(not freeze)
            g.w1.requires_grad_(not freeze)
            g.b1.requires_grad_(True)            # freeze_bias(False): output biases stay trainable online


class Proceed(OnlineMethod):
    name = "proceed"
    needs_base_pred = False

    def __init__(self, concept_dim: int = 50, bottleneck: int = 32, lr: float = 1e-3, online_lr: float = 3e-5,
                 train_epochs: int = 10, patience: int = 3, train_batch: int = 32, **kw):
        super().__init__(**kw)
        self.concept_dim, self.bottleneck = int(concept_dim), int(bottleneck)
        self.lr, self.online_lr = float(lr), float(online_lr)
        self.train_epochs, self.patience, self.train_batch = int(train_epochs), int(patience), int(train_batch)

    def setup(self, ctx):
        super().setup(ctx)
        L, H, C, dev = ctx.L, ctx.H, ctx.C, ctx.device
        self.net = ProceedNet(ctx.backbone, L, H, C, self.concept_dim, self.bottleneck).to(dev)
        self.net.recent = self.net.recent.to(dev)
        self.opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad], lr=self.lr)
        self._offline(ctx)
        self.net.freeze_adapter(True)
        for gp in self.opt.param_groups:
            gp["lr"] = self.online_lr
        self.net.eval()
        self.n_updates = 0

    def _offline(self, ctx):
        X, Y = _released_windows(ctx, ctx.L, ctx.H)
        key = ("proceed", id(ctx.backbone), ctx.L, ctx.H, ctx.C, self.concept_dim, self.bottleneck, self.lr,
               self.train_epochs, self.patience, self.train_batch, ctx.seed,
               hashlib.sha1(X.tobytes() + Y.tobytes()).hexdigest())
        if key in _OFFLINE_CACHE:
            net_state, opt_state, recent = _OFFLINE_CACHE[key]
            self.net.load_state_dict(net_state)
            self.opt.load_state_dict(copy.deepcopy(opt_state))   # load_state_dict aliases same-device tensors
            self.net.recent = recent.clone()
            return
        dev = ctx.device
        if len(X) >= 20:
            nv = max(1, len(X) // 10)
            Xt, Yt, Xv, Yv = X[:-nv], Y[:-nv], X[-nv:], Y[-nv:]
            rng = np.random.default_rng(ctx.seed)
            best, bad = float("inf"), 0
            best_state = (copy.deepcopy(self.net.state_dict()), copy.deepcopy(self.opt.state_dict()))
            for _ in range(self.train_epochs):
                self.net.train()
                for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // self.train_batch)):
                    xb, yb = torch.from_numpy(Xt[b]).to(dev), torch.from_numpy(Yt[b]).to(dev)
                    self.opt.zero_grad(set_to_none=True)
                    loss = ((self.net(xb) - yb) ** 2).mean()
                    loss.backward()
                    self.opt.step()
                    self.net.recent = torch.cat([xb, yb], 1).detach()         # Exp_Proceed._update
                self.net.eval()
                with torch.no_grad():                                          # validation: plain forward per batch
                    v, n = 0.0, 0
                    for i in range(0, len(Xv), 256):
                        xv, yv = torch.from_numpy(Xv[i:i + 256]).to(dev), torch.from_numpy(Yv[i:i + 256]).to(dev)
                        v += float(((self.net(xv) - yv) ** 2).sum())
                        n += yv.numel()
                v /= max(n, 1)
                if v < best - 1e-12:
                    best, bad = v, 0
                    best_state = (copy.deepcopy(self.net.state_dict()), copy.deepcopy(self.opt.state_dict()))
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            self.net.load_state_dict(best_state[0])
            self.opt.load_state_dict(best_state[1])
        self.net.zero_grad(set_to_none=True)
        if len(X):                                   # newest complete sample before the stream = `recent`
            self.net.recent = torch.from_numpy(np.concatenate([X[-1:], Y[-1:]], 1)).to(dev)
        _OFFLINE_CACHE[key] = (copy.deepcopy(self.net.state_dict()), copy.deepcopy(self.opt.state_dict()),
                               self.net.recent.clone())

    def update(self, info: RevealedInfo):
        if not info.matured:
            return
        dev = self.ctx.device
        for m in sorted(info.matured, key=lambda q: q.origin):       # one matured block per step in practice
            xb = torch.from_numpy(np.asarray(m.x, np.float32)[None]).to(dev)
            yb = torch.from_numpy(np.asarray(m.y, np.float32)[None]).to(dev)
            self.net.train()
            self.opt.zero_grad(set_to_none=True)
            loss = ((self.net(xb) - yb) ** 2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.net.recent = torch.cat([xb, yb], 1).detach()
            self.n_updates += 1
        self.net.eval()

    @torch.no_grad()
    def predict(self, x, base_pred, t):
        xb = torch.from_numpy(np.asarray(x, np.float32)[None]).to(self.ctx.device)
        return self.net(xb)[0].float().cpu().numpy()

    def state_breakdown(self):
        return {"net": self.net, "recent_sample": self.net.recent, "optimizer": [v for s in self.opt.state.values() for v in s.values()
                                               if torch.is_tensor(v)]}

    def extra_metrics(self):
        return {"n_updates": self.n_updates}
