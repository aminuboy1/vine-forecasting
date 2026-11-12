"""Compact, clean re-implementations of three standard backbones.

All models map x: (B, L, C) -> y: (B, H, C), are channel-count agnostic (so they work with
variable-set changes in the continual harness) and expose:
  * `head_parameters()`  -> parameters of the final projection (for "last-layer" online updates)
  * optional `channel_mask` (B, C) bool argument (True = channel present). DLinear and PatchTST are
    channel-independent so the mask only matters for iTransformer (key padding in attention).

These follow the published architectures (DLinear: Zeng et al. AAAI'23; PatchTST: Nie et al. ICLR'23;
iTransformer: Liu et al. ICLR'24) and the defaults of thuml/Time-Series-Library (MIT licence), but are
written from scratch here; no TSLib code is vendored.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .layers import RevIN, SeriesDecomp


class Backbone(nn.Module):
    L: int
    H: int

    def head_parameters(self):
        raise NotImplementedError

    def head_module(self) -> nn.Module:
        raise NotImplementedError


# --------------------------------------------------------------------------- DLinear
class DLinear(Backbone):
    """Decomposition-linear. Shared weights across channels (`individual=False`, TSLib default)."""

    def __init__(self, L: int, H: int, kernel_size: int = 25, revin: bool = True, **_):
        super().__init__()
        self.L, self.H = L, H
        self.decomp = SeriesDecomp(kernel_size)
        self.seasonal = nn.Linear(L, H)
        self.trend = nn.Linear(L, H)
        with torch.no_grad():  # TSLib initialisation: average over the window
            self.seasonal.weight.fill_(1.0 / L)
            self.trend.weight.fill_(1.0 / L)
        self.revin = RevIN() if revin else None

    def forward(self, x, channel_mask=None):
        if self.revin is not None:
            x, st = self.revin.norm(x)
        s, t = self.decomp(x)
        y = self.seasonal(s.transpose(1, 2)) + self.trend(t.transpose(1, 2))
        y = y.transpose(1, 2)
        if self.revin is not None:
            y = self.revin.denorm(y, st)
        return y

    def head_parameters(self):
        return list(self.seasonal.parameters()) + list(self.trend.parameters())

    def head_module(self):
        return nn.ModuleList([self.seasonal, self.trend])


# --------------------------------------------------------------------------- Transformer bits
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm="layer"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.norm_kind = norm
        if norm == "batch":
            self.n1, self.n2 = nn.BatchNorm1d(d_model), nn.BatchNorm1d(d_model)
        else:
            self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def _norm(self, n, x):
        if self.norm_kind == "batch":
            return n(x.transpose(1, 2)).transpose(1, 2)
        return n(x)

    def forward(self, x, key_padding_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self._norm(self.n1, x + self.drop(a))
        x = self._norm(self.n2, x + self.drop(self.ff(x)))
        return x


# --------------------------------------------------------------------------- PatchTST
class PatchTST(Backbone):
    """Channel-independent patch Transformer (supervised PatchTST)."""

    def __init__(self, L: int, H: int, patch_len: int = 16, stride: int = 8, d_model: int = 128,
                 n_heads: int = 16, e_layers: int = 3, d_ff: int = 256, dropout: float = 0.2,
                 head_dropout: float = 0.0, revin: bool = True, **_):
        super().__init__()
        self.L, self.H = L, H
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (L - patch_len) // stride + 2  # +1 for end padding (as in PatchTST)
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, norm="batch")
                                     for _ in range(e_layers)])
        self.head = nn.Sequential(nn.Flatten(-2), nn.Dropout(head_dropout), nn.Linear(self.n_patches * d_model, H))
        self.revin = RevIN() if revin else None

    def forward(self, x, channel_mask=None):
        B, L, C = x.shape
        if self.revin is not None:
            x, st = self.revin.norm(x)
        u = x.transpose(1, 2).reshape(B * C, L)
        u = torch.cat([u, u[:, -1:].expand(-1, self.stride)], 1)        # replicate-pad end
        p = u.unfold(-1, self.patch_len, self.stride)                    # (BC, N, P)
        h = self.drop(self.embed(p) + self.pos)
        for lyr in self.layers:
            h = lyr(h)
        y = self.head(h).reshape(B, C, self.H).transpose(1, 2)
        if self.revin is not None:
            y = self.revin.denorm(y, st)
        return y

    def head_parameters(self):
        return list(self.head.parameters())

    def head_module(self):
        return self.head


# --------------------------------------------------------------------------- iTransformer
class iTransformer(Backbone):
    """Inverted Transformer: each variate's whole lookback is one token; attention across variates."""

    def __init__(self, L: int, H: int, d_model: int = 128, n_heads: int = 8, e_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1, revin: bool = True, **_):
        super().__init__()
        self.L, self.H = L, H
        self.embed = nn.Linear(L, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, norm="layer")
                                     for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, H)
        self.revin = RevIN() if revin else None

    def forward(self, x, channel_mask=None):
        if self.revin is not None:
            x, st = self.revin.norm(x)
        h = self.drop(self.embed(x.transpose(1, 2)))                     # (B, C, D)
        kpm = None if channel_mask is None else ~channel_mask.bool()
        for lyr in self.layers:
            h = lyr(h, key_padding_mask=kpm)
        y = self.head(self.norm(h)).transpose(1, 2)                      # (B, H, C)
        if self.revin is not None:
            y = self.revin.denorm(y, st)
        return y

    def head_parameters(self):
        return list(self.head.parameters())

    def head_module(self):
        return self.head


# --------------------------------------------------------------------------- naive references
class Naive(Backbone):
    """Repeat-last-value forecaster (parameter-free reference)."""

    def __init__(self, L: int, H: int, **_):
        super().__init__()
        self.L, self.H = L, H
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, channel_mask=None):
        return x[:, -1:].expand(-1, self.H, -1).clone()

    def head_parameters(self):
        return []

    def head_module(self):
        return nn.Identity()


REGISTRY = {"DLinear": DLinear, "PatchTST": PatchTST, "iTransformer": iTransformer, "Naive": Naive}


def build_backbone(name: str, L: int, H: int, **kwargs) -> Backbone:
    if name not in REGISTRY:
        raise KeyError(f"unknown backbone {name}; options {list(REGISTRY)}")
    return REGISTRY[name](L=L, H=H, **kwargs)


def _check_patchtst(L, patch_len, stride):
    if L < patch_len:
        raise ValueError(f"lookback {L} shorter than patch_len {patch_len}")
    return math.floor((L - patch_len) / stride) + 2
