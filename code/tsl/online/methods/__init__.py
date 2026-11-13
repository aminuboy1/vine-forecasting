"""Registry of online methods. Register new methods with `@register("name")` or by adding to METHODS."""
from __future__ import annotations

from .audit_rls import AuditRLSBank
from .dric import DRiC
from .elf import ELFStyle
from .epoc import EPOC
from .rls import RLSBank
from .simple import DelayedOGD, Frozen, PeriodicRetrain, Persistence
from .vine import MCLS, VINE, MCLSPerRow, SafeMix, VINELambda

METHODS = {
    "frozen": Frozen,
    "persistence": Persistence,  # kind: last|seasonal (backbone-free naive)
    "ogd": DelayedOGD,          # cfg: scope: full|head
    "ogd_full": lambda **kw: DelayedOGD(scope="full", **kw),
    "ogd_head": lambda **kw: DelayedOGD(scope="head", **kw),
    "periodic": PeriodicRetrain,
    "rls": RLSBank,
    "elf": ELFStyle,
    "rls_audit": AuditRLSBank,  # faithful port of the 2026 audit's RLS-Bank (median of passthrough + 6 lambdas)
    "dric": DRiC,               # faithful port; cfg: variant l8|l4|g|both; phases single|all; update_stride
    "dric_l8": lambda **kw: DRiC(variant="l8", **kw),
    "dric_g": lambda **kw: DRiC(variant="g", **kw),
    "epoc": EPOC,               # faithful port; cfg: phases single|all; offgrid phase|latest; update_stride
    "vine": VINE,               # paper 2 method (lam=1); lam<1 blends in matured targets; target=teacher: DSOF-style
    "vine_g": lambda **kw: VINE(pool=True, **kw),
    "vine_lam": VINELambda,     # TD(lambda)-style blend of revision targets and per-row matured labels
    "mc_ls": MCLS,              # matured-only least squares on VINE's features (lam = 0)
    "mc_ls_pr": MCLSPerRow,     # label-only LS with per-row maturity at the point lag l = d - H (EFC-style)
    "safe_mix": SafeMix,        # delayed EWA + fixed share over {frozen, MC-LS, VINE fast, VINE slow}
}


def register(name: str):
    def deco(cls):
        METHODS[name] = cls
        return cls
    return deco


def build_method(spec: dict):
    """spec = {"type": "rls", "name": optional display name, **kwargs}."""
    spec = dict(spec)
    typ = spec.pop("type")
    display = spec.pop("name", None)
    if typ not in METHODS:
        raise KeyError(f"unknown online method {typ}; options {list(METHODS)}")
    m = METHODS[typ](**spec)
    if display:
        m.name = display
    return m
