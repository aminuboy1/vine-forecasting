"""Registry of online methods. Register new methods with `@register("name")` or by adding to METHODS."""
from __future__ import annotations

from .elf import ELFStyle
from .rls import RLSBank
from .simple import DelayedOGD, Frozen, PeriodicRetrain
from .vine import VINE

METHODS = {
    "frozen": Frozen,
    "ogd": DelayedOGD,          # cfg: scope: full|head
    "ogd_full": lambda **kw: DelayedOGD(scope="full", **kw),
    "ogd_head": lambda **kw: DelayedOGD(scope="head", **kw),
    "periodic": PeriodicRetrain,
    "rls": RLSBank,
    "elf": ELFStyle,
    "vine": VINE,               # stub (raises until implemented)
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
