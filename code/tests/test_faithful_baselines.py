"""Tests for the faithful ports of the audit RLS bank, DRiC and EPOC (tsl/online/methods/{audit_rls,dric,epoc}.py).

1. causality: `future_perturbation_test` (NaN and noise futures, two channels, d = H and d = 2H) under P-strict;
2. sanity: each method removes a constant bias and helps on a periodic toy;
3. retained-state numbers equal those of the original code / papers (C = 7, H = 24);
4. optional numerical equivalence with the ORIGINAL code: set FAITHFUL_EXT_DIR to a directory holding clones
   `audit/` (Xodios/TRAINING-ON-THE-FUTURE-..., 4940d02), `dric/` (keiotakmin/dric, 7ce0a33) and `epoc/`
   (keiotakmin/endpoint-preserving-residual-correction, f31dcfe). Their code is then run in a separate
   `python -I` process; our forecasts must be BIT-IDENTICAL (tolerance 0) to theirs cast to float32
   (the audit runs in float32 end to end; DRiC/EPOC run in float64 and our evaluator stores float32 forecasts,
   so the float64 difference is <= 1 float32 ulp of the output, ~1.2e-7 on this stream).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from tsl.data.registry import synthetic_dataset
from tsl.models import build_backbone
from tsl.online import OnlineEvaluator, future_perturbation_test, make_provisional
from tsl.online.methods.audit_rls import AuditRLSBank
from tsl.online.methods.dric import DRiC
from tsl.online.methods.epoc import EPOC

L, H = 48, 8          # DRiC needs H >= 8 DCT components and L % H == 0
T, FIRST = 700, 300

FACTORIES = {
    "rls_audit": lambda: AuditRLSBank(),
    "rls_audit_f64_noskip": lambda: AuditRLSBank(dtype="float64", skip_nonfinite=False),
    "dric_l8": lambda: DRiC(variant="l8"),
    "dric_l4": lambda: DRiC(variant="l4"),
    "dric_g": lambda: DRiC(variant="g"),
    "dric_l8_stride1": lambda: DRiC(variant="l8", update_stride=1),
    "dric_l8_phased": lambda: DRiC(variant="l8", phases="all"),
    "dric_g_phased": lambda: DRiC(variant="g", phases="all"),
    "epoc_phase": lambda: EPOC(),
    "epoc_latest": lambda: EPOC(offgrid="latest"),
    "epoc_stride1": lambda: EPOC(update_stride=1),
    "epoc_phased": lambda: EPOC(phases="all"),
}


def _backbone():
    torch.manual_seed(0)
    return build_backbone("DLinear", L, H)


# ----------------------------------------------------------------------------- 1. causality
@pytest.mark.parametrize("key", list(FACTORIES))
@pytest.mark.parametrize("dmult", [1, 2])
@pytest.mark.parametrize("fill", ["nan", "noise"])
def test_future_perturbation_strict(key, dmult, fill):
    y = synthetic_dataset(T=T, C=3, seed=1).values
    x = make_provisional(y, seed=7)          # two channels: inputs != labels
    ok, rep = future_perturbation_test(FACTORIES[key], _backbone(), y, FIRST, L, H, delay=dmult * H,
                                       mode="strict", x=x, fill=fill)
    assert ok, rep


@pytest.mark.parametrize("key", ["rls_audit", "dric_l8", "dric_g", "epoc_phase"])
def test_future_perturbation_prefix_mode_is_also_clean(key):
    y = synthetic_dataset(T=T, C=3, seed=2).values
    ok, rep = future_perturbation_test(FACTORIES[key], _backbone(), y, FIRST, L, H, delay=2 * H, mode="prefix")
    assert ok, rep


# ----------------------------------------------------------------------------- 2. sanity
def _toy(kind, C=3, N=1200, Hs=24, Ls=96, seed=0):
    rng = np.random.default_rng(seed)
    first = 200
    n = first + N + Hs + 1
    t = np.arange(n)
    y = (np.sin(2 * np.pi * t / 24)[:, None] * rng.uniform(.5, 1.5, C) + 0.1 * rng.standard_normal((n, C)))
    y = y.astype(np.float32)
    tgt = np.stack([y[s + 1:s + 1 + Hs] for s in range(first, first + N)])
    if kind == "bias":     # frozen forecast = truth + noise + constant bias
        base = tgt + 0.1 * rng.standard_normal(tgt.shape) + 1.0
    else:                  # periodic toy: frozen forecast misses the daily cycle entirely (predicts the level)
        base = np.zeros_like(tgt) + 0.05 * rng.standard_normal(tgt.shape)
    return y, base.astype(np.float32), first, N, Hs, Ls


@pytest.mark.parametrize("kind", ["bias", "periodic"])
@pytest.mark.parametrize("key", ["rls_audit", "dric_l8", "dric_l4", "dric_g", "epoc_phase", "dric_l8_phased",
                                 "dric_g_phased", "epoc_phased"])
def test_removes_bias_and_helps_on_periodic_toy(kind, key):
    y, base, first, N, Hs, Ls = _toy(kind)
    torch.manual_seed(0)
    ev = OnlineEvaluator(y, first, first + N - 1, Ls, Hs, delay=Hs, keep_preds=True)
    res = ev.run(FACTORIES[key](), build_backbone("DLinear", Ls, Hs), base_preds=base)
    tgt = np.stack([y[s + 1:s + 1 + Hs] for s in range(first, first + N)])
    idx = np.arange(N)
    # single-grid DRiC/EPOC train on one phase only: judge them on their own (grid) origins; the per-origin
    # versions (audit bank, phased) on all origins. Frozen MSE: bias toy ~1.01, periodic toy ~0.37.
    single_grid = key in ("dric_l8", "dric_l4", "dric_g", "epoc_phase")
    sel = (idx >= N // 2) & ((idx % Hs == 0) if single_grid else True)
    mse_frozen = np.mean((base[sel] - tgt[sel]) ** 2)
    mse = np.mean((res.preds[sel] - tgt[sel]) ** 2)
    assert mse < 0.2 * mse_frozen, (key, kind, mse, mse_frozen)        # measured: 80-99% reductions
    if single_grid:                                                     # still helps on average over all origins
        allo = idx >= N // 2
        assert np.mean((res.preds[allo] - tgt[allo]) ** 2) < 0.8 * np.mean((base[allo] - tgt[allo]) ** 2)


def test_audit_season_longer_than_lookback_reads_guarded_history():
    """season > L goes through ctx.history.inputs and must give the same forecasts as an L >= season window."""
    y, base, first, N, Hs, _ = _toy("periodic", N=300)
    outs = []
    for Ls in (24, 96):
        torch.manual_seed(0)
        ev = OnlineEvaluator(y, first, first + N - 1, Ls, Hs, delay=Hs, keep_preds=True)
        outs.append(ev.run(AuditRLSBank(season=48), build_backbone("DLinear", Ls, Hs), base_preds=base).preds)
    np.testing.assert_array_equal(outs[0], outs[1])


def test_skip_nonfinite_is_noop_on_finite_data():
    y, base, first, N, Hs, Ls = _toy("bias", N=400)
    pairs = [(AuditRLSBank, {}), (DRiC, {"variant": "l8"}), (DRiC, {"variant": "g"}), (EPOC, {})]
    for cls, kw in pairs:
        p = []
        for skip in (True, False):
            torch.manual_seed(0)
            ev = OnlineEvaluator(y, first, first + N - 1, Ls, Hs, delay=Hs, keep_preds=True)
            p.append(ev.run(cls(skip_nonfinite=skip, **kw), build_backbone("DLinear", Ls, Hs), base_preds=base).preds)
        np.testing.assert_array_equal(p[0], p[1])


def test_nan_label_does_not_poison_state():
    y, base, first, N, Hs, Ls = _toy("bias", N=400)
    y = y.copy()
    y[first + 100, 1] = np.nan                       # one missing settled value
    for m in (AuditRLSBank(), DRiC(variant="l8"), DRiC(variant="g"), EPOC()):
        torch.manual_seed(0)
        ev = OnlineEvaluator(y, first, first + N - 1, Ls, Hs, delay=Hs, keep_preds=True)
        p = ev.run(m, build_backbone("DLinear", Ls, Hs), base_preds=base).preds
        assert np.isfinite(p[200:]).all(), m.name


# ----------------------------------------------------------------------------- 3. state numbers of the originals
def test_retained_state_matches_original_accounting():
    y, base, first, N, Hs, Ls = _toy("bias", C=7, N=100)
    want = {"l8": 19824, "l4": 7504, "g": 12096}       # their test formula / DRiC paper (C = 7, H = 24, L = 96)
    for v, b in want.items():
        m = DRiC(variant=v)
        OnlineEvaluator(y, first, first + N - 1, Ls, Hs).run(m, build_backbone("DLinear", Ls, Hs), base_preds=base)
        assert m.their_state_bytes() == b
    m = EPOC()
    OnlineEvaluator(y, first, first + N - 1, Ls, Hs).run(m, build_backbone("DLinear", Ls, Hs), base_preds=base)
    assert m.their_state_bytes() == 4528                # EPOC paper / their test, C = 7
    m = AuditRLSBank()
    OnlineEvaluator(y, first, first + N - 1, Ls, Hs).run(m, build_backbone("DLinear", Ls, Hs), base_preds=base)
    assert m.w.nbytes + m.P.nbytes == 7 * 7 * 24 * 4 * 4 + 6 * 7 * 24 * 16 * 4   # 83,328 B (float32)


# ----------------------------------------------------------------------------- 4. equivalence with the originals
THEIRS = textwrap.dedent(r'''
    import json, re, sys, time, importlib.util
    sys.path.append(sys.argv[4])                     # numpy's site dir only (python -I drops user site)
    import numpy as np
    ext, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    D = dict(np.load(inp))
    z, base, first, L, H, d, P = D["z"], D["base"], int(D["first"]), int(D["L"]), int(D["H"]), int(D["d"]), int(D["P"])
    N = base.shape[0]; out = {}; mb = d // H
    nb = json.load(open(f"{ext}/audit/code_supplement/notebooks/rls_delay_tta_full_run.ipynb", encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"np": np, "time": time}
    exec(next(c for c in cells if c.lstrip().startswith("class Adapter")), ns)
    c12 = next(c for c in cells if "def build_feats" in c)
    exec(re.search(r"def build_feats\(f, featset\):.*?(?=\n# =====|\Z)", c12, re.S).group(0), ns)
    origins = np.arange(first, first + N) + 1
    swH = np.lib.stride_tricks.sliding_window_view(z, H, axis=0)
    swP = np.lib.stride_tricks.sliding_window_view(z, P, axis=0)
    f = {"preds": base.transpose(0, 2, 1).copy(),
         "trues": np.stack([z[o:o + H] for o in origins]).transpose(0, 2, 1).copy(),
         "last": z[origins - 1].copy(), "mday": swP[origins - P].mean(-1).astype(np.float32),
         "snv": swH[origins - P].copy()}
    a = ns["RLSBankMed"](z.shape[1], H, 4, (1.0, 0.9999, 0.9995, 0.999, 0.995, 0.99), 10.0, 10.0)
    X = ns["build_feats"](f, "feat"); pr = np.empty_like(f["preds"])
    for i in range(N):                               # their replay() loop
        if i - d >= 0: a.update(X[i - d], f["trues"][i - d])
        pr[i] = a.predict(X[i])
    out["audit"] = pr.transpose(0, 2, 1)
    spec = importlib.util.spec_from_file_location("ridge_corrector", f"{ext}/dric/ridge_corrector.py")
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    grid = np.arange(0, N, H)
    gb = base[grid].astype(float); gt = np.stack([z[first + k + 1:first + k + 1 + H] for k in grid]).astype(float)
    gw = np.stack([z[first + k - L + 1:first + k + 1] for k in grid]).astype(float)
    for name, cfg in [("l8", rc.Config(local=8)), ("g", rc.Config(local=0, global_=True))]:
        cor = rc.Corrector(z.shape[1], H, L, cfg); pend = {}; iss = np.empty_like(gb)
        for k in range(len(grid)):
            if k - mb >= 0: j = k - mb; cor.update(gb[j], gt[j], *pend.pop(j))
            iss[k], c_, p_ = cor.issue(gb[k], gw[k]); pend[k] = (c_, p_)
        out["dric_" + name] = iss
        if mb == 1: out["dric_" + name + "_replay"] = rc.replay(gb, gt, gw, cfg)[0]
    sys.path.append(f"{ext}/epoc/src")
    import endpoint_review_validation as erv
    from channel_control import RollingControl
    tr = erv.Tracker(H, z.shape[1], erv.Config()); g = RollingControl(z.shape[1], "global", "blend", 32); pend = {}
    iss = np.empty_like(gb)
    for k in range(len(grid)):
        if k - mb >= 0:
            j = k - mb; v_, c_ = pend.pop(j); g.update(c_, gt[j] - gb[j]); tr.update(gt[j] - gb[j], v_)
        corr, tok = tr.issue(gb[k]); raw = gb[k] + corr; corr = raw - gb[k]
        iss[k] = gb[k] + g.alpha()[0] * corr; pend[k] = (tok, corr)
    out["epoc"] = iss
    if mb == 1: out["epoc_evaluate"] = erv.evaluate(gb, gt, erv.Config(), predictions=True)["predictions"]
    out["grid"] = grid
    np.savez(outp, **out)
''')

EXT = os.environ.get("FAITHFUL_EXT_DIR")


@pytest.mark.skipif(not EXT, reason="set FAITHFUL_EXT_DIR to clones of the original repos to run the equivalence check")
@pytest.mark.parametrize("dmult,shift", [(1, 0), (2, 0), (1, 5)])
def test_bit_identical_to_original_code(tmp_path, dmult, shift):
    """shift > 0: their code is run on the block grid starting `shift` origins later, which must equal phase
    `shift` of our `phases="all"` versions (and the audit bank on the shifted stream)."""
    rng = np.random.default_rng(0)
    C, Ls, Hs, N, first = 7, 96, 24, 720, 200
    n = first + N + Hs + 1
    t = np.arange(n)
    z = (np.sin(2 * np.pi * t / 24)[:, None] * rng.uniform(.5, 2, C) + 0.3 * np.sin(2 * np.pi * t / 168)[:, None]
         + 0.2 * rng.standard_normal((n, C)) + np.where(t > first + N // 2, .8, 0.)[:, None]).astype(np.float32)
    z[first + 10 * Hs + 5:first + 10 * Hs + 17, 2] = -162.          # missing-value code: exercises DRiC's guard
    base = np.stack([0.8 * z[s + 1 + np.arange(Hs) - 24] + 0.1 * rng.standard_normal((Hs, C))
                     for s in range(first, first + N)]).astype(np.float32)
    d = dmult * Hs
    np.savez(tmp_path / "s.npz", z=z, base=base[shift:], first=first + shift, L=Ls, H=Hs, d=d, P=24)
    (tmp_path / "theirs.py").write_text(THEIRS)
    r = subprocess.run([sys.executable, "-I", str(tmp_path / "theirs.py"), EXT, str(tmp_path / "s.npz"),
                        str(tmp_path / "o.npz"), os.path.dirname(os.path.dirname(np.__file__))],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    th = dict(np.load(tmp_path / "o.npz"))
    grid = th["grid"]

    def ours(m):
        torch.manual_seed(0)
        ev = OnlineEvaluator(z, first, first + N - 1, Ls, Hs, delay=d, keep_preds=True)
        return ev.run(m, build_backbone("DLinear", Ls, Hs), base_preds=base).preds

    if shift == 0:
        np.testing.assert_array_equal(ours(AuditRLSBank(season=24)), th["audit"])
        for v in ("l8", "g"):
            p = ours(DRiC(variant=v))[grid]
            np.testing.assert_array_equal(p, th["dric_" + v].astype(np.float32))
            if dmult == 1:
                np.testing.assert_array_equal(p, th["dric_" + v + "_replay"].astype(np.float32))
        for off in ("phase", "latest"):
            p = ours(EPOC(offgrid=off))[grid]
            np.testing.assert_array_equal(p, th["epoc"].astype(np.float32))
            if dmult == 1:
                np.testing.assert_array_equal(p, th["epoc_evaluate"].astype(np.float32))
    g2 = grid + shift
    for v in ("l8", "g"):
        np.testing.assert_array_equal(ours(DRiC(variant=v, phases="all"))[g2], th["dric_" + v].astype(np.float32))
    np.testing.assert_array_equal(ours(EPOC(phases="all"))[g2], th["epoc"].astype(np.float32))
