"""Build the Paper 2 result tables (tables/*.tex) and Fig. 3 (fig3_delay.pdf) from the output of
code/scripts/vine_report.py. Nothing here selects or filters results: every complete setting is shown.

    python make_results.py <report_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DS_ORDER = ["ETTm2", "Weather", "Exchange", "ETTh1", "ETTh2", "ETTm1"]      # confirmation first
DEV = {"ETTh1", "ETTh2", "ETTm1"}
NAMES = {"frozen": "Frozen", "persist_last": "Persistence (last)", "persist_seasonal": "Persistence (seasonal)",
         "periodic": "Periodic retraining", "ogd_head": "OGD (head)", "rls_audit": "RLS bank",
         "elf": "ELF", "epoc": "EPOC", "dric_g": "DRiC-G", "dric_l8": "DRiC-L8", "dric_l8_single": "DRiC-L8 (single grid)",
         "dsof": "DSOF", "proceed": "Proceed", "mc_ls": "MC-LS", "mc_ls_pr": "MC-LS-PR", "rev_regress": "Naive rev.\\ regression",
         "safe_mix": "SAFE-MIX", "safe_mix_lo": "SAFE-MIX-LO", "vine": "\\VINE{} (base)", "vine_teacher": "Teacher target",
         "vine_lam": "\\VINE{}-$\\lambda$", "vine_s": "\\VINE{}-S", "vine_s_lo": "\\VINE{}-S-LO",
         "vine_s_pure": "\\VINE{}-S-pure", "vine_lam_lm": "\\VINE{}-$\\lambda$ (no base)",
         "vine_s_g9995": "\\VINE{}-S, $\\gamma=0.9995$", "vine_s_g9999": "\\VINE{}-S, $\\gamma=0.9999$"}
MAIN_ROWS = [["frozen", "persist_seasonal", "periodic", "ogd_head"],
             ["rls_audit", "elf", "epoc", "dric_g", "dric_l8", "dsof", "proceed"],
             ["mc_ls_pr", "vine_s_lo", "safe_mix"],
             ["vine_lam", "vine_s"]]


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def finalize(con: pd.DataFrame) -> pd.DataFrame:
    """Significance rule of the 2026-11-16 amendment (same as vine_report.py): Holm over each family on the DM-HAC
    p-values, and the bootstrap 95% CI must exclude zero. Idempotent, so older report outputs can be re-scored."""
    con = con.copy()
    con["p_holm"] = np.nan
    for fam, idx in con.groupby("family").groups.items():
        con.loc[idx, "p_holm"] = holm(con.loc[idx, "p_dm"].values)
    con["sig"] = (con.p_holm < 0.05) & ((con.rel_hi < 0) | (con.rel_lo > 0))
    return con


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table_main(means: pd.DataFrame, con: pd.DataFrame, lag: int) -> str:
    """MSE at d = (lag+1) H for every stream and H; bold best, underline second; markers for Holm-significant
    primary contrasts of the co-primary methods (dagger: vs DRiC-L8, ddagger: vs label-only control)."""
    m = means[(means.n_seeds == 3) & (means.delay == (lag + 1) * means.H)]
    cols = [(ds, H) for ds in DS_ORDER for H in (24, 96) if ((m.dataset == ds) & (m.H == H)).any()]
    rows = [r for g in MAIN_ROWS for r in g]
    val = {(r.method, r.dataset, r.H): r.mse for r in m.itertuples()}
    best = {}
    for c in cols:
        v = sorted(val[(k, *c)] for k in rows if (k, *c) in val)
        best[c] = v[:2]
    sig = {}
    if len(con):
        p = con[(con.family == "primary") & (con.lag == con.H * lag) & con.sig & (con.rel < 0)]
        for r in p.itertuples():
            sig.setdefault((r.a, r.dataset, r.H), "")
            sig[(r.a, r.dataset, r.H)] += "\\textsuperscript{\\dag}" if r.kind == "vs_dric_l8" else "\\textsuperscript{\\ddag}"
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{ds}{'$^*$' if ds in DEV else ''}}}" for ds in dict.fromkeys(c[0] for c in cols))
    sub = " & ".join(str(H) for _, H in cols)
    out = [f"\\begin{{tabular}}{{l{'c' * len(cols)}}}", "\\toprule", f"Stream & {head} \\\\",
           f"$H$ & {sub} \\\\", "\\midrule"]
    for gi, g in enumerate(MAIN_ROWS):
        for k in g:
            cells = []
            for c in cols:
                x = val.get((k, *c))
                s = fmt(x)
                if x is not None and best[c] and x == best[c][0]:
                    s = f"\\textbf{{{s}}}"
                elif x is not None and len(best[c]) > 1 and x == best[c][1]:
                    s = f"\\underline{{{s}}}"
                cells.append(s + sig.get((k, *c), ""))
            out.append(f"{NAMES[k]} & " + " & ".join(cells) + " \\\\")
        if gi < len(MAIN_ROWS) - 1:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_primary(con: pd.DataFrame) -> str:
    """Relative MSE change (%) of each co-primary method against DRiC-L8 and against its control, by delay."""
    p = con[con.family == "primary"]
    if p.empty:
        return "% no primary contrasts yet"
    lags = sorted(p.lag.div(p.H).astype(int).unique())
    out = ["\\begin{tabular}{ll" + "c" * (2 * len(lags)) + "}", "\\toprule",
           " & & \\multicolumn{" + str(len(lags)) + "}{c}{vs DRiC-L8} & \\multicolumn{" + str(len(lags)) +
           "}{c}{vs label-only control} \\\\",
           "Stream & $H$ & " + " & ".join(f"$\\ell={l}H$" if l else "$\\ell=0$" for l in lags) * 1 + " & " +
           " & ".join(f"$\\ell={l}H$" if l else "$\\ell=0$" for l in lags) + " \\\\"]
    for m, title in [("vine_s", "\\VINE{}-S"), ("vine_lam", "\\VINE{}-$\\lambda$")]:
        out += ["\\midrule", f"\\multicolumn{{{2 + 2 * len(lags)}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for ds in DS_ORDER:
            for H in (24, 96):
                q = p[(p.a == m) & (p.dataset == ds) & (p.H == H)]
                if q.empty:
                    continue
                cells = []
                for kind in ["vs_dric_l8", "vs_control"]:
                    for l in lags:
                        r = q[(q.kind == kind) & (q.lag == l * H)]
                        if r.empty:
                            cells.append("--"); continue
                        r = r.iloc[0]
                        s = f"{100 * r.rel:+.1f}"
                        if r.sig:
                            s = f"\\textbf{{{s}}}"
                        cells.append(s)
                out.append(f"{ds}{'$^*$' if ds in DEV else ''} & {H} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_summary(con: pd.DataFrame, family: str, kinds: list[tuple[str, str, str]]) -> str:
    """One row per contrast: settings, wins, Holm-significant wins/losses, median and range of the relative change."""
    out = ["\\begin{tabular}{lccccc}", "\\toprule",
           "Contrast & Settings & Wins & Sig.\\ wins & Sig.\\ losses & Median (range), \\% \\\\", "\\midrule"]
    for a, kind, label in kinds:
        q = con[(con.family == family) & (con.a == a) & (con.kind == kind)]
        if q.empty:
            continue
        sw = int(((q.rel < 0) & q.sig).sum())
        sl = int(((q.rel > 0) & q.sig).sum())
        out.append(f"{label} & {len(q)} & {int((q.rel < 0).sum())} & {sw} & {sl} & "
                   f"{100 * q.rel.median():+.1f} ({100 * q.rel.min():+.1f}, {100 * q.rel.max():+.1f}) \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def table_cost(cost: pd.DataFrame) -> str:
    rows = ["frozen", "ogd_head", "rls_audit", "elf", "epoc", "dric_g", "dric_l8", "dsof", "proceed", "mc_ls_pr",
            "safe_mix", "vine_lam", "vine_s"]
    c = cost[cost.dataset.isin(["ETTh1", "ETTh2", "ETTm1", "ETTm2"])]
    g = c.groupby(["method", "H"]).agg(med=("step_us_med", "median"), p95=("step_us_p95", "median"),
                                       st=("state_learned_per_channel", "median"),
                                       pend=("state_pending", "median")).reset_index()
    out = ["\\begin{tabular}{lcccccc}", "\\toprule",
           " & \\multicolumn{3}{c}{$H=24$} & \\multicolumn{3}{c}{$H=96$} \\\\",
           "Method & Median & p95 & State & Median & p95 & State \\\\",
           " & ($\\mu$s) & ($\\mu$s) & (kB/ch.) & ($\\mu$s) & ($\\mu$s) & (kB/ch.) \\\\", "\\midrule"]
    for k in rows:
        cells = []
        for H in (24, 96):
            r = g[(g.method == k) & (g.H == H)]
            if r.empty:
                cells += ["--"] * 3; continue
            r = r.iloc[0]
            cells += [f"{r.med:,.0f}", f"{r.p95:,.0f}", fmt(r.st / 1e3, 1)]
        out.append(f"{NAMES[k]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


CAP = 20.0     # Fig. 3 y-axis cap (%); series above it are named in the panel


def fig_delay(means: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.5})
    show = [("frozen", "Frozen", "#000000", ":"), ("rls_audit", "RLS bank", "#999999", "-"),
            ("dric_l8", "DRiC-L8", "#E69F00", "-"), ("mc_ls_pr", "MC-LS-PR", "#56B4E9", "--"),
            ("dsof", "DSOF", "#CC79A7", "-."), ("proceed", "Proceed", "#009E73", "-."),
            ("vine_lam", "VINE-$\\lambda$", "#0072B2", "-"), ("vine_s", "VINE-S", "#D55E00", "-")]
    m = means[means.n_seeds == 3]
    panels = [(ds, H) for ds in DS_ORDER for H in (24, 96) if ((m.dataset == ds) & (m.H == H)).any()]
    nc = 4
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.16, 1.55 * nr + 0.35), squeeze=False)
    for ax, (ds, H) in zip(axes.flat, panels):
        sub = m[(m.dataset == ds) & (m.H == H)]
        fz = sub[sub.method == "frozen"].set_index("delay").mse
        ys, off = [], []
        for k, lab, col, ls in show:
            s = sub[sub.method == k].set_index("delay").mse
            if s.empty:
                continue
            x = s.index.values / H
            y = 100 * (s / fz.reindex(s.index) - 1)
            ax.plot(x, y, ls=ls, color=col, lw=0.9, marker="o", ms=1.8, label=lab)
            ys.append(y.values)
            if y.max() > CAP:
                off.append(f"{lab} up to {y.max():+.0f}%")
        allv = np.concatenate(ys)
        ax.set_ylim(allv.min() - 0.08 * (min(allv.max(), CAP) - allv.min() + 1),
                    min(allv.max(), CAP) + 0.08 * (min(allv.max(), CAP) - allv.min() + 1))
        if off:
            ax.text(0.97, 0.96, "; ".join(off) + " (off scale)", transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.5, color="#555555")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
        ax.axhline(0, color="#000000", lw=0.4)
        ax.set_title(f"{ds}{' (dev.)' if ds in DEV else ''}, $H={H}$", fontsize=7, pad=2)
        ax.tick_params(length=2, width=0.4, pad=1)
        ax.grid(alpha=0.25, lw=0.3)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE vs frozen (%)")
    for ax in axes[-1]:
        ax.set_xlabel("$d/H$")
    hl = {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l_, h_)
    order = [lab for _, lab, _, _ in show if lab in hl]
    h, l = [hl[x] for x in order], order
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / (1.55 * nr + 0.35)), w_pad=0.6, h_pad=0.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)


def main():
    rep = Path(sys.argv[1])
    means = pd.read_csv(rep / "means.csv")
    con = pd.read_csv(rep / "contrasts.csv")
    con = finalize(con)
    con.to_csv(rep / "contrasts_final.csv", index=False)
    cost = pd.read_csv(rep / "cost.csv")
    t = HERE / "tables"
    t.mkdir(exist_ok=True)
    (t / "main_mse_l1.tex").write_text(table_main(means, con, lag=1))
    (t / "main_mse_l0.tex").write_text(table_main(means, con, lag=0))
    (t / "main_mse_l3.tex").write_text(table_main(means, con, lag=3))
    (t / "main_mse_l7.tex").write_text(table_main(means, con, lag=7))
    (t / "primary.tex").write_text(table_primary(con))
    (t / "external.tex").write_text(table_summary(con, "secondary", [
        (m, f"vs_{b}", f"{NAMES[m]} vs {NAMES[b]}") for m in ["vine_s", "vine_lam"]
        for b in ["rls_audit", "elf", "epoc", "dric_g", "dsof", "proceed"]]))
    (t / "ablation.tex").write_text(table_summary(con, "ablation", [
        ("vine_s", "vs_vine_s_lo", "\\VINE{}-S vs \\VINE{}-S-LO (revision learning)"),
        ("vine_s", "vs_vine_s_pure", "\\VINE{}-S vs \\VINE{}-S-pure ($\\lambda=1$)"),
        ("vine_s", "vs_vine_lam_lm", "\\VINE{}-S vs top layer without base"),
        ("vine_s", "vs_vine_lam", "\\VINE{}-S vs \\VINE{}-$\\lambda$ (stacking)"),
        ("vine_teacher", "teacher_vs_vine", "Teacher target vs fixed point (base \\VINE{})")]))
    (t / "memory.tex").write_text(table_summary(con, "ablation", [
        ("vine_s", "vs_vine_s_g9995", "$\\gamma=0.9998$ vs $0.9995$"),
        ("vine_s", "vs_vine_s_g9999", "$\\gamma=0.9998$ vs $0.9999$")]))
    (t / "cost.tex").write_text(table_cost(cost))
    hm = con[con.family == "harm"]
    if len(hm):
        g = hm.assign(harm=(hm.rel > 0) & hm.sig, help=(hm.rel < 0) & hm.sig).groupby("a")
        hr = {a: {"n": len(x), "harm": x.harm.mean(), "help": x.help.mean()} for a, x in g}
        rows = [k for g in MAIN_ROWS for k in g if k in hr]
        out = ["\\begin{tabular}{lccc}", "\\toprule", "Method & Settings & Harm (\\%) & Help (\\%) \\\\", "\\midrule"]
        out += [f"{NAMES[k]} & {hr[k]['n']} & {100 * hr[k]['harm']:.0f} & {100 * hr[k]['help']:.0f} \\\\" for k in rows]
        (t / "harm.tex").write_text("\n".join(out + ["\\bottomrule", "\\end{tabular}"]))
    fig_delay(means, HERE / "fig3_delay.pdf")
    print("tables and figure written")


if __name__ == "__main__":
    main()
