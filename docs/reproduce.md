# Reproducing the VINE paper

This guide reproduces every number, table and figure of the paper from raw data. It follows the order in
which the experiments were actually run.

## 1. What the experiments test

A forecaster issues an H-step forecast at every time step, but the true values for that forecast only arrive d ≥ H
steps later (*truthful delay*). Correctors that learn only from matured labels go stale as d grows. VINE also learns
from the forecaster's own **revisions**, the change in what it predicts for the same target time between consecutive
origins, which is available one step after every forecast.

The evaluation asks:

- Does stacking VINE on the strongest delay-aware corrector (DRiC-L8) lower its error, and is the gain due to
  revisions (against the same stack learning only from labels)?
- How does every method behave as the delay grows from d = H to d = 8H?
- When do revisions help? (Revision Predictability Index, nesting defect.)

Hypotheses H1–H7, the primary contrasts and the significance rule were fixed in [`PREREGISTRATION.md`](../PREREGISTRATION.md)
before the confirmation results existed. Every later change is in its amendment table with its date and reason.

## 2. Experimental design

| Item | Setting |
|---|---|
| Streams | ETTh1, ETTh2, ETTm1 (**development**: used while designing the method) and ETTm2, Weather, Exchange (**confirmation**: untouched until the design was frozen) |
| Split | FSNet split 20 : 5 : 75 (train : validation : test) of each stream; scaler fitted on training rows only |
| Backbone | DLinear, lookback L = 96, trained on the training split with early stopping on validation; frozen afterwards and shared by every method in a cell |
| Horizons | H ∈ {24, 96} |
| Delays | d ∈ {H, 2H, 4H, 8H}, i.e. point lag ℓ = d − H ∈ {0, H, 3H, 7H} |
| Seeds | 0, 1, 2 (backbone training seeds; the correctors are deterministic) |
| Tuning | On the 5% validation segment, run as an online stream under the same protocol; at most 24 configurations per method, the published default always included, the same random-search seed for every method; tuned per delay (d = 8H reuses the 4H choice) |
| Test | Every method starts online at the beginning of validation and is scored on the test origins only, once per (method, delay) |
| Methods | 27 methods in total; see [`code_map.md`](code_map.md). Co-primary: VINE-S (`vine_s`) and VINE-λ (`vine_lam`) |
| Statistics | Per-origin losses averaged over the three seeds; circular moving-block bootstrap (B = 1000, block ≥ max(2 × season, 2H)) for the 95% interval; Diebold–Mariano with HAC variance for the p-value; Holm over the whole primary table (192 tests). A contrast is significant when the Holm-adjusted DM p-value is below 0.05 **and** the bootstrap interval excludes 0 |

A *setting* is (stream, H, d): 6 × 2 × 4 = 48 settings. A *cell* is (stream, H, seed): 36 cells.

## 3. Setup

```bash
git clone <this repository> && cd <repository>/code
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or .../whl/cpu
pip install -r requirements-lock.txt                                         # exact versions of the reported runs
export TSL_DATA=$PWD/data                                                    # or any directory with ~1 GB free
python scripts/download_tslib.py                                             # ETT, Weather, Exchange (checksum-verified)
python -m pytest tests -q                                                    # no data needed
```

`download_tslib.py` fetches the whole TSLib collection; the paper uses ETTh1, ETTh2, ETTm1, ETTm2, Weather and Exchange. Dataset sources, sizes, checksums and licences are in [`code/DATASETS.md`](../code/DATASETS.md).

## 4. Quick check (one cell, under an hour)

Exchange at H = 24 is the cheapest cell:

```bash
cd code
python scripts/online_grid.py --dataset Exchange --H 24 --seed 0 --backbone DLinear --threads 2
```

This tunes all 27 methods, runs the test segment at four delays and writes
`results/online_grid/Exchange/DLinear/H24/seed0/`. It took about 50 minutes on the shared lab server. To try a few
methods only, add `--methods frozen persist_last dric_l8 vine_s_lo vine_s` (methods referenced by a stack, here
`dric_l8`, must be in the list).

## 5. Full grid

```bash
bash reproduce/run_grid.sh                       # all 36 cells, 6 in parallel; logs in code/logs/
bash reproduce/run_grid.sh my_jobs.txt 3         # a custom job list ("<dataset> <H> <seed>" per line), 3 in parallel
```

The grid is resumable: re-running it skips finished tuning entries and finished (method, delay) test rows.

Measured wall-clock per cell (seed 0, 2 threads per cell, on a shared server that was heavily loaded by other users'
jobs, so treat these as upper bounds):

| Stream | H = 24 | H = 96 |
|---|---|---|
| Exchange | 0.8 h | 1.2 h |
| ETTh2 | 1.9 h | 8.3 h |
| ETTh1 | 7.0 h | 5.1 h |
| ETTm2 | 9.8 h | 16.5 h |
| ETTm1 | 13.8 h | 34.9 h |
| Weather | 20.5 h | 47.6 h |

For the four ETT streams these times cover the first pass only; the methods added later (Section 8) ran
separately and add roughly half as much again. About 170 hours per seed in total for the first pass; with 6 cells in
parallel the whole grid takes several days. DSOF and Proceed run on
the GPU (one RTX 3090 was used); everything else runs on the CPU.

## 6. Analysis, tables and figures

```bash
bash reproduce/run_analysis.sh report            # writes report/ and paper/{tables,fig3_delay.pdf,fig4_rpi.pdf}
```

The five steps, in order:

1. **RPI replay** (`scripts/vine_rpi.py`, one process per cell). Replays the tuned methods on validation and computes
   the Revision Predictability Index of the frozen backbone (B), DRiC-L8 (F) and VINE-S (G), the exploratory online
   RPI and the nesting defect (hypotheses H3, H6).
2. **Pre-registered statistics** (`scripts/vine_report.py --boot 1000 --harm`). Means, every contrast, hypotheses
   H1, H2, H4, H5, H7. The `--harm` family (every method against frozen in every setting, 1,248 tests) makes this the
   slowest step: about 2.5 hours.
3. **Collect RPI** into `rpi.csv`.
4. **Exploratory fallback** (`scripts/vine_fallback.py`): a causal switch between VINE-S and last-value persistence,
   replayed exactly from the stored per-origin errors. Added after the confirmation results; it is not part of the
   pre-registered evaluation.
5. **Tables and figures** (`paper/make_results.py <report>`). See [`outputs.md`](outputs.md) for the
   mapping from files to paper tables and figures.

Use `contrasts_final.csv` (written in step 5) for any count of significant results; it applies the amended rule of
the pre-registration to every family.

## 7. Checking the main numbers

With the report directory from step 6, the paper's headline counts can be recomputed directly:

```python
import pandas as pd
c = pd.read_csv("report/contrasts_final.csv")
p = c[(c.family == "primary") & (c.a == "vine_s")]
for (conf, kind), g in p.groupby(["confirmation", "kind"]):
    print("confirmation" if conf else "development", kind,
          "wins", (g.rel < 0).sum(), "of", len(g),
          "| significant wins", (g.sig & (g.rel < 0)).sum(), "| significant losses", (g.sig & (g.rel > 0)).sum())
```

Expected output for the reported runs: on the 24 confirmation settings VINE-S beats DRiC-L8 in 23 (7 significant)
and its label-only control in 19 (none significant), with no significant losses. On the 24 development settings
the counts are 19 (6 significant) and 12 (1 significant win, 2 significant losses).

## 8. How the runs were actually carried out

The reported results were produced with the code in this repository. Before release, every source, config and test
file on the server was compared with the repository and found identical (ignoring line endings). Two changes were
made since: the harm-family fix in `vine_report.py` (commit "Fix vine_report.py: score every harm contrast"), which
affects only the `harm_rate` entry of `hypotheses.json`, not the paper's harm table; and the removal of the companion
SOMA paper's continual-learning code, which this repository does not need (commit "Split").

The grid was not run in a single pass. Following the pre-registration's amendments, methods were added in stages:

1. 2026-11-13: main grid with the methods of the original plan.
2. 2026-11-14: `vine_s`, `vine_s_lo`, `vine_s_pure`, `vine_lam_lm`, `vine_s_g9995`, `vine_s_g9999` added to cells that
   had already finished, with `--methods <names>`. Their tuning and test rows were added without touching the
   existing ones.
3. 2026-11-15: DSOF and Proceed added the same way (`--methods dsof proceed`), before any of their results existed.
4. 2026-11-16: the server disk filled up and the running jobs died. Damaged files were removed and the resumable
   scripts recomputed exactly the missing pieces.

Each method is tuned and tested independently with fixed seeds, so a single-pass run with the current code produces
the same configurations and errors. The deep methods trained on the GPU (DSOF, Proceed, delayed OGD, periodic
retraining) can differ in the last digits on other hardware.

## 9. What is exploratory or disclosed

- ETTh1, ETTh2 and ETTm1 influenced the design and are marked as development streams in every table.
- The significance rule (Holm on DM p-values plus a bootstrap interval excluding zero) was amended on 2026-11-16
  after the first analysis run, because the original rule could never reject. The amendment is disclosed in the
  pre-registration.
- The online RPI and the persistence fallback were added after seeing results and are labelled exploratory.
- Five pre-registered criteria fail as written (H1's cross-stream test, H3, H4, H5, H7); the paper reports them.

The methodology documents that the pre-registration refers to (`methodology/*.md`) accompany the paper and are not
part of this code release. `PREREGISTRATION.md` is kept unchanged, including its short section on the companion
SOMA paper, because its history is the record of when each decision was made.
