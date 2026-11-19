# Output files

Every result is written under `code/results/` (git-ignored). This page lists what each file contains, so the
numbers in the paper can be traced back to raw per-origin errors.

## Per cell: `code/results/online_grid/<dataset>/DLinear/H<H>/seed<k>/`

| File | Content |
|---|---|
| `tuning.json` | For every method: all evaluated configurations with their validation MSE, the chosen configuration per tuning delay (`best_by_dfac`: d = H, 2H, 4H; d = 8H uses the 4H choice) and a hash of the search space. Written before the test segment is run. |
| `test.csv` | One row per (method, delay): `mse`, `mae`, `n` (scored origins), latency (`upd_us_med`, `step_us_med`, `step_us_p95`, ...), learned and pending state in bytes, wall-clock, the tuning delay used and the full resolved configuration (`config`, JSON). |
| `steps_<method>_d<d>.npz` | Per-origin errors on the test segment: `origins` (n,), `se` (n, H) squared error averaged over channels, `ae` (n, H) absolute error. All statistics are computed from these. |
| `env.json` | Package versions, hardware, git commit and backbone pretraining information. |
| `rpi.json` | Written by `vine_rpi.py`: RPI of B (frozen), F (DRiC-L8), G (VINE-S), the online RPI, the nesting defect and the chosen ridge penalties, per delay. |

Backbone checkpoints are cached in `code/results/_checkpoints/`.

## Report directory (`reproduce/run_analysis.sh [report dir]`)

| File | Written by | Content |
|---|---|---|
| `means.csv` | `vine_report.py` | Mean MSE/MAE per (dataset, H, delay, method) over the three seeds, with the number of seeds. |
| `contrasts.csv` | `vine_report.py` | Every pairwise test: `rel` = MSE(a)/MSE(b) − 1, bootstrap 95% interval (`rel_lo`, `rel_hi`), DM p-value, family (`primary`, `secondary`, `ablation`, `harm`) and kind. |
| `contrasts_final.csv` | `make_results.py` | `contrasts.csv` re-scored with the amended rule: significant when the Holm-adjusted DM p-value (Holm within each family) is below 0.05 **and** the bootstrap interval excludes 0. Use this file, not the `sig` column of `contrasts.csv`. |
| `hypotheses.json` | `vine_report.py` | H1 (win share at ℓ ≥ H, per-stream medians, Wilcoxon), H2, H4, H5 slopes, H7 cost, primary summary. |
| `cost.csv`, `coverage.csv` | `vine_report.py` | Latency and state per method; which (dataset, H, seed, method, delay) results exist. |
| `rpi.csv` | `vine_rpi.py collect` | One row per (dataset, H, seed, delay) with all RPI and nesting-defect values. |
| `rpi_cells.csv`, `rpi_hypotheses.json` | `make_results.py` | Seed-averaged RPI cells joined with the test gains; H3 correlations and H6. |
| `fallback.csv` | `vine_fallback.py` | Exploratory VINE-S + persistence switch against VINE-S, persistence and DRiC-L8, with the share of origin–horizon pairs that used persistence. |

## Paper tables and figures (`paper/`)

`make_results.py` writes `tables/*.tex` and `fig3_delay.pdf`, `fig4_rpi.pdf`:

| File | Paper element |
|---|---|
| `tables/main_mse_l1.tex` | Main table: mean test MSE at point lag ℓ = H (d = 2H). `main_mse_l0`, `l3`, `l7` are the other delays (supplement). |
| `tables/primary.tex` | Primary contrasts of VINE-S and VINE-λ against DRiC-L8 and against their label-only controls. |
| `tables/external.tex` | Co-primary methods against every external corrector. |
| `tables/harm.tex` | Harm and help rates against the frozen backbone. |
| `tables/ablation.tex` | Ablations of VINE-S. |
| `tables/rpi.tex`, `fig4_rpi.pdf` | Revision predictability, nesting defect and their relation to the gains. |
| `tables/memory.tex` | Memory sensitivity (γ = 0.9995, 0.9999). |
| `tables/cost.tex` | Latency and learned state. |
| `fig3_delay.pdf` | MSE relative to frozen against d/H for every stream and horizon. |
