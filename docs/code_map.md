# Code map

Where every part of the paper lives. Paths are relative to `code/` unless stated otherwise.

## Infrastructure

| Path | What it does |
|---|---|
| `run.py` | Generic runner for a YAML config: `python run.py --config <yaml> --seeds 0 1 2 [--set key=value]`. |
| `tsl/data/registry.py`, `tsl/data/loaders.py` | Dataset registry and loaders (TSLib CSVs; OPSD as an optional extra). `TSL_DATA` sets the data directory. |
| `tsl/data/splits.py` | Chronological splits. The paper uses the `fsnet` split 20 : 5 : 75 (train : validation : test). The z-score scaler is fitted on training rows only. |
| `tsl/data/synth_online.py` | Synthetic online streams used by the unit tests. |
| `tsl/models/backbones.py`, `tsl/models/layers.py` | DLinear (used in the paper), PatchTST and iTransformer, written from the papers with TSLib defaults. |
| `tsl/models/train.py` | Backbone pretraining with validation early stopping; checkpoints cached per (dataset, split, model, L, H, seed) so that all methods in a cell share one frozen backbone. |
| `tsl/stats.py` | Diebold–Mariano test with Newey–West (HAC) variance, Wilcoxon, Holm, circular moving-block bootstrap. |
| `tsl/utils.py` | Config loading and overrides, seeding, thread and device setup, environment capture written next to every result. |

## Protocol and leak checks (`tsl/online/`)

| Path | What it does |
|---|---|
| `protocol.py` | Two-channel stream. The label block of origin s, y[s+1..s+H], is released at s+d with d ≥ H. `LeakChecker` asserts on every label handed to a method; `HistoryView` guards every read of past inputs and labels. |
| `evaluator.py` | The online loop: at every origin, `update(revealed)` then `predict(x)`; errors are computed by the evaluator, never by the method. Records per-origin errors, latency and state size. |
| `leaktest.py` | Future-perturbation test: re-runs a method with every row from t0 on replaced by NaN and requires bit-identical forecasts before t0. Applied to every method, including the ported external code. |
| `base.py` | `OnlineMethod` interface (`setup`, `predict`, `update`, `state_breakdown`). |
| `experiment.py` | Config-driven online experiments (the `run.py` path). |
| `provisional.py` | Provisional-input stress test (inputs that differ from the settled labels). |

## Methods (`tsl/online/methods/`)

| Name in the paper | Grid name (`scripts/online_grid.py`) | Implementation |
|---|---|---|
| **VINE-S** (co-primary) | `vine_s` | `stack.py` (`Stack`) with a `vine_lam` top layer on the tuned `dric_l8` base |
| **VINE-λ** (co-primary) | `vine_lam` | `vine.py` (`VINELambda`) |
| VINE (base, λ = 1) | `vine` | `vine.py` (`VINE`) |
| VINE-S-LO (label-only control of VINE-S) | `vine_s_lo` | `stack.py` with an `mc_ls_pr` top layer |
| VINE-S-pure, VINE-λ without base | `vine_s_pure`, `vine_lam_lm` | ablations, same files |
| Memory sensitivity | `vine_s_g9995`, `vine_s_g9999` | tuned `vine_s` with γ changed |
| Teacher target (DSOF-style) | `vine_teacher` | `vine.py` (`VINE`, `target="teacher"`) |
| MC-LS, MC-LS-PR (label-only, same features) | `mc_ls`, `mc_ls_pr` | `vine.py` (`MCLS`, `MCLSPerRow`) |
| Naive revision regression | `rev_regress` | `vine.py` (`MCLSPerRow` without level/EWMA features) |
| SAFE-MIX, SAFE-MIX-LO | `safe_mix`, `safe_mix_lo` | `vine.py` (`SafeMix`) |
| RLS bank (2026 audit) | `rls_audit` | `audit_rls.py` (faithful port) |
| DRiC-L8, DRiC-G | `dric_l8`, `dric_g`, `dric_l8_single` | `dric.py` (faithful port) |
| EPOC | `epoc` | `epoc.py` (faithful port) |
| ELF | `elf` | `elf.py` (reimplementation; deviations in the docstring) |
| DSOF (ICLR 2025) | `dsof` | `dsof.py` (port of the official code to the truthful protocol; deviations in the docstring) |
| Proceed (KDD 2025) | `proceed` | `proceed.py` (port of the official code; deviations in the docstring) |
| Frozen, persistence | `frozen`, `persist_last`, `persist_seasonal` | `simple.py` |
| Delayed OGD (head), periodic retraining | `ogd_head`, `periodic` | `simple.py` |

The registry that maps a YAML `type:` to a class is `METHODS` in `tsl/online/methods/__init__.py`. The search space
of every method in the grid is the `METHODS` table at the top of `scripts/online_grid.py`.

## Scripts

| Script | Role |
|---|---|
| `scripts/online_grid.py` | One cell (dataset, H, seed): pretrain, tune every method on validation, run the test segment at d ∈ {H, 2H, 4H, 8H}. |
| `scripts/vine_rpi.py` | Revision Predictability Index (RPI) of the frozen backbone, DRiC-L8 and VINE-S on validation, plus the nesting-defect diagnostic. The `collect` subcommand flattens all cells into one CSV. |
| `scripts/vine_report.py` | Pre-registered statistics over all settings: means, primary/secondary/ablation/harm contrasts, hypotheses H1, H2, H4, H5, H7, cost. |
| `scripts/vine_fallback.py` | Exploratory (not pre-registered) causal switch between VINE-S and persistence, replayed from stored errors. |
| `scripts/significance.py` | Generic DM/Wilcoxon/Holm tests for `run.py` experiments. |
| `scripts/download_tslib.py`, `download_opsd.py`, `download_all.py` | Dataset downloads with pinned checksums. |
| `../paper/make_results.py` | Builds the LaTeX tables and Fig. 3/Fig. 4 from the report directory. |

## Tests (`tests/`)

The tests need no downloaded data (data-dependent tests are skipped without it): `python -m pytest tests -q`.

| File | Covers |
|---|---|
| `test_online_leakage.py` | The release protocol, the leak checker, and the future-perturbation test on every method; detection of the leaky protocol. |
| `test_vine.py` | VINE's fixed point, the propagation discount, constant-bias propagation, stacking. |
| `test_faithful_baselines.py` | The RLS bank, DRiC and EPOC ports against reference behaviour. |
| `test_dsof_proceed.py` | Leak tests and protocol checks for the DSOF and Proceed ports. |
| `test_data_and_methods.py` | Loaders, splits, scalers, backbones. |
