# tsl: leak-checked harness for online forecasting under truthful label delay

Experimental infrastructure of the VINE paper: two-channel streams in which the label block of origin s is released
at s+d (d >= H), an assertion-based leak checker, a future-perturbation leak test, per-origin and per-horizon errors,
latency, state bytes and peak memory. To reproduce the paper, follow `../docs/reproduce.md`; this file documents the
harness itself.

Pure PyTorch + NumPy/pandas/SciPy, YAML-configured, seed-controlled, device-agnostic (CPU laptop or CUDA server).
VINE is in `tsl/online/methods/vine.py` and `tsl/online/methods/stack.py`.

---

## Quick start on Linux (lab server): one command

```bash
git clone <this repository> && cd <repository>/code
bash server_setup.sh            # venv in code/.venv; CUDA torch if `nvidia-smi` works, else CPU torch;
                                # downloads the datasets; runs pytest; runs the smoke config
# options: ENV_KIND=conda | SKIP_DATA=1 | TSL_DATA=/scratch/$USER/tsl_data | TORCH_CUDA=cu126
source .venv/bin/activate
python run.py --config configs/online/etth2_delay_sweep.yaml          # a full experiment
```

Manual equivalent:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or .../whl/cpu ; match your driver
pip install -r requirements.txt
python scripts/download_tslib.py        # ETT, Weather, Exchange, ECL, Traffic (checksum-verified); see DATASETS.md
python -m pytest tests -q
python run.py --config configs/online/smoke_etth2.yaml
```

Multi-GPU servers: runs are single-process; launch independent configs/seeds on different GPUs, e.g.
`CUDA_VISIBLE_DEVICES=0 python run.py --config A.yaml --seeds 0 & CUDA_VISIBLE_DEVICES=1 python run.py --config A.yaml --seeds 1 &`.
`runtime.device: auto` picks CUDA when available; set `--set runtime.device=cpu` to force CPU.

## Windows laptop (CPU only)

PowerShell:

```powershell
cd "code"
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts\download_all.py
.\.venv\Scripts\python -m pytest tests -q
.\.venv\Scripts\python run.py --config configs\online\smoke_etth2.yaml
$env:TSL_THREADS = "4"   # optional: threads for pretraining/batched inference (default = physical cores)
```

Git Bash: the same commands with `.venv/Scripts/python` and forward slashes.

CPU notes: everything is float32; `runtime.threads` (default = physical cores) is used for pretraining and batched
backbone inference, `runtime.online_threads` (default 1) for the per-step loop, so reported latencies are single-core.
Per-step correctors (RLS, ELF-style) are vectorised NumPy over (horizon x channel). Wide streams (ECL, Traffic) never
materialise all backbone forecasts: they are computed in chunks of 1,024 origins under a 1.5 GB cache budget
(`runtime.forecast_cache_bytes`). Run long blocks overnight, plugged in, one heavy job at a time.

## Measured wall-clock (laptop, smoke configs; machine was ~50% loaded by other processes)

See the bottom of this file ("Smoke-test timings") for the measured numbers and the extrapolation to full runs.

---

## Layout

```
code/
  run.py                      # experiment runner: --config, --seeds, --set key=value, --out
  server_setup.sh             # Linux one-shot bootstrap
  configs/online/*.yaml       # (delay sweeps per dataset, prefix, provisional, leaky appendix, val tuning)
  scripts/                    # download_*.py (pinned checksums), significance.py (Holm-corrected tests)
  tsl/data/                   # registry + loaders (TSLib CSVs, OPSD), splits, windows
  tsl/models/                 # DLinear, PatchTST, iTransformer (+RevIN), offline pretraining with checkpoint cache
  tsl/online/                 # protocol, evaluator, forecast archive, leak test, provisional inputs, methods/
  tsl/stats.py                # Diebold-Mariano (HAC + HLN), Wilcoxon, Holm, moving-block bootstrap
  tests/                      # pytest tests (leakage proofs, protocol, metrics, VINE, faithful ports)
  results/                    # outputs (git-ignored)
```

## Online harness

**Protocol** (`tsl/online/protocol.py`). Inputs x and settled labels y are separate channels (identical in public
benchmarks; different in the provisional stress test). At time s the model forecasts y[s+1..s+H] from x[s-L+1..s].
* **P-strict** (`mode: strict`, headline): the label block of origin s is released at s+d, d >= H (default d = H).
  Update-time label limit: y[..now-d+H].
* **P-prefix** (`mode: prefix`): additionally exposes, for pending origins, the settled prefix y[s'+1..now-l] with lag
  l = d-H by default (`prefix_lag`).
* **P-leaky** (`mode: leaky`): FSNet/OneNet-style immediate full-label update, only with `allow_leaky: true`; results are
  tagged `_LEAKY` (appendix only).
* The evaluator calls `update(revealed)` then `predict(x)` at every origin; errors are computed by the evaluator;
  `drop_last=False` (every origin with a complete target is scored).

**Leak checks.** (1) `LeakChecker` asserts on every label handed to a method; (2) `HistoryView` guards every history read
(inputs up to now, labels up to the label limit) and returns copies; (3) `ForecastArchive.get(s)` refuses origins > now;
(4) `tsl.online.leaktest.future_perturbation_test` re-runs a method with all rows >= t0 replaced by NaN and asserts
bit-identical, finite forecasts before t0 - applicable to any method, including wrapped external code. The tests prove
all baselines pass under strict and prefix modes and that the leaky mode is detected.

**Interface for new methods** (`tsl/online/base.py`):

```python
class MyCorrector(OnlineMethod):
    def setup(self, ctx): ...                   # ctx.backbone (frozen), ctx.history, ctx.base_forecast(s), ctx.delay ...
    def predict(self, x, base_pred, t): ...     # (L,C) input window + frozen forecast -> (H,C)
    def update(self, info): ...                 # info.matured (labels released now), info.prefixes (P-prefix)
    def state_breakdown(self): ...              # {"name": array}; keys "pending_*"/"raw_*" are reported separately
```
Register with `@register("my_corrector")` in `tsl/online/methods/__init__.py`; use `{type: my_corrector, ...}` in YAML.

**Baselines**: `frozen`; `ogd` (delayed OGD, `scope: full|head`, mini-batch of the last k matured pairs, optional
prefix loss); `periodic` (retrain on all released rows every `period` steps or `period_frac` of the stream, warm or
from scratch); `rls` (per-(horizon, channel) RLS filter bank on frozen-backbone residuals with forgetting factor:
features = bias, backbone forecast, p lags of matured residuals); `elf` (**reimplementation** of ELF, Lee et al.,
ICML 2025 / PMLR v267 lee25ag / arXiv 2502.12920: shared complex Fourier-domain ridge forecaster, alpha = 0.9,
lambda = 20, refit every M = 200 matured pairs, Welford row scaling, fast/slow/merge exponential weighter with
eta = 0.5, B = 5; deviations listed in the module docstring; no official code was found).

**Backbones** (`tsl/models/backbones.py`): compact re-implementations of DLinear, PatchTST, iTransformer (written from
scratch following the papers and TSLib defaults; no TSLib code vendored; TSLib is MIT-licensed). Pretraining on the
train split with validation early stopping, checkpoints cached per (dataset, split, model config, L, H, seed) so all
correctors share the same frozen backbone. Note: DLinear's only parameters are its output linears, so `ogd` with
`scope: head` equals `scope: full` for DLinear.

**Metrics** (`summary.csv`, `metrics.json`, `steps.npz`): MSE/MAE (standardised units, train-only scaler), per-horizon
curves, cumulative-MSE curve, per-origin losses (for DM / block-bootstrap tests), median/mean/p95 update latency
(single core), predict latency, state bytes (learned / pending / raw), peak RSS (process-wide; set
`protocol.trace_memory: true` for a per-run tracemalloc peak).

## Running experiments

```bash
python run.py --config configs/online/etth2_delay_sweep.yaml --seeds 0 1 2
python run.py --config configs/online/prefix_etth2.yaml --set "horizons=[24]"
python scripts/significance.py results/etth2_delay_sweep --reference frozen        # DM per cell + Wilcoxon, Holm
```
Outputs: `results/<name>/summary.csv`, per-run JSON (config, environment, git commit, timings), `steps.npz`,
`timing.json`.

## Smoke-test timings (measured on a laptop early in the project, before VINE existed; numbers are NOT results)

Machine: i7-8550U (4C/8T), 16 GB, Windows 11, torch 2.15 CPU. **The CPU was ~50% busy with other processes during all
measurements**, so expect somewhat faster runs on an idle machine; re-time with the smoke configs before planning.
Online per-step loop single-threaded (`online_threads: 1`); pretraining used 4 threads.

| What | Wall-clock |
|---|---|
| Online smoke: ETTh2, DLinear (3 epochs), H=24, d in {24, 96}, 3,000 forecasts x 6 methods | 101 s total |
| Probe: ECL (321 ch), H=24, d=24, 1,000 forecasts | 52 s incl. 2-epoch pretraining |

Per-step costs (ETTh2, C=7, H=24; median update / mean predict): frozen ~2 us / 3 us; RLS bank ~175 us / 70 us;
ELF-style 25 us (median; refit every 200 -> mean ~290 us) / 215 us; delayed OGD on DLinear ~4 ms / 1 ms; periodic
retrain: mean ~0.4 ms per step amortised. ECL (C=321, H=24): RLS ~2.1 ms, ELF ~4.7 ms, head-OGD ~9.6 ms per step.
State: RLS 43 KiB (d=H) - 91 KiB (d=4H); ELF ~0.5-0.8 MiB on ETTh2 (dominated by the M=200 raw refit block) and
~32 MiB on ECL; OGD 58 KiB (params + Adam + buffer) for DLinear.

**Extrapolation to full runs on this laptop (rough, from the numbers above):**
* ETTh2 full online segment (~13k forecasts): ~3-5 min per (H, d, seed) cell for all six DLinear baselines; the
  `etth2_delay_sweep` grid (3 H x 3 d x 5 seeds = 45 cells) ~3-4 h. ETTm1/ETTm2 (~52k forecasts) ~4x that per cell.
* ECL (~19.7k forecasts): frozen + RLS + ELF + head-OGD ~5-6 min per cell at H=24 and ~3-4x at H=96 -> the ECL sweep
  (27 cells) ~6-9 h; Traffic (862 ch) ~2.5-3x ECL -> run overnight or restrict to H in {24, 48} on the laptop.
* Deep per-step baselines with PatchTST/iTransformer cost 5-20x DLinear per step: narrow streams only on the laptop.

Smoke numbers (ETTh2, first 3,000 test forecasts, DLinear trained 3 epochs, 1 seed; sanity check only):

| d | frozen | RLS bank | ELF-style | delayed OGD | periodic |
|---|---|---|---|---|---|
| 24 (= H) | 2.268 / 0.539 | 2.304 / 0.562 | 2.423 / 0.550 | 2.016 / 0.520 | 2.267 / 0.541 |
| 96 (= 4H) | 2.268 / 0.539 | 2.998 / 0.656 | 2.438 / 0.554 | 2.419 / 0.531 | 2.264 / 0.541 |

(MSE / MAE in standardised units.) On drift-heavy ETTh2 the cheap correctors do not beat the frozen backbone at this
tiny scale and RLS degrades at d = 4H; this is consistent with the audit's ETTh2 finding but says nothing yet - the
backbone was trained for 3 epochs and only 23% of the test segment was used.
