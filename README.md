# VINE: vintage-consistent error correction under truthful label delay

Code, pre-registration and reproduction scripts for the VINE paper.

The label of an H-step forecast arrives at least H steps after the forecast is issued. Correctors that learn only
from these delayed (matured) labels go stale as the delay grows. A deployed forecaster, however, produces a
label-free training signal one step after every forecast: its own **revision**, the change in what it predicts for
the same target time between consecutive origins. An h-step error decomposes exactly into h − 1 revisions and one
one-step error, so only the one-step part of the correction needs delayed labels.

VINE is a closed-form online corrector that blends a revision-bootstrapped least-squares fixed point with matured
labels and can be stacked on any causal corrector. The paper evaluates it under a pre-registered, leak-tested
protocol with delays up to 8H on six streams. The grid runs 27 methods: VINE's variants, controls and ablations,
faithful ports of the strongest delay-aware correctors, leak-free ports of DSOF and Proceed, and reference methods.

## Start here

| If you want to | Read |
|---|---|
| Reproduce the paper step by step | [`docs/reproduce.md`](docs/reproduce.md) |
| Find where a method is implemented | [`docs/code_map.md`](docs/code_map.md) |
| Understand a result file | [`docs/outputs.md`](docs/outputs.md) |
| See what was fixed before the results, and every later amendment | [`PREREGISTRATION.md`](PREREGISTRATION.md) |
| Get the datasets | [`code/DATASETS.md`](code/DATASETS.md) |
| Add your own corrector to the harness | [`code/README.md`](code/README.md) |

## Quick start

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu      # or a CUDA build
pip install -r requirements-lock.txt
python -m pytest tests -q                    # no data needed
export TSL_DATA=$PWD/data && python scripts/download_tslib.py
python scripts/online_grid.py --dataset Exchange --H 24 --seed 0     # one cell: tune and test all 27 methods
```

Full pipeline:

```bash
bash reproduce/run_grid.sh                   # 36 cells: 6 streams x 2 horizons x 3 seeds
bash reproduce/run_analysis.sh report        # RPI, pre-registered statistics, tables and figures
```

## Repository layout

```
PREREGISTRATION.md        protocol, hypotheses and dated amendments (its git history records when each was made)
reproduce/                launch scripts for the full pipeline
docs/                     reproduction guide, code map, output formats
code/
  run.py                  generic YAML experiment runner
  scripts/                online_grid.py (the grid), vine_rpi.py, vine_report.py, vine_fallback.py (analysis),
                          download_*.py (datasets, pinned checksums)
  configs/online/         YAML configs for single experiments
  tsl/                    the harness: data, backbones, release protocol, leak checks, methods
  tests/                  unit and leak tests for every method
  requirements-lock.txt   exact package versions of the reported runs
paper/                    make_results.py: builds the paper's tables and figures from the analysis output
```

## Environment of the reported runs

Linux, Python 3.10.12, torch 2.6.0 (CUDA 12.4), NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, on an Intel Core i9-10940X
(14 cores, 125 GB RAM) with an NVIDIA RTX 3090, shared with other users' jobs. The closed-form correctors run on the
CPU; the backbone and the deep adapters on the GPU.

## Authors

Aminu Bashir Suleiman, Haowei Ran, Hamza Ibrahim, Sun Meijun and Wang Zheng. The code is maintained by
Hamza Ibrahim and Aminu Bashir Suleiman.
