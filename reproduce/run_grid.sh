#!/usr/bin/env bash
# VINE, step 1 of 2: the online grid.
#
#   bash reproduce/run_grid.sh [jobs file] [parallel cells]
#
# One job = one cell (dataset, H, seed). For each cell, scripts/online_grid.py
#   1. pretrains the DLinear backbone on the train split (cached in code/results/_checkpoints),
#   2. tunes every method on the 5% validation segment (<= 24 configurations, same random-search seed for all),
#      and writes the chosen configurations to tuning.json BEFORE touching the test segment,
#   3. runs every method once on the test segment at each delay d in {H, 2H, 4H, 8H}.
# Outputs: code/results/online_grid/<dataset>/DLinear/H<H>/seed<k>/{tuning.json, test.csv, steps_<method>_d<d>.npz}
#
# The script is resumable: finished tuning entries and finished (method, delay) test rows are skipped, so it can
# simply be re-run after an interruption. Defaults: all 36 cells of the paper, 6 cells in parallel, 2 threads each.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$(cd "$(dirname "${1:-$ROOT/reproduce/jobs.txt}")" && pwd)/$(basename "${1:-jobs.txt}")"
PAR="${2:-6}"
PY="${PYTHON:-python}"
cd "$ROOT/code"
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TSL_DATA="${TSL_DATA:-$ROOT/code/data}"
echo "jobs: $JOBS ($(grep -c . "$JOBS") cells), $PAR in parallel, data in $TSL_DATA"
grep . "$JOBS" | xargs -P "$PAR" -L 1 bash -c \
  '"$0" scripts/online_grid.py --dataset "$1" --H "$2" --seed "$3" --backbone DLinear --threads 2 \
     > "logs/grid_DLinear_$1_H$2_s$3.log" 2>&1 && echo "done $1 H$2 seed$3" || echo "FAILED $1 H$2 seed$3 (see log)"' \
  "$PY"
echo "grid finished; check that every log in code/logs ends with a [done] line"
