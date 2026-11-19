#!/usr/bin/env bash
# VINE, step 2 of 2: diagnostics, statistics, tables and figures (after reproduce/run_grid.sh).
#
#   bash reproduce/run_analysis.sh [report dir] [parallel RPI replays]
#
#   1. scripts/vine_rpi.py      Revision Predictability Index and nesting defect per cell (hypotheses H3, H6)
#                               -> code/results/online_grid/.../rpi.json
#   2. scripts/vine_report.py   pre-registered statistics over all settings (block bootstrap CI, Diebold-Mariano,
#                               Holm), hypotheses H1, H2, H4, H5, H7 and the harm rate -> <report>/{means.csv,
#                               contrasts.csv, hypotheses.json, cost.csv, coverage.csv}
#   3. vine_rpi.py collect      flatten every rpi.json -> <report>/rpi.csv
#   4. scripts/vine_fallback.py exploratory persistence fallback (not pre-registered) -> <report>/fallback.csv
#   5. paper/make_results.py  LaTeX tables (paper/tables/*.tex), Fig. 3 and Fig. 4,
#                               and <report>/{contrasts_final.csv, rpi_cells.csv, rpi_hypotheses.json}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REP="${1:-$ROOT/report}"
mkdir -p "$REP"
REP="$(cd "$REP" && pwd)"
PAR="${2:-3}"
PY="${PYTHON:-python}"
cd "$ROOT/code"
mkdir -p logs/rpi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TSL_DATA="${TSL_DATA:-$ROOT/code/data}"

echo "== [1/5] RPI replays"
: > logs/rpi/jobs.txt
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do for H in 24 96; do for s in 0 1 2; do
  c=results/online_grid/$ds/DLinear/H$H/seed$s
  [ -f "$c/tuning.json" ] || { echo "skip $ds H$H seed$s: no grid output"; continue; }
  [ -f "$c/rpi.json" ] && grep -q '"delay": '$((8 * H)) "$c/rpi.json" && continue      # already complete
  echo "$ds $H $s" >> logs/rpi/jobs.txt
done; done; done
xargs -P "$PAR" -L 1 bash -c \
  '"$0" scripts/vine_rpi.py --dataset "$1" --H "$2" --seed "$3" --device cpu --threads 1 > "logs/rpi/$1_H$2_s$3.log" 2>&1' \
  "$PY" < logs/rpi/jobs.txt

echo "== [2/5] pre-registered statistics (about 2.5 h with --harm on 14 cores)"
OMP_NUM_THREADS=4 "$PY" scripts/vine_report.py --out "$REP" --boot 1000 --harm

echo "== [3/5] collect RPI"
"$PY" scripts/vine_rpi.py collect results/online_grid "$REP/rpi.csv"

echo "== [4/5] exploratory persistence fallback"
"$PY" scripts/vine_fallback.py --root results/online_grid --out "$REP/fallback.csv"

echo "== [5/5] tables and figures"
"$PY" "$ROOT/paper/make_results.py" "$REP"
echo "done: statistics in $REP, tables and figures in paper/"
