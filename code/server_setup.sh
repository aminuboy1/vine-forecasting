#!/usr/bin/env bash
# One-shot Linux bootstrap for the tsl harness (lab server or any Linux box).
#   bash server_setup.sh                 # venv in code/.venv, CUDA torch if nvidia-smi works, else CPU torch
#   ENV_KIND=conda bash server_setup.sh  # use conda instead of venv (env name: tsl)
#   SKIP_DATA=1 bash server_setup.sh     # do not download datasets
#   TSL_DATA=/scratch/$USER/tsl_data bash server_setup.sh   # put datasets elsewhere (recommended on servers)
#   TORCH_CUDA=cu126 bash server_setup.sh                    # force a CUDA wheel tag (default: auto)
# Steps: environment -> torch (CUDA/CPU) -> requirements -> datasets -> pytest -> smoke runs.
set -euo pipefail
cd "$(dirname "$0")"
CODE_DIR="$(pwd)"
PY="${PYTHON:-python3}"
ENV_KIND="${ENV_KIND:-venv}"

echo "== [1/6] Python environment ($ENV_KIND)"
if [[ "$ENV_KIND" == "conda" ]]; then
  if ! command -v conda >/dev/null; then echo "conda not found"; exit 1; fi
  eval "$(conda shell.bash hook)"
  conda env list | grep -qE '^tsl\s' || conda create -y -n tsl python=3.12
  conda activate tsl
  PYBIN="$(command -v python)"
else
  "$PY" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'
  [[ -d .venv ]] || "$PY" -m venv .venv
  PYBIN="$CODE_DIR/.venv/bin/python"
fi
"$PYBIN" -m pip install --upgrade pip -q

echo "== [2/6] PyTorch"
if "$PYBIN" -c 'import torch' 2>/dev/null; then
  echo "torch already installed: $("$PYBIN" -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"
elif command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  TAG="${TORCH_CUDA:-}"
  if [[ -z "$TAG" ]]; then
    CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    # pick the newest wheel tag not newer than the driver's CUDA version
    TAG=cu118
    for t in 121 124 126 128; do [[ -n "$CUDA_VER" && "$CUDA_VER" -ge "$t" ]] && TAG="cu$t"; done
  fi
  echo "GPU detected (driver CUDA ${CUDA_VER:-?}); installing torch wheel $TAG"
  "$PYBIN" -m pip install torch --index-url "https://download.pytorch.org/whl/$TAG" \
    || { echo "CUDA wheel $TAG failed; falling back to default PyPI torch"; "$PYBIN" -m pip install torch; }
else
  echo "no NVIDIA GPU found; installing CPU torch"
  "$PYBIN" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "== [3/6] Requirements"
"$PYBIN" -m pip install -r requirements.txt -q
"$PYBIN" - <<'PY'
import torch, numpy, pandas, scipy
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count(), "| numpy", numpy.__version__, "| pandas", pandas.__version__)
PY

echo "== [4/6] Datasets (TSL_DATA=${TSL_DATA:-$CODE_DIR/data})"
if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  "$PYBIN" scripts/download_all.py \
    || echo "WARNING: some downloads failed (see above). re-run later."
fi

echo "== [5/6] Unit tests (leakage proofs, harness)"
"$PYBIN" -m pytest tests -q

echo "== [6/6] Smoke runs (tiny; numbers are NOT results)"
"$PYBIN" run.py --config configs/online/smoke_etth2.yaml --quiet
echo "Done. Activate with: source $CODE_DIR/.venv/bin/activate (or: conda activate tsl)"
