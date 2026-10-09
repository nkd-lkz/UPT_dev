#!/usr/bin/env bash
# Isolated AlphaBrain RLT/Pi0.5 research entry; leaves existing RLT_a runs intact.
set -euo pipefail
RLT_PI05_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RLT_PI05_EXTERNAL=${RLT_EXTERNAL_ROOT:-"$RLT_PI05_ROOT/../third_party"}
RLT_PI05_PYTHON=${RLT_PI05_VENV:-"$RLT_PI05_ROOT/../.venv-libero-acceptance"}/bin/python
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${TMPDIR:-/dev/shm}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RLT_PI05_EXTERNAL/alphabrain-deps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RLT_PI05_EXTERNAL/alphabrain-egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$RLT_PI05_EXTERNAL/alphabrain-egl/nvidia.json"
cd "$RLT_PI05_ROOT"
exec "$RLT_PI05_PYTHON" -u -m toolkits.rlt.pi05_baseline "$@"
