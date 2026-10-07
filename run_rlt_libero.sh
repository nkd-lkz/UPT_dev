#!/usr/bin/env bash
# Run the pinned LIBERO reproduction in the shared Python environment.
set -euo pipefail

RLT_LIBERO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RLT_LIBERO_PYTHON=${RLINF_VENV:-"$RLT_LIBERO_ROOT/../UPT_dev/.venv"}/bin/python
RLT_EXTERNAL_ROOT=${RLT_EXTERNAL_ROOT:-"$RLT_LIBERO_ROOT/../third_party"}
RLT_ALPHABRAIN_SOURCE=${RLT_ALPHABRAIN_SOURCE:-"$RLT_EXTERNAL_ROOT/AlphaBrain"}
RLT_STORAGE=${RLT_STORAGE:-/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill}
RLT_LIBERO_ASSETS=${RLT_LIBERO_ASSETS:-"$RLT_STORAGE/research/libero_reproduction_20261006/assets"}

if [[ ! -x "$RLT_LIBERO_PYTHON" ]]; then
    echo "Missing Python: $RLT_LIBERO_PYTHON; set RLINF_VENV." >&2
    exit 2
fi

export TMPDIR=${TMPDIR:-/dev/shm}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
if [[ -d "$RLT_EXTERNAL_ROOT/alphabrain-deps" ]]; then
    export PYTHONPATH="$RLT_EXTERNAL_ROOT/alphabrain-deps${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -f "$RLT_EXTERNAL_ROOT/alphabrain-egl/nvidia.json" ]]; then
    export LD_LIBRARY_PATH="$RLT_EXTERNAL_ROOT/alphabrain-egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$RLT_EXTERNAL_ROOT/alphabrain-egl/nvidia.json"
fi

if [[ $# -eq 0 ]]; then
    set -- check
fi
if [[ "$1" == train ]]; then
    RLT_ALPHABRAIN_SOURCE=${RLT_ALPHABRAIN_TRAIN_SOURCE:-"$RLT_EXTERNAL_ROOT/AlphaBrain_rlt_runtime"}
fi
cd "$RLT_LIBERO_ROOT"
RLT_LIBERO_MODULE=toolkits.rlt.libero_reproduction
if [[ "$1" == audit ]]; then
    RLT_LIBERO_MODULE=toolkits.rlt.libero_audit
    shift
elif [[ "$1" == environment-audit ]]; then
    RLT_LIBERO_MODULE=toolkits.rlt.libero_environment_audit
    shift
fi
exec "$RLT_LIBERO_PYTHON" -u -m "$RLT_LIBERO_MODULE" "$@" \
    --source "$RLT_ALPHABRAIN_SOURCE" --storage "$RLT_LIBERO_ASSETS"
