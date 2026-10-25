#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ASSET_ROOT="${LIBERO_PRO_RLT_ROOT:-/mnt/hdd_toshiba_1/luokz/libero_pro_baseline_20261025}"
readonly PYTHON_DEPS="${ASSET_ROOT}/python_deps"
readonly SYSTEM_LIBS="${ASSET_ROOT}/system_libs"

export PYTHONPATH="${PYTHON_DEPS}:${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SYSTEM_LIBS}:${LD_LIBRARY_PATH:-}"
export LIBERO_TYPE=pro
export LIBERO_PERTURBATION="${LIBERO_PERTURBATION:-all}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$@"
