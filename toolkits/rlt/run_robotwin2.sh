#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ASSET_ROOT="${ROBOTWIN2_RLT_ROOT:-/mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025}"
readonly ROBOTWIN_ROOT="${ASSET_ROOT}/repos/RoboTwin"
readonly EGL_LIB="${ASSET_ROOT}/system_libs/root/usr/lib/x86_64-linux-gnu"
readonly PYTHON_DEPS="${ASSET_ROOT}/python_deps"

# RoboTwin pins mplib 0.2.1 while the shared ManiSkill environment pins 0.1.1.
# Keep the benchmark-specific wheel in an overlay instead of mutating the
# shared venv used by other GPU jobs.
export PYTHONPATH="${PYTHON_DEPS}:${REPO_ROOT}:/home/luokz/rlinf_rlt/third_party/AlphaBrain:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${EGL_LIB}:${LD_LIBRARY_PATH:-}"
export ROBOTWIN_PATH="${ROBOTWIN_ROOT}"
export ROBOTWIN_PI05_CHECKPOINT="${ASSET_ROOT}/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
export REPO_PATH="${REPO_ROOT}"
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec "$@"
