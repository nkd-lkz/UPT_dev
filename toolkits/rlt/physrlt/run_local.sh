#!/usr/bin/env bash
# Host profile for the isolated LIBERO-Pro PhysRLT development worktree.
set -euo pipefail
PHYSRLT_CODE=/home/luokz/rlinf_rlt/UPT_physrlt_liberopro_dev
PHYSRLT_PYTHON=/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python
export PYTHONPATH=/mnt/hdd_toshiba_1/luokz/libero_pro_baseline_20261025/python_deps:"$PHYSRLT_CODE"
export LD_LIBRARY_PATH=/mnt/hdd_toshiba_1/luokz/libero_pro_baseline_20261025/system_libs:/usr/local/cuda-12.1/lib64
export LIBERO_TYPE=pro
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
cd "$PHYSRLT_CODE"
exec "$PHYSRLT_PYTHON" -u -m toolkits.rlt.physrlt.campaign "$@"
