#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
source "${RLINF_VENV:-/home/luokz/rlinf_rlt/UPT_dev/.venv}/bin/activate"
export PYTHONPATH="$root:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export RLT_STORAGE="${RLT_STORAGE:-/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill}"
export HF_HUB_CACHE="$RLT_STORAGE/cache/huggingface-hub"
export HF_LEROBOT_HOME="$RLT_STORAGE/datasets/lerobot"
export TORCH_HOME="$RLT_STORAGE/cache/torch"
export RLINF_VULKAN_PREFIX=/home/luokz/.local/rlinf-vulkan
export LD_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/lib64"
export SAPIEN_VULKAN_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357"
export VK_DRIVER_FILES="$RLINF_VULKAN_PREFIX/share/vulkan/icd.d/nvidia_headless_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
unset DISPLAY WAYLAND_DISPLAY RAY_ADDRESS
dataset="$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
mode=${1:---help}
case "$mode" in
  run|check) module=toolkits.rlt_vr.online_server ;;
  smoke) module=toolkits.rlt_vr.smoke_online ;;
  *) echo 'Usage: bash run_rlt_vr_online_gpu2.sh run|check|smoke [extra CLI arguments]'; exit 0 ;;
esac
shift
export RLT_STAGE1_ACTOR="${RLT_STAGE1_ACTOR:-$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016/checkpoints/global_step_2000/actor}"
output="$RLT_STORAGE/runs/vr_online/$(date +%Y%m%d_%H%M%S)_$$"
if [[ "$mode" == check ]]; then
  set -- --check "$@"
fi
python -m "$module" --stage1 "$RLT_STAGE1_ACTOR" --dataset "$dataset" --output "$output" "$@"
