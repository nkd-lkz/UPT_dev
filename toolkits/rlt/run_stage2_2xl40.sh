#!/usr/bin/env bash
# Launch the formal two-L40 ManiSkill RLT Stage2 job after Stage1 is released.
set -euo pipefail

mode=${1:---check}
if [[ "$mode" != --check && "$mode" != run ]]; then
    echo "Usage: bash toolkits/rlt/run_stage2_2xl40.sh [--check|run]" >&2
    exit 2
fi

RLINF_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$RLINF_ROOT"
source "$RLINF_ROOT/.venv/bin/activate"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RLT_STORAGE="${RLT_STORAGE:-/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill}"
export RLT_STAGE1_ACTOR="${RLT_STAGE1_ACTOR:-$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016/checkpoints/global_step_2000/actor}"
export RLT_DATASET_DIR="${RLT_DATASET_DIR:-$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint}"
run_stamp=$(date +%Y%m%d_%H%M%S)_$$
export RLT_STAGE2_RUN_DIR="${RLT_STAGE2_RUN_DIR:-$RLT_STORAGE/runs/stage2_baseline_2xl40/stage2_$run_stamp}"
export RLT_STAGE2_EXPERIMENT_NAME="${RLT_STAGE2_EXPERIMENT_NAME:-maniskill_rlt_stage2_2xl40_$run_stamp}"
export HF_HUB_CACHE="$RLT_STORAGE/cache/huggingface-hub"
export HF_LEROBOT_HOME="$RLT_STORAGE/datasets/lerobot"
export TORCH_HOME="$RLT_STORAGE/cache/torch"
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=c6522513-sustech
export WANDB_PROJECT=rlinf-rlt
export WANDB_MODE=online

export RLINF_VULKAN_PREFIX="${RLINF_VULKAN_PREFIX:-/home/luokz/.local/rlinf-vulkan}"
export PATH="$RLINF_VULKAN_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/lib64"
export SAPIEN_VULKAN_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357"
export VK_DRIVER_FILES="$RLINF_VULKAN_PREFIX/share/vulkan/icd.d/nvidia_headless_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
unset DISPLAY WAYLAND_DISPLAY VK_LAYER_PATH VK_INSTANCE_LAYERS
unset __NV_PRIME_RENDER_OFFLOAD RAY_ADDRESS RLINF_NODE_RANK CUDA_VISIBLE_DEVICES

python toolkits/rlt/check_stage2_2xl40.py
if [[ "$mode" == --check ]]; then
    exit 0
fi

for gpu in 0 1; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ ! "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || (( used > 1024 )); then
        echo "ERROR: GPU $gpu is busy or unreadable (used MiB: $used); refusing to start." >&2
        exit 1
    fi
done

python - <<'PY'
import wandb

api = wandb.Api()
if api.viewer.username != "c6522513" or api.default_entity != "c6522513-sustech":
    raise RuntimeError(
        f"Unexpected W&B identity: {api.viewer.username}/{api.default_entity}"
    )
print(f"W&B identity: {api.viewer.username}; entity: {api.default_entity}")
PY

ray_port=${RLT_STAGE2_RAY_PORT:-6396}
python - "$ray_port" <<'PY'
import shutil
import socket
import sys

port = int(sys.argv[1])
for candidate in (port, port + 1, port + 2):
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", candidate))
assert shutil.disk_usage("/dev/shm").free > 8 * 1024**3, "Need 8 GiB free /dev/shm"
PY

ray_temp_dir=$(mktemp -d /dev/shm/rlt-stage2-2xl40.XXXXXXXX)
export RAY_TMPDIR="$ray_temp_dir"
mkdir -p "$RLT_STAGE2_RUN_DIR/ray_spill"
exec > >(tee "$RLT_STAGE2_RUN_DIR/train.log") 2>&1
echo "Run directory: $RLT_STAGE2_RUN_DIR"
echo "Ray temp directory: $ray_temp_dir"
git rev-parse HEAD
git status --short

ray_head_pid=
train_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
        kill -TERM "$train_pid" 2>/dev/null || true
    fi
    if [[ -n "$ray_head_pid" ]] && kill -0 "$ray_head_pid" 2>/dev/null; then
        kill -TERM "$ray_head_pid" 2>/dev/null || true
        wait "$ray_head_pid" || true
    fi
    echo "Stage2 exit code: $status; logs: $RLT_STAGE2_RUN_DIR"
    echo "Ray diagnostics retained: $ray_temp_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ray start --head --block \
    --port="$ray_port" \
    --ray-client-server-port="$((ray_port + 1))" \
    --dashboard-host=127.0.0.1 --dashboard-port="$((ray_port + 2))" \
    --dashboard-agent-listen-port=0 \
    --min-worker-port=0 --max-worker-port=0 \
    --num-cpus=16 --num-gpus=2 \
    --object-store-memory=4294967296 \
    --temp-dir="$ray_temp_dir" \
    --object-spilling-directory="$RLT_STAGE2_RUN_DIR/ray_spill" \
    --include-dashboard=true --disable-usage-stats \
    > "$RLT_STAGE2_RUN_DIR/ray-head.log" 2>&1 &
ray_head_pid=$!

for ((attempt=0; attempt<120; attempt++)); do
    if ! kill -0 "$ray_head_pid" 2>/dev/null; then
        tail -n 80 "$RLT_STAGE2_RUN_DIR/ray-head.log"
        echo "ERROR: isolated Ray head exited before startup." >&2
        exit 1
    fi
    if [[ -s "$ray_temp_dir/ray_current_cluster" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -s "$ray_temp_dir/ray_current_cluster" ]]; then
    echo "ERROR: isolated Ray startup timed out." >&2
    exit 1
fi
export RAY_ADDRESS
RAY_ADDRESS=$(< "$ray_temp_dir/ray_current_cluster")
echo "Using isolated Ray at $RAY_ADDRESS"

train_cmd=(
    python examples/embodiment/train_embodied_agent.py
    --config-name maniskill_rlt_stage2_ac_mlp
    +experiment=rlt_baseline_2xl40
)
if [[ -n "${RLT_STAGE2_RESUME_DIR:-}" ]]; then
    train_cmd+=("runner.resume_dir=$RLT_STAGE2_RESUME_DIR")
fi
"${train_cmd[@]}" --cfg job --resolve > "$RLT_STAGE2_RUN_DIR/resolved-config.yaml"
"${train_cmd[@]}" &
train_pid=$!
wait "$train_pid"
train_pid=
