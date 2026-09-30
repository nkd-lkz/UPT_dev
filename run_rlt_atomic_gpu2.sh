#!/usr/bin/env bash
# Start an isolated, bounded Stage 2 smoke job; never attach to Stage 1 Ray.
set -euo pipefail

# Parse the whole job before executing it, so later source edits cannot corrupt it.
main() {

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check && "$1" != --probe && "$1" != --run ) ]]; then
    echo "Usage: bash run_rlt_atomic_gpu2.sh [--check (default)|--probe|--run]" >&2
    exit 2
fi

RLINF_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$RLINF_ROOT"
source "${RLINF_VENV:-/home/luokz/rlinf_rlt/UPT_dev/.venv}/bin/activate"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RLT_STORAGE="${RLT_STORAGE:-/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill}"
export RLT_STAGE1_ACTOR="${RLT_STAGE1_ACTOR:-$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016/checkpoints/global_step_2000/actor}"
export RLT_DATASET_DIR="${RLT_DATASET_DIR:-$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint}"
export HF_HUB_CACHE="$RLT_STORAGE/cache/huggingface-hub"
export HF_LEROBOT_HOME="$RLT_STORAGE/datasets/lerobot"
export TORCH_HOME="$RLT_STORAGE/cache/torch"
export TOKENIZERS_PARALLELISM=false
# Logging is local TensorBoard; no W&B credentials are needed for this smoke.
export WANDB_MODE=offline

export RLINF_VULKAN_PREFIX="${RLINF_VULKAN_PREFIX:-/home/luokz/.local/rlinf-vulkan}"
export PATH="$RLINF_VULKAN_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/lib64"
export SAPIEN_VULKAN_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357"
export VK_DRIVER_FILES="$RLINF_VULKAN_PREFIX/share/vulkan/icd.d/nvidia_headless_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Known GPU 2 address for read-only --check; query it again before real launch.
export RLT_SMOKE_RENDER_DEVICE=pci:0000:e1:00.0
unset DISPLAY WAYLAND_DISPLAY VK_LAYER_PATH VK_INSTANCE_LAYERS __NV_PRIME_RENDER_OFFLOAD
unset RAY_ADDRESS RLINF_NODE_RANK

profile="${RLT_ATOMIC_PROFILE:-smoke}"
case "$profile" in
    smoke) config_name=maniskill_rlt_stage2_atomic_gpu2; default_steps=2 ;;
    pilot) config_name=maniskill_rlt_stage2_atomic_pilot_gpu2; default_steps=20 ;;
    residual) config_name=maniskill_rlt_stage2_residual_pilot_gpu2; default_steps=20 ;;
    *) echo "ERROR: RLT_ATOMIC_PROFILE must be smoke, pilot or residual." >&2; exit 2 ;;
esac
run_id="atomic_${profile}_gpu2_$(date +%Y%m%d_%H%M%S)_$$"
export RLT_SMOKE_RUN_DIR="${RLT_ATOMIC_OUTPUT_DIR:-$RLT_STORAGE/runs/atomic_smoke/$run_id}"
export RLT_SMOKE_RAY_PORT="${RLT_SMOKE_RAY_PORT:-6512}"

# Read-only preflight: no Ray, CUDA allocation, checkpoint deserialization or mkdir.
export RLT_ATOMIC_STEPS="${RLT_ATOMIC_STEPS:-$default_steps}"
python -m toolkits.rlt.check_atomic_gpu2 --config-name "$config_name"
if [[ "${1:---check}" == --check ]]; then
    exit 0
fi
if [[ -e "$RLT_SMOKE_RUN_DIR" ]]; then
    echo "ERROR: refusing to reuse an existing run directory: $RLT_SMOKE_RUN_DIR" >&2
    exit 1
fi

# Cooperating launches share a lock; unrelated jobs are protected by busy checks.
exec 9> /tmp/rlt-atomic-gpu2.lock
flock -n 9 || { echo 'Another atomic GPU 2 job holds the lock.' >&2; exit 1; }
# Reject a busy GPU or occupied port before creating any run or starting Ray.
gpu_used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ ! "$gpu_used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || (( gpu_used > 1024 )); then
    echo "ERROR: GPU 2 is busy or unreadable (used MiB: $gpu_used); refusing to start." >&2
    exit 1
fi
gpu_processes=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
    echo "ERROR: GPU 2 has a compute process; refusing to start: $gpu_processes" >&2
    exit 1
fi
RLT_SMOKE_RENDER_DEVICE=$(python - <<'PY'
import subprocess

bus = subprocess.check_output(
    ['nvidia-smi', '-i', '2', '--query-gpu=pci.bus_id', '--format=csv,noheader'],
    text=True,
).strip().lower()
domain, bus_id, slot = bus.split(':')
print(f'pci:{int(domain, 16):04x}:{bus_id}:{slot}')
PY
)
python - <<'PY'
import os
import shutil
import socket

port = int(os.environ['RLT_SMOKE_RAY_PORT'])
for candidate in (port, port + 1, port + 2):
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', candidate))
assert shutil.disk_usage('/dev/shm').free > 6 * 1024**3, 'Need at least 6 GiB free /dev/shm'
PY

# Short local socket paths; CIFS is used only for output and object spilling.
ray_temp_dir=$(mktemp -d /dev/shm/rlt-atomic.XXXXXXXX)
export RAY_TMPDIR="$ray_temp_dir"
mkdir -p "$RLT_SMOKE_RUN_DIR/ray_spill"
exec > >(tee "$RLT_SMOKE_RUN_DIR/train.log") 2>&1
echo "Run directory: $RLT_SMOKE_RUN_DIR"
echo "Ray temp directory: $ray_temp_dir"
echo "Vulkan render device: $RLT_SMOKE_RENDER_DEVICE"
git rev-parse HEAD
git status --short

ray_head_pid=
train_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    # These PIDs are children of this shell, never discovered via pgrep/ray stop.
    if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
        kill -TERM "$train_pid" 2>/dev/null || true
    fi
    if [[ -n "$ray_head_pid" ]] && kill -0 "$ray_head_pid" 2>/dev/null; then
        # ray start --block owns its node and installs a SIGTERM cleanup handler.
        kill -TERM "$ray_head_pid" 2>/dev/null || true
        wait "$ray_head_pid" || true
    fi
    echo "Smoke exit code: $status; logs: $RLT_SMOKE_RUN_DIR"
    echo "Ray diagnostics retained: $ray_temp_dir (no broad cleanup performed)."
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ray start --head --block \
    --port="$RLT_SMOKE_RAY_PORT" \
    --ray-client-server-port="$((RLT_SMOKE_RAY_PORT + 1))" \
    --dashboard-host=127.0.0.1 --dashboard-port="$((RLT_SMOKE_RAY_PORT + 2))" \
    --dashboard-agent-listen-port=0 \
    --min-worker-port=0 --max-worker-port=0 \
    --num-cpus=8 --num-gpus=1 \
    --object-store-memory=2147483648 \
    --temp-dir="$ray_temp_dir" \
    --object-spilling-directory="$RLT_SMOKE_RUN_DIR/ray_spill" \
    --include-dashboard=true --disable-usage-stats \
    > "$RLT_SMOKE_RUN_DIR/ray-head.log" 2>&1 &
ray_head_pid=$!

# Ray writes this file only once this particular head has completed startup.
for (( attempt=0; attempt<120; attempt++ )); do
    if ! kill -0 "$ray_head_pid" 2>/dev/null; then
        tail -n 60 "$RLT_SMOKE_RUN_DIR/ray-head.log"
        echo 'ERROR: isolated Ray head exited before startup.' >&2
        exit 1
    fi
    if [[ -s "$ray_temp_dir/ray_current_cluster" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -s "$ray_temp_dir/ray_current_cluster" ]]; then
    echo 'ERROR: isolated Ray startup timed out; inspect ray-head.log.' >&2
    exit 1
fi
export RAY_ADDRESS
RAY_ADDRESS=$(< "$ray_temp_dir/ray_current_cluster")
if [[ "$RAY_ADDRESS" != *":$RLT_SMOKE_RAY_PORT" ]]; then
    echo "ERROR: unexpected Ray address: $RAY_ADDRESS" >&2
    exit 1
fi
echo "Using isolated Ray at $RAY_ADDRESS (head PID $ray_head_pid)"

python toolkits/rlt/probe_gpu2.py --config-name "$config_name"
if [[ "${1:-}" == --probe ]]; then
    echo 'GPU 2 placement probe passed; training was not started.'
    exit 0
fi

python examples/embodiment/train_embodied_agent.py \
    --config-name "$config_name" --cfg job --resolve runner.max_steps="$RLT_ATOMIC_STEPS" runner.max_epochs="$RLT_ATOMIC_STEPS" \
    > "$RLT_SMOKE_RUN_DIR/resolved-config.yaml"
python examples/embodiment/train_embodied_agent.py \
    --config-name "$config_name" \
    runner.max_steps="$RLT_ATOMIC_STEPS" runner.max_epochs="$RLT_ATOMIC_STEPS" &
train_pid=$!
wait "$train_pid"
train_pid=

}
main "$@"
