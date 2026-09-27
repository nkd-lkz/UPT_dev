#!/usr/bin/env bash
# Run an isolated Stage1 evaluation or Stage2 baseline job on physical GPU 2.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash toolkits/rlt/run_baseline_gpu2_job.sh <stage1-eval|stage2-train> [--check|--probe]" >&2
    exit 2
fi
job=$1
mode=${2:-run}
if [[ "$job" != stage1-eval && "$job" != stage2-train ]]; then
    echo "Unknown job: $job" >&2
    exit 2
fi
if [[ "$mode" != run && "$mode" != --check && "$mode" != --probe ]]; then
    echo "Unknown mode: $mode" >&2
    exit 2
fi

RLINF_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$RLINF_ROOT"
source "$RLINF_ROOT/.venv/bin/activate"
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
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline

export RLINF_VULKAN_PREFIX="${RLINF_VULKAN_PREFIX:-/home/luokz/.local/rlinf-vulkan}"
export PATH="$RLINF_VULKAN_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/lib64"
export SAPIEN_VULKAN_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357"
export VK_DRIVER_FILES="$RLINF_VULKAN_PREFIX/share/vulkan/icd.d/nvidia_headless_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export RLT_GPU2_RENDER_DEVICE="${RLT_GPU2_RENDER_DEVICE:-pci:0000:e1:00.0}"
unset DISPLAY WAYLAND_DISPLAY VK_LAYER_PATH VK_INSTANCE_LAYERS __NV_PRIME_RENDER_OFFLOAD
unset RAY_ADDRESS RLINF_NODE_RANK

run_stamp=$(date +%Y%m%d_%H%M%S)_$$
if [[ "$job" == stage1-eval ]]; then
    export RLT_JOB_RUN_DIR="${RLT_STAGE1_EVAL_RUN_DIR:-$RLT_STORAGE/runs/stage1_eval20/eval20_$run_stamp}"
    export RLT_STAGE1_EVAL_RUN_DIR="$RLT_JOB_RUN_DIR"
    export RLT_RAY_PORT="${RLT_STAGE1_EVAL_RAY_PORT:-6390}"
    log_name=eval.log
else
    export RLT_JOB_RUN_DIR="${RLT_STAGE2_RUN_DIR:-$RLT_STORAGE/runs/stage2_baseline/baseline_$run_stamp}"
    export RLT_STAGE2_RUN_DIR="$RLT_JOB_RUN_DIR"
    export RLT_RAY_PORT="${RLT_STAGE2_RAY_PORT:-6393}"
    log_name=train.log
fi

# This check only composes config and reads files. It does not create output,
# connect to Ray, initialize CUDA, or deserialize the checkpoint.
python toolkits/rlt/check_baseline_gpu2.py "$job"
if [[ "$mode" == --check ]]; then
    exit 0
fi

gpu_used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ ! "$gpu_used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || (( gpu_used > 1024 )); then
    echo "ERROR: GPU 2 is busy or unreadable (used MiB: $gpu_used); refusing to start." >&2
    exit 1
fi
export RLT_GPU2_RENDER_DEVICE
RLT_GPU2_RENDER_DEVICE=$(python - <<'PY'
import subprocess

bus = subprocess.check_output(
    ["nvidia-smi", "-i", "2", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
    text=True,
).strip().lower()
domain, bus_id, slot = bus.split(":")
print(f"pci:{int(domain, 16):04x}:{bus_id}:{slot}")
PY
)
python - <<'PY'
import os
import shutil
import socket

port = int(os.environ["RLT_RAY_PORT"])
for candidate in (port, port + 1, port + 2):
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", candidate))
assert shutil.disk_usage("/dev/shm").free > 6 * 1024**3, "Need at least 6 GiB free /dev/shm"
PY

ray_temp_dir=$(mktemp -d /dev/shm/rlt-baseline.XXXXXXXX)
export RAY_TMPDIR="$ray_temp_dir"
mkdir -p "$RLT_JOB_RUN_DIR/ray_spill"
exec > >(tee "$RLT_JOB_RUN_DIR/$log_name") 2>&1
echo "Job: $job"
echo "Run directory: $RLT_JOB_RUN_DIR"
echo "Ray temp directory: $ray_temp_dir"
echo "Vulkan render device: $RLT_GPU2_RENDER_DEVICE"
git rev-parse HEAD
git status --short

ray_head_pid=
job_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$job_pid" ]] && kill -0 "$job_pid" 2>/dev/null; then
        kill -TERM "$job_pid" 2>/dev/null || true
    fi
    if [[ -n "$ray_head_pid" ]] && kill -0 "$ray_head_pid" 2>/dev/null; then
        kill -TERM "$ray_head_pid" 2>/dev/null || true
        wait "$ray_head_pid" || true
    fi
    echo "Job exit code: $status; logs: $RLT_JOB_RUN_DIR"
    echo "Ray diagnostics retained: $ray_temp_dir (no broad cleanup performed)."
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ray start --head --block \
    --port="$RLT_RAY_PORT" \
    --ray-client-server-port="$((RLT_RAY_PORT + 1))" \
    --dashboard-host=127.0.0.1 --dashboard-port="$((RLT_RAY_PORT + 2))" \
    --dashboard-agent-listen-port=0 \
    --min-worker-port=0 --max-worker-port=0 \
    --num-cpus=8 --num-gpus=1 \
    --object-store-memory=2147483648 \
    --temp-dir="$ray_temp_dir" \
    --object-spilling-directory="$RLT_JOB_RUN_DIR/ray_spill" \
    --include-dashboard=true --disable-usage-stats \
    > "$RLT_JOB_RUN_DIR/ray-head.log" 2>&1 &
ray_head_pid=$!

for (( attempt=0; attempt<120; attempt++ )); do
    if ! kill -0 "$ray_head_pid" 2>/dev/null; then
        tail -n 60 "$RLT_JOB_RUN_DIR/ray-head.log"
        echo "ERROR: isolated Ray head exited before startup." >&2
        exit 1
    fi
    if [[ -s "$ray_temp_dir/ray_current_cluster" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -s "$ray_temp_dir/ray_current_cluster" ]]; then
    echo "ERROR: isolated Ray startup timed out; inspect ray-head.log." >&2
    exit 1
fi
export RAY_ADDRESS
RAY_ADDRESS=$(< "$ray_temp_dir/ray_current_cluster")
if [[ "$RAY_ADDRESS" != *":$RLT_RAY_PORT" ]]; then
    echo "ERROR: unexpected Ray address: $RAY_ADDRESS" >&2
    exit 1
fi
echo "Using isolated Ray at $RAY_ADDRESS (head PID $ray_head_pid)"

if [[ "$job" == stage1-eval ]]; then
    probe_args=(
        --config-dir "$RLINF_ROOT/evaluations/maniskill"
        --config-name maniskill_rlt_stage1_eval20
    )
else
    probe_args=(
        --config-name maniskill_rlt_stage2_ac_mlp
        --override +experiment=rlt_baseline_gpu2
    )
fi
python toolkits/rlt/probe_gpu2.py "${probe_args[@]}"
if [[ "$mode" == --probe ]]; then
    echo "GPU 2 placement probe passed; evaluation/training was not started."
    exit 0
fi

if [[ "$job" == stage1-eval ]]; then
    eval_cmd=(
        python evaluations/eval_embodied_agent.py
        --config-path "$RLINF_ROOT/evaluations/maniskill"
        --config-name maniskill_rlt_stage1_eval20
    )
    "${eval_cmd[@]}" --cfg job --resolve > "$RLT_JOB_RUN_DIR/resolved-config.yaml"
    "${eval_cmd[@]}" &
else
    train_cmd=(
        python examples/embodiment/train_embodied_agent.py
        --config-name maniskill_rlt_stage2_ac_mlp
        +experiment=rlt_baseline_gpu2
    )
    if [[ -n "${RLT_STAGE2_MAX_STEPS:-}" ]]; then
        train_cmd+=("runner.max_steps=$RLT_STAGE2_MAX_STEPS")
    fi
    if [[ -n "${RLT_STAGE2_RESUME_DIR:-}" ]]; then
        train_cmd+=("runner.resume_dir=$RLT_STAGE2_RESUME_DIR")
    fi
    "${train_cmd[@]}" --cfg job --resolve > "$RLT_JOB_RUN_DIR/resolved-config.yaml"
    "${train_cmd[@]}" &
fi
job_pid=$!
wait "$job_pid"
job_pid=
