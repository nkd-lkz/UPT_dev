#!/usr/bin/env bash
# Start an isolated, bounded Stage 2 smoke job; never attach to Stage 1 Ray.
set -euo pipefail

world_overrides=()
world_enabled=0
if [[ "${1:-}" == --world ]]; then
    world_enabled=1
    : "${RLT_WORLD_CHECKPOINT:?Set RLT_WORLD_CHECKPOINT to a trained sidecar}"
    world_overrides=(+experiment=rlt_latent_world runner.logger.experiment_name=stage2_world_smoke)
    shift
fi

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check && "$1" != --probe ) ]]; then
    echo "Usage: bash run_rlt_stage2_smoke_gpu2.sh [--world] [--check|--probe]" >&2
    exit 2
fi

smoke_steps=${RLT_SMOKE_STEPS:-2}
max_smoke_steps=20
if [[ "${RLT_LONG_RUN:-0}" == 1 ]]; then
    max_smoke_steps=5000
fi
if [[ ! "$smoke_steps" =~ ^[0-9]+$ ]] || (( smoke_steps < 1 || smoke_steps > max_smoke_steps )); then
    echo "RLT_SMOKE_STEPS must be in [1, $max_smoke_steps]. Set RLT_LONG_RUN=1 for a bounded long run." >&2
    exit 2
fi
save_interval=${RLT_SAVE_INTERVAL:-1}
val_interval=${RLT_VAL_INTERVAL:-1}
episode_steps=${RLT_EPISODE_STEPS:-40}
for interval in "$save_interval" "$val_interval"; do
    if [[ ! "$interval" =~ ^[0-9]+$ ]] || (( interval < 1 || interval > smoke_steps )); then
        echo 'RLT_SAVE_INTERVAL and RLT_VAL_INTERVAL must be positive and no larger than RLT_SMOKE_STEPS.' >&2
        exit 2
    fi
done
if (( save_interval % val_interval != 0 )); then
    echo 'RLT_SAVE_INTERVAL must be divisible by RLT_VAL_INTERVAL.' >&2
    exit 2
fi
if [[ ! "$episode_steps" =~ ^[0-9]+$ ]] \
    || (( episode_steps < 10 || episode_steps > 500 || episode_steps % 10 != 0 )); then
    echo 'RLT_EPISODE_STEPS must be a multiple of 10 in [10, 500].' >&2
    exit 2
fi
world_overrides+=(
    "runner.max_steps=$smoke_steps"
    "runner.max_epochs=$smoke_steps"
    "runner.save_interval=$save_interval"
    "runner.val_check_interval=$val_interval"
    "env.train.max_episode_steps=$episode_steps"
    "env.train.max_steps_per_rollout_epoch=$episode_steps"
    "env.eval.max_episode_steps=$episode_steps"
    "env.eval.max_steps_per_rollout_epoch=$episode_steps"
)
if [[ -n "${RLT_SMOKE_RESUME_DIR:-}" ]]; then
    [[ -d "$RLT_SMOKE_RESUME_DIR/actor" ]] || { echo 'Missing resume actor directory.' >&2; exit 2; }
    world_overrides+=("runner.resume_dir=$RLT_SMOKE_RESUME_DIR")
fi

RLINF_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$RLINF_ROOT"
source "${RLINF_VENV:-$RLINF_ROOT/.venv}/bin/activate"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RLT_STORAGE="${RLT_STORAGE:-/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill}"
export RLT_STAGE1_ACTOR="${RLT_STAGE1_ACTOR:-$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750/actor}"
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

run_id="stage2_gpu2_$(date +%Y%m%d_%H%M%S)_$$"
export RLT_SMOKE_RUN_DIR="$RLT_STORAGE/runs/stage2_smoke/$run_id"
export RLT_SMOKE_RAY_PORT="${RLT_SMOKE_RAY_PORT:-6382}"

# Read-only preflight: no Ray, CUDA allocation, checkpoint deserialization or mkdir.
python - "${world_overrides[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(config_dir=os.environ['EMBODIED_PATH'] + '/config', version_base='1.1'):
    cfg = compose(config_name='maniskill_rlt_stage2_smoke_gpu2', overrides=sys.argv[1:])
OmegaConf.resolve(cfg)
if OmegaConf.select(cfg, 'actor.model.latent_world.enabled', default=False):
    assert Path(cfg.actor.model.latent_world.checkpoint).is_file(), 'Missing world checkpoint'
    assert cfg.actor.fsdp_config.use_orig_params
weights = Path(cfg.rollout.rlt_feature_model.model_path) / 'model_state_dict/full_weights.pt'
assert weights.is_file() and weights.stat().st_size > 0, f'Missing weights: {weights}'
stats_path = Path(cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path)
stats = json.loads(stats_path.read_text())
assert {'state', 'actions'} <= stats['norm_stats'].keys(), f'Invalid norm stats: {stats_path}'
for name in ('SAPIEN_VULKAN_LIBRARY_PATH', 'VK_DRIVER_FILES', '__EGL_VENDOR_LIBRARY_FILENAMES'):
    assert Path(os.environ[name]).is_file(), f'Missing {name}: {os.environ[name]}'
assert dict(cfg.cluster.component_placement) == dict(actor='2-2', env='2-2', rollout='2-2')
assert cfg.rollout.expert_model is None
assert cfg.actor.global_batch_size % cfg.actor.micro_batch_size == 0
assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates > 0
port = int(os.environ['RLT_SMOKE_RAY_PORT'])
assert 1024 <= port <= 65533 and not set(range(port, port + 3)) & {6379, 6385, 6386, 6387}, 'Ray ports overlap Stage 1'
print('Config: maniskill_rlt_stage2_smoke_gpu2; RLinf physical rank: 2; worker CUDA ordinal: 0')
print(f'Stage 1: {weights} ({weights.stat().st_size:,} bytes)')
print(f'Norm stats: {stats_path}')
print(
    f'Budget: 2 train envs / 1 eval env; '
    f'{cfg.env.train.max_episode_steps} control steps; stop at global step {cfg.runner.max_steps}'
)
print(f'Resume: {cfg.runner.resume_dir}')
print(f'Intervals: evaluate every {cfg.runner.val_check_interval}; save every {cfg.runner.save_interval}')
print('Batch: global=4, micro=2; at most 2 AC updates per iteration')
print(f'Planned output: {os.environ["RLT_SMOKE_RUN_DIR"]}')
print('Preflight OK (paths/config only; GPU execution has not been tested).')
PY
if [[ "${1:-}" == --check ]]; then
    exit 0
fi

# Reject a busy GPU or occupied port before creating any run or starting Ray.
gpu_used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ ! "$gpu_used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || (( gpu_used > 1024 )); then
    echo "ERROR: GPU 2 is busy or unreadable (used MiB: $gpu_used); refusing to start." >&2
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
ray_temp_dir=$(mktemp -d /dev/shm/rlt2.XXXXXXXX)
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

python toolkits/rlt/probe_gpu2.py
if [[ "${1:-}" == --probe ]]; then
    echo 'GPU 2 placement probe passed; training was not started.'
    exit 0
fi

python examples/embodiment/train_embodied_agent.py \
    --config-name maniskill_rlt_stage2_smoke_gpu2 "${world_overrides[@]}" --cfg job --resolve \
    > "$RLT_SMOKE_RUN_DIR/resolved-config.yaml"
python examples/embodiment/train_embodied_agent.py \
    --config-name maniskill_rlt_stage2_smoke_gpu2 "${world_overrides[@]}" &
train_pid=$!
wait "$train_pid"
train_pid=
if [[ "$world_enabled" == 1 ]] && (( smoke_steps >= 2 )); then
    audit_root="$RLT_SMOKE_RUN_DIR/stage2_world_smoke/checkpoints"
    latest="$audit_root/global_step_$smoke_steps/actor/model_state_dict/full_weights.pt"
    mapfile -t saved_steps < <(
        find "$audit_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
            | sed 's/^global_step_//' | sort -n
    )
    previous=
    if (( ${#saved_steps[@]} >= 2 )); then
        previous_step=${saved_steps[$((${#saved_steps[@]} - 2))]}
        previous="$audit_root/global_step_$previous_step/actor/model_state_dict/full_weights.pt"
    elif [[ -n "${RLT_SMOKE_RESUME_DIR:-}" ]]; then
        previous="$RLT_SMOKE_RESUME_DIR/actor/model_state_dict/full_weights.pt"
    fi
    if [[ -f "$previous" && -f "$latest" ]]; then
        python -m toolkits.rlt.audit_adapter --before "$previous" --after "$latest" \
            --prefix "latent_world." --require-change
    else
        echo 'ERROR: Adapter audit requires two consecutive saved checkpoints.' >&2
        exit 1
    fi
fi
