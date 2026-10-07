#!/usr/bin/env bash
# Run a bounded pilot on one explicitly selected physical GPU.
set -euo pipefail

world_overrides=()
world_enabled=0
memory_enabled=0
: "${RLT_PHYSICAL_GPU:?Set RLT_PHYSICAL_GPU to a physical GPU index}"
[[ "$RLT_PHYSICAL_GPU" =~ ^[0-9]+$ ]] || { echo 'Invalid GPU index.' >&2; exit 2; }
case "${1:-}" in
    --world)
        world_enabled=1
        : "${RLT_WORLD_CHECKPOINT:?Set RLT_WORLD_CHECKPOINT to a trained sidecar}"
        world_overrides=(+experiment=rlt_latent_world)
        shift ;;
    --memory)
        memory_enabled=1
        case "${RLT_MEMORY_READER:-attention}" in
            attention) world_overrides=(+experiment=rlt_memory) ;;
            response) world_overrides=(+experiment=rlt_memory_response) ;;
            zero) world_overrides=(+experiment=rlt_memory_zero) ;;
            *) echo 'RLT_MEMORY_READER must be attention, response or zero.' >&2; exit 2 ;;
        esac
        shift ;;
esac
case "${RLT_SMOKE_PROFILE:-smoke}" in
    smoke) ;;
    overnight) world_overrides+=(+pilot=rlt_overnight) ;;
    matched) world_overrides+=(+pilot=rlt_memory_matched) ;;
    *) echo 'RLT_SMOKE_PROFILE must be smoke, overnight or matched.' >&2; exit 2 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check && "$1" != --probe ) ]]; then
    echo "Usage: RLT_PHYSICAL_GPU=N bash run_rlt_portable.sh [--world|--memory] [--check|--probe]" >&2
    exit 2
fi
world_overrides+=(
    "cluster.component_placement.actor=$RLT_PHYSICAL_GPU-$RLT_PHYSICAL_GPU"
    "cluster.component_placement.env=$RLT_PHYSICAL_GPU-$RLT_PHYSICAL_GPU"
    "cluster.component_placement.rollout=$RLT_PHYSICAL_GPU-$RLT_PHYSICAL_GPU"
    "runner.logger.experiment_name=stage2_portable"
)

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
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$RLT_PHYSICAL_GPU"
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

export RLINF_VULKAN_PREFIX="${RLINF_VULKAN_PREFIX:-$HOME/.local/rlinf-vulkan}"
# Load only the Vulkan loader; keep this host's NVIDIA driver and C++ runtime.
if [[ -f "$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357" ]]; then
    export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VULKAN_LIBRARY_PATH:-$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357}"
else
    export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VULKAN_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu/libvulkan.so.1}"
fi
export RLT_NVIDIA_EGL_LIBRARY="${RLT_NVIDIA_EGL_LIBRARY:-/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export RLT_SMOKE_RENDER_DEVICE
RLT_SMOKE_RENDER_DEVICE=$(python - <<'PY'
import os
import subprocess
bus = subprocess.check_output(
    ['nvidia-smi', '-i', os.environ['RLT_PHYSICAL_GPU'], '--query-gpu=pci.bus_id', '--format=csv,noheader'],
    text=True,
).strip().lower()
domain, bus_id, slot = bus.split(':')
print(f'pci:{int(domain, 16):04x}:{bus_id}:{slot}')
PY
)
unset DISPLAY WAYLAND_DISPLAY VK_LAYER_PATH VK_INSTANCE_LAYERS __NV_PRIME_RENDER_OFFLOAD
unset RAY_ADDRESS RLINF_NODE_RANK

run_id="stage2_$(hostname -s)_gpu${RLT_PHYSICAL_GPU}_$(date +%Y%m%d_%H%M%S)_$$"
export RLT_SMOKE_RUN_DIR="${RLT_OUTPUT_ROOT:-$RLT_STORAGE/runs/portable_pilots}/$run_id"
requested_ray_port=${RLT_SMOKE_RAY_PORT:-}
# This default is only for config validation; auto-selection happens after the GPU guard.
export RLT_SMOKE_RAY_PORT="${requested_ray_port:-6510}"

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
for name in ('SAPIEN_VULKAN_LIBRARY_PATH', 'RLT_NVIDIA_EGL_LIBRARY', '__EGL_VENDOR_LIBRARY_FILENAMES'):
    assert Path(os.environ[name]).is_file(), f'Missing {name}: {os.environ[name]}'
gpu = os.environ['RLT_PHYSICAL_GPU']
assert dict(cfg.cluster.component_placement) == dict.fromkeys(('actor', 'env', 'rollout'), f'{gpu}-{gpu}')
if OmegaConf.select(cfg, 'actor.model.interaction_memory.enabled', default=False):
    from rlinf.algorithms.rlt.interaction_memory import validate_interaction_memory_cfg
    validate_interaction_memory_cfg(cfg)
assert cfg.rollout.expert_model is None
assert cfg.actor.global_batch_size % cfg.actor.micro_batch_size == 0
assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates > 0
port = int(os.environ['RLT_SMOKE_RAY_PORT'])
assert 1024 <= port <= 65533 and not set(range(port, port + 3)) & {6379, 6385, 6386, 6387}, 'Ray ports overlap Stage 1'
print(f'Config: maniskill_rlt_stage2_smoke_gpu2, profile={os.environ.get("RLT_SMOKE_PROFILE", "smoke")}; RLinf physical rank: {gpu}; worker CUDA ordinal: 0')
print(f'Stage 1: {weights} ({weights.stat().st_size:,} bytes)')
print(f'Norm stats: {stats_path}')
print(
    f'Budget: {cfg.env.train.total_num_envs} train envs / {cfg.env.eval.total_num_envs} eval envs; '
    f'{cfg.env.train.max_episode_steps} control steps; stop at global step {cfg.runner.max_steps}'
)
print(f'Resume: {cfg.runner.resume_dir}')
print(f'Intervals: evaluate every {cfg.runner.val_check_interval}; save every {cfg.runner.save_interval}')
print(f'Batch: global={cfg.actor.global_batch_size}, micro={cfg.actor.micro_batch_size}; '
      f'at most {cfg.algorithm.rlt_schedule.max_updates_per_train_step} AC updates per iteration; '
      f'warmup={cfg.algorithm.rlt_schedule.warmup_post_collect_updates}')
print(f'Planned output: {os.environ["RLT_SMOKE_RUN_DIR"]}')
print('Preflight OK (paths/config only; GPU execution has not been tested).')
PY
if [[ "${1:-}" == --check ]]; then
    exit 0
fi

# Reject a busy GPU or occupied port before creating any run or starting Ray.
exec {rlt_gpu_lease}>"/tmp/rlt-atomic-gpu${RLT_PHYSICAL_GPU}.lock"
flock -n "$rlt_gpu_lease" || { echo "GPU $RLT_PHYSICAL_GPU project lease is held; refusing to start." >&2; exit 1; }
gpu_used=$(nvidia-smi -i "$RLT_PHYSICAL_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ ! "$gpu_used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || (( gpu_used > 1024 )); then
    echo "ERROR: GPU $RLT_PHYSICAL_GPU is busy or unreadable (used MiB: $gpu_used); refusing to start." >&2
    exit 1
fi
RLT_SMOKE_RAY_PORT=$(python - "$requested_ray_port" <<'PY'
import os
import shutil
import sys

from toolkits.rlt.portable_ports import select_ray_ports

assert shutil.disk_usage('/dev/shm').free > 6 * 1024**3, 'Need at least 6 GiB free /dev/shm'
print(select_ray_ports(int(sys.argv[1]) if sys.argv[1] else None))
PY
)
export RLT_SMOKE_RAY_PORT
echo "Selected Ray head/client/dashboard ports: $RLT_SMOKE_RAY_PORT/$((RLT_SMOKE_RAY_PORT + 1))/$((RLT_SMOKE_RAY_PORT + 2))"

# Short local socket paths; CIFS is used only for output and object spilling.
ray_temp_dir=$(mktemp -d "/dev/shm/rlt-gpu${RLT_PHYSICAL_GPU}.XXXXXXXX")
export RAY_TMPDIR="$ray_temp_dir"
mkdir -p "$RLT_SMOKE_RUN_DIR/ray_spill"
export VK_DRIVER_FILES="$RLT_SMOKE_RUN_DIR/nvidia_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
python - <<'PY'
import json
import os
from pathlib import Path
Path(os.environ["VK_DRIVER_FILES"]).write_text(json.dumps({
    "file_format_version": "1.0.0",
    "ICD": {"library_path": os.environ["RLT_NVIDIA_EGL_LIBRARY"], "api_version": "1.2.0"},
}))
PY
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
    echo "Pilot exit code: $status; logs: $RLT_SMOKE_RUN_DIR"
    printf '%s\n' "$status" > "$RLT_SMOKE_RUN_DIR/exit_code.txt"
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

timeout --signal=TERM --kill-after=30s 300s python -m toolkits.rlt.probe_portable --gpu "$RLT_PHYSICAL_GPU" "${world_overrides[@]}"
if [[ "${1:-}" == --probe ]]; then
    echo "GPU $RLT_PHYSICAL_GPU placement and RGB simulation probes passed."
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
if (( (world_enabled == 1 || memory_enabled == 1) && smoke_steps >= 2 )); then
    adapter_prefix=latent_world.
    [[ "$memory_enabled" == 0 ]] || adapter_prefix=memory_encoder.
    if [[ "$memory_enabled" == 1 && "${RLT_MEMORY_READER:-attention}" != attention ]]; then
        # These readers have no parameters; the learned critic must still update.
        adapter_prefix=q_head.
    fi
    audit_root="$RLT_SMOKE_RUN_DIR/stage2_portable/checkpoints"
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
            --prefix "$adapter_prefix" --require-change
    else
        echo 'ERROR: Adapter audit requires two consecutive saved checkpoints.' >&2
        exit 1
    fi
fi
