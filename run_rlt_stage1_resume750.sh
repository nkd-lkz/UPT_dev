#!/usr/bin/env bash
# Resume the interrupted baseline without modifying its original run directory.
set -euo pipefail
cd /home/luokz/rlinf_rlt/UPT_dev
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EMBODIED_PATH="$PWD/examples/embodiment" REPO_PATH="$PWD"
export RLT_STORAGE=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill
export HF_LEROBOT_HOME="$RLT_STORAGE/datasets/lerobot"
export HF_HUB_CACHE="$RLT_STORAGE/cache/huggingface-hub"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online WANDB__SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=300
export WANDB_ENTITY=c6522513-sustech
# The long-lived tmux server predates the user's working HTTPS proxy settings.
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17891}"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1,::1,10.16.61.119${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
unset RAY_ADDRESS RLINF_NODE_RANK
# Use the personal login in ~/.netrc, not credentials inherited by tmux.
unset WANDB_API_KEY WANDB_BASE_URL WANDB_RUN_ID WANDB_RESUME

resume_dir="$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750"
dataset_dir="$HF_LEROBOT_HOME/maniskill_peginsertionside_joint"
model_dir="$RLT_STORAGE/models/pi05_base"
test -s "$resume_dir/actor/dcp_checkpoint/.metadata"
test -s "$resume_dir/actor/dcp_checkpoint/__0_0.distcp"
test -s "$resume_dir/actor/dcp_checkpoint/__1_0.distcp"
test -s "$dataset_dir/norm_stats.json"
test -s "$model_dir/model.safetensors"
python - <<'PY'
import socket
import subprocess
import wandb

username = wandb.Api().viewer.username
assert username == 'c6522513', f'Unexpected W&B account: {username}'
print(f'W&B account verified: {username}; entity: c6522513-sustech')

for port in (6385, 6386, 6387):
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', port))
for gpu in ('0', '1'):
    used = int(subprocess.check_output(
        ['nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True
    ).strip())
    assert used < 1024, f'GPU {gpu} is busy ({used} MiB); refusing to start'
PY
run_name="maniskill_rlt_stage1_resume750_$(date +%Y%m%d_%H%M%S)"
log_root="$RLT_STORAGE/runs/stage1"
run_dir="$log_root/$run_name"
ray_temp_dir=$(mktemp -d /dev/shm/rlt1.XXXXXXXX)
export RAY_TMPDIR="$ray_temp_dir"
mkdir -p "$run_dir/ray_spill"
exec > >(tee "$run_dir/train.log") 2>&1
echo "RUN_DIR=$run_dir"
echo "RESUME_DIR=$resume_dir"
echo "RAY_TEMP_DIR=$ray_temp_dir"
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
    echo "Stage 1 exit code: $status; logs: $run_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ray start --head --block --port=6385 --ray-client-server-port=6386 \
    --dashboard-host=127.0.0.1 --dashboard-port=6387 --dashboard-agent-listen-port=0 \
    --min-worker-port=0 --max-worker-port=0 \
    --num-cpus=32 --num-gpus=2 --object-store-memory=8589934592 \
    --temp-dir="$ray_temp_dir" --object-spilling-directory="$run_dir/ray_spill" \
    --include-dashboard=true --disable-usage-stats > "$run_dir/ray-head.log" 2>&1 &
ray_head_pid=$!
for ((attempt=0; attempt<120; attempt++)); do
    kill -0 "$ray_head_pid" 2>/dev/null || { tail -n 40 "$run_dir/ray-head.log"; exit 1; }
    [[ -s "$ray_temp_dir/ray_current_cluster" ]] && break
    sleep 1
done
test -s "$ray_temp_dir/ray_current_cluster"
export RAY_ADDRESS
RAY_ADDRESS=$(< "$ray_temp_dir/ray_current_cluster")
[[ "$RAY_ADDRESS" == *:6385 ]]
echo "RAY_ADDRESS=$RAY_ADDRESS"
python examples/sft/train_vla_sft.py \
    --config-name maniskill_rlt_stage1_sft_openpi_pi05 \
    'cluster.component_placement={actor\,env\,rollout:0-1}' \
    data.train_data_paths.0.dataset_path="$dataset_dir" \
    actor.model.model_path="$model_dir" \
    actor.model.openpi_data.norm_stats_path="$dataset_dir/norm_stats.json" \
    actor.micro_batch_size=1 actor.global_batch_size=256 \
    actor.optim.lr=2.5e-5 actor.optim.lr_warmup_steps=500 \
    actor.optim.total_training_steps=10000 \
    runner.max_steps=2000 runner.save_interval=250 runner.val_check_interval=-1 \
    +runner.resume_dir="$resume_dir" \
    runner.logger.log_path="$log_root" \
    runner.logger.project_name=rlinf-rlt runner.logger.experiment_name="$run_name" \
    +runner.logger.wandb_entity="$WANDB_ENTITY" \
    'runner.logger.logger_backends=[wandb]' &
train_pid=$!
wait "$train_pid"
train_pid=
