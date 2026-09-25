#!/usr/bin/env bash

set -uo pipefail

RLINF_ROOT=/home/luokz/rlinf_rlt/UPT_dev
RLT_STORAGE=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill

DATASET_DIR="$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
PI05_DIR="$RLT_STORAGE/models/pi05_base"
RUNS_DIR="$RLT_STORAGE/runs"
LOG_ROOT="$RUNS_DIR/stage1"
RAY_SPILL_DIR="$RLT_STORAGE/ray_spill"
RAY_TEMP_DIR=/dev/shm/luokz_ray_stage1

cd "$RLINF_ROOT" || exit 1
source .venv/bin/activate

export RLINF_ROOT RLT_STORAGE DATASET_DIR PI05_DIR RUNS_DIR LOG_ROOT
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export REPO_PATH="$RLINF_ROOT"
export HF_LEROBOT_HOME="$RLT_STORAGE/datasets/lerobot"
export HF_HUB_CACHE="$RLT_STORAGE/cache/huggingface-hub"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
# The 4.1B OpenPI + RLT model leaves little headroom on 46 GB L40s.  Let the
# CUDA allocator grow segments instead of leaving unusable fragments between
# the 128 gradient-accumulation micro-batches.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
unset RAY_ADDRESS

test -s "$PI05_DIR/model.safetensors" || {
    echo "ERROR: missing $PI05_DIR/model.safetensors"
    exit 1
}
test -s "$DATASET_DIR/norm_stats.json" || {
    echo "ERROR: missing $DATASET_DIR/norm_stats.json"
    exit 1
}

mkdir -p "$LOG_ROOT" "$RAY_SPILL_DIR" "$RAY_TEMP_DIR"

if ! ray status >/dev/null 2>&1; then
    ray start \
        --head \
        --port=6379 \
        --num-cpus=32 \
        --num-gpus=2 \
        --object-store-memory=8589934592 \
        --temp-dir="$RAY_TEMP_DIR" \
        --object-spilling-directory="$RAY_SPILL_DIR" \
        --include-dashboard=true \
        --disable-usage-stats
fi

RUN_NAME="maniskill_rlt_stage1_2xl40_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$RUN_NAME"
export RUN_NAME RUN_DIR
mkdir -p "$RUN_DIR"

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ray status

python examples/sft/train_vla_sft.py \
    --config-name maniskill_rlt_stage1_sft_openpi_pi05 \
    'cluster.component_placement={actor\,env\,rollout:0-1}' \
    data.train_data_paths.0.dataset_path="$DATASET_DIR" \
    actor.model.model_path="$PI05_DIR" \
    actor.model.openpi_data.norm_stats_path="$DATASET_DIR/norm_stats.json" \
    actor.micro_batch_size=1 \
    actor.global_batch_size=256 \
    actor.optim.lr=2.5e-5 \
    actor.optim.lr_warmup_steps=500 \
    actor.optim.total_training_steps=10000 \
    runner.max_steps=2000 \
    runner.save_interval=250 \
    runner.val_check_interval=-1 \
    runner.logger.log_path="$LOG_ROOT" \
    runner.logger.project_name=rlinf-rlt \
    runner.logger.experiment_name="$RUN_NAME" \
    'runner.logger.logger_backends=[wandb]' \
    2>&1 | tee "$RUN_DIR/train.log"

train_status=${PIPESTATUS[0]}
echo "Training process exited with status: $train_status"
echo "Training log: $RUN_DIR/train.log"
exit "$train_status"
