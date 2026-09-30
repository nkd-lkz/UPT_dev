#!/usr/bin/env bash
# Run an integration smoke and then a bounded pilot in the current branch.
set -euo pipefail
if [[ $# != 2 || ( "$1" != world && "$1" != memory ) || ! "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash run_rlt_overnight.sh <world|memory> <physical GPU index>" >&2
    exit 2
fi
variant=$1
export RLT_PHYSICAL_GPU=$2
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
export RLINF_VENV="${RLINF_VENV:-$root/../UPT_dev/.venv}"
export RLT_STORAGE="${RLT_STORAGE:-/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill}"
export RLT_STAGE1_ACTOR="${RLT_STAGE1_ACTOR:-$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016/checkpoints/global_step_2000/actor}"
if [[ "$variant" == world ]]; then
    export RLT_WORLD_CHECKPOINT="${RLT_WORLD_CHECKPOINT:-$RLT_STORAGE/research/flare_step2000_pilot_20260930_005943/residual/stage1b/best.pt}"
fi
export RLT_SMOKE_RAY_PORT="${RLT_SMOKE_RAY_PORT:-$((6510 + RLT_PHYSICAL_GPU * 10))}"
export RLT_OUTPUT_ROOT="${RLT_OUTPUT_ROOT:-$RLT_STORAGE/runs/inspur_${variant}}"
unset RLT_SMOKE_RESUME_DIR
night_steps=${RLT_NIGHT_STEPS:-1000}
[[ "$night_steps" =~ ^[1-9][0-9]*$ ]] && (( night_steps >= 20 && night_steps <= 5000 && night_steps % 10 == 0 )) || {
    echo 'RLT_NIGHT_STEPS must be a multiple of 10 in [20, 5000].' >&2; exit 2;
}
if [[ "${RLT_NIGHT_CHILD:-0}" != 1 ]]; then
    hours=${RLT_NIGHT_HOURS:-12}
    [[ "$hours" =~ ^[0-9]+$ ]] && (( hours >= 1 && hours <= 24 )) || {
        echo 'RLT_NIGHT_HOURS must be in [1, 24].' >&2; exit 2;
    }
    mkdir -p "$RLT_OUTPUT_ROOT"
    night_log="$RLT_OUTPUT_ROOT/night_$(hostname -s)_gpu${RLT_PHYSICAL_GPU}_$(date +%Y%m%d_%H%M%S)_$$.log"
    exec > >(tee "$night_log") 2>&1
    echo "Night log: $night_log"
    echo "Wall-time budget: ${hours}h including probe, smoke and training."
    set +e
    RLT_NIGHT_CHILD=1 timeout --signal=TERM --kill-after=120s "${hours}h" bash "$0" "$variant" "$RLT_PHYSICAL_GPU"
    result=$?
    set -e
    printf '%s\n' "$result" > "$night_log.exit_code"
    if [[ "$result" == 124 ]]; then
        echo 'Wall-time budget reached. Use the last periodic checkpoint; an interrupted update is not saved.'
    fi
    echo "Night exit code: $result"
    exit "$result"
fi
export RLT_SMOKE_PROFILE=smoke RLT_SMOKE_STEPS=2 RLT_EPISODE_STEPS=500
export RLT_SAVE_INTERVAL=1 RLT_VAL_INTERVAL=1 RLT_LONG_RUN=0
bash run_rlt_portable.sh "--$variant"
# Wait only for this finished smoke's GPU contexts to disappear.
for (( attempt=0; attempt<30; attempt++ )); do
    used=$(nvidia-smi -i "$RLT_PHYSICAL_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && (( used <= 1024 )); then
        break
    fi
    sleep 2
done
export RLT_SMOKE_PROFILE=overnight RLT_LONG_RUN=1
export RLT_SMOKE_STEPS="${RLT_NIGHT_STEPS:-1000}" RLT_EPISODE_STEPS=500
export RLT_SAVE_INTERVAL=10 RLT_VAL_INTERVAL=10
echo 'Smoke passed. Starting a fresh pilot with 512 BC warmup updates and periodic evaluation.'
bash run_rlt_portable.sh "--$variant"
