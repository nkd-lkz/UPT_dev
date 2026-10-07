#!/usr/bin/env bash
# One bounded member of the same-capacity zero/response comparison.
set -euo pipefail
if [[ $# != 2 || ! "$1" =~ ^(zero|response|attention)$ || ! "$2" =~ ^[0-9]+$ ]]; then
    echo 'Usage: bash run_rlt_memory_comparison.sh <zero|response|attention> <physical GPU>' >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
export RLT_MEMORY_READER=$1 RLT_PHYSICAL_GPU=$2
export RLINF_VENV="${RLINF_VENV:-$root/../UPT_dev/.venv}"
: "${RLT_OUTPUT_ROOT:?Set a unique output root for this comparison member}"
hours=${RLT_COMPARISON_HOURS:-8}
steps=${RLT_COMPARISON_STEPS:-1000}
interval=${RLT_COMPARISON_INTERVAL:-25}
if [[ ! "$hours" =~ ^[0-9]+$ || ! "$steps" =~ ^[0-9]+$ || ! "$interval" =~ ^[0-9]+$ ]] \
    || (( hours < 1 || hours > 24 || steps < 20 || steps > 5000 || interval < 1 \
    || steps % interval != 0 || interval > steps / 2 )); then
    echo 'Invalid comparison budget: hours 1..24, steps 20..5000, interval must divide steps and leave two saves.' >&2
    exit 2
fi
if [[ "${RLT_COMPARISON_CHILD:-0}" != 1 ]]; then
    mkdir -p "$RLT_OUTPUT_ROOT"
    [[ ! -e "$RLT_OUTPUT_ROOT/comparison.log" ]] || { echo 'Output already exists; choose a fresh root.' >&2; exit 2; }
    exec > >(tee "$RLT_OUTPUT_ROOT/comparison.log") 2>&1
    printf 'Reader=%s GPU=%s budget=%sh max_steps=%s save/eval=%s\n' "$1" "$2" "$hours" "$steps" "$interval"
    git rev-parse HEAD > "$RLT_OUTPUT_ROOT/code_revision.txt"
    date --iso-8601=seconds > "$RLT_OUTPUT_ROOT/started_at.txt"
    set +e
    RLT_COMPARISON_CHILD=1 timeout --signal=TERM --kill-after=120s "${hours}h" bash "$0" "$1" "$2"
    result=$?
    set -e
    printf '%s\n' "$result" > "$RLT_OUTPUT_ROOT/exit_code.txt"
    date --iso-8601=seconds > "$RLT_OUTPUT_ROOT/finished_at.txt"
    echo "Comparison member stopped: exit=$result (124 means wall-time budget)."
    exit "$result"
fi
unset RLT_SMOKE_RAY_PORT RLT_SMOKE_RESUME_DIR
export RLT_SMOKE_PROFILE=smoke RLT_SMOKE_STEPS=2 RLT_EPISODE_STEPS=100
export RLT_SAVE_INTERVAL=1 RLT_VAL_INTERVAL=1 RLT_LONG_RUN=0
echo 'phase=smoke' > "$RLT_OUTPUT_ROOT/phase.txt"
bash run_rlt_portable.sh --memory
for (( attempt=0; attempt<30; attempt++ )); do
    used=$(nvidia-smi -i "$RLT_PHYSICAL_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && (( used <= 1024 )); then break; fi
    sleep 2
done
export RLT_SMOKE_PROFILE=matched RLT_LONG_RUN=1 RLT_EPISODE_STEPS=500
export RLT_SMOKE_STEPS="$steps" RLT_SAVE_INTERVAL="$interval" RLT_VAL_INTERVAL="$interval"
echo 'phase=training' > "$RLT_OUTPUT_ROOT/phase.txt"
bash run_rlt_portable.sh --memory
echo 'phase=completed' > "$RLT_OUTPUT_ROOT/phase.txt"
