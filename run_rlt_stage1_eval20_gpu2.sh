#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$RLINF_ROOT/toolkits/rlt/run_baseline_gpu2_job.sh" stage1-eval "$@"
