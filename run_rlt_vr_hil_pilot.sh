#!/usr/bin/env bash
# One local Windows environment + baseline RLT h=1 learner on physical GPU 2.
set -euo pipefail
rlt_vr_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rlt_vr_mode=${1:-check}
if [[ "$rlt_vr_mode" != check && "$rlt_vr_mode" != run ]]; then
  echo 'Usage: bash run_rlt_vr_hil_pilot.sh check|run [--resume /path/to/learner.pt]'
  exit 2
fi
if (( $# )); then shift; fi
if [[ "$rlt_vr_mode" == run && -z "${RLT_VR_TOKEN:-}" ]]; then
  read -rsp 'Connection secret (at least 32 characters, reuse on Windows): ' RLT_VR_TOKEN
  printf '\n'
  export RLT_VR_TOKEN
fi
exec bash "$rlt_vr_root/run_rlt_vr_online_gpu2.sh" "$rlt_vr_mode" \
  --config "$rlt_vr_root/toolkits/rlt_vr/online_pilot.yaml" "$@"
