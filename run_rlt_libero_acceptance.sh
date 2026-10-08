#!/usr/bin/env bash
# Run the gated RLT_a campaign in its version-pinned dependency override venv.
set -euo pipefail
RLT_ACCEPTANCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RLINF_VENV=${RLINF_VENV:-"$RLT_ACCEPTANCE_ROOT/../.venv-libero-acceptance"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TMPDIR=${TMPDIR:-/dev/shm}
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$RLINF_VENV/bin/python" ]]; then
    echo "Missing acceptance environment: $RLINF_VENV" >&2
    exit 2
fi
cd "$RLT_ACCEPTANCE_ROOT"
exec "$RLINF_VENV/bin/python" -u -m toolkits.rlt.libero_acceptance_campaign "$@"
