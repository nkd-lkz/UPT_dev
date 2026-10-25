#!/usr/bin/env bash

set -euo pipefail

readonly PYTHON_BIN="${PYTHON_BIN:-/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python}"
# Hugging Face's local-dir downloader uses POSIX locks and chmod.  Downloading
# directly to the CIFS NAS left only metadata after a transient Xet failure, so
# keep the active, resumable copy on the local data disk.  A validated copy can
# be archived to ROBOTWIN2_RLT_ARCHIVE_ROOT after the smoke test.
readonly ASSET_ROOT="${ROBOTWIN2_RLT_ROOT:-/mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025}"
readonly ARCHIVE_ROOT="${ROBOTWIN2_RLT_ARCHIVE_ROOT:-/mnt/nas_ailab_434/Personal_File/luokz/robotwin2_rlt_20261025}"
readonly MODEL_REVISION="fa8df6ed103db0f5549c122f3a17c00ba6426c98"
readonly DATA_REVISION="c9944782f5473feef33d98af6cc033e23acfd875"
readonly ROBOTWIN_REVISION="0008ae6800df9f75fc8de7098bacb01735fd8fd2"
readonly ROBOTWIN_ASSETS_REVISION="3dc3b798668feb99ac61cc9086d84cbcc3d79186"

mkdir -p "${ASSET_ROOT}/models" "${ASSET_ROOT}/datasets" "${ASSET_ROOT}/repos" "${ASSET_ROOT}/logs"

# RoboTwin needs mplib 0.2.1, but the shared RLinf venv keeps 0.1.1 for
# ManiSkill. Install the benchmark-specific binary wheel into a PYTHONPATH
# overlay so setup does not break concurrent projects.
if [[ ! -f "${ASSET_ROOT}/python_deps/mplib/sapien_utils/__init__.py" ]]; then
  "${PYTHON_BIN}" -m pip install --no-deps --target "${ASSET_ROOT}/python_deps" \
    'mplib==0.2.1'
fi

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${ASSET_ROOT}/cache/huggingface"

download_snapshot() {
  local repo_id="$1"
  local repo_type="$2"
  local revision="$3"
  local target="$4"
  local attempt

  for attempt in 1 2 3; do
    if "${PYTHON_BIN}" -m huggingface_hub.commands.huggingface_cli download \
      "${repo_id}" \
      --repo-type "${repo_type}" \
      --revision "${revision}" \
      --max-workers 3 \
      --local-dir "${target}"; then
      return 0
    fi
    printf 'Download attempt %d/3 failed for %s; resuming.\n' "${attempt}" "${repo_id}" >&2
  done
  return 1
}

download_snapshot \
  RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  model \
  "${MODEL_REVISION}" \
  "${ASSET_ROOT}/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"

"${PYTHON_BIN}" "$(dirname "$0")/merge_safetensor_shards.py" \
  "${ASSET_ROOT}/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"

download_snapshot \
  RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data \
  dataset \
  "${DATA_REVISION}" \
  "${ASSET_ROOT}/datasets/robotwin-adjust-bottle-clean50"

if [[ ! -d "${ASSET_ROOT}/repos/RoboTwin/.git" ]]; then
  git clone --branch RLinf_support --single-branch \
    https://github.com/RoboTwin-Platform/RoboTwin.git \
    "${ASSET_ROOT}/repos/RoboTwin"
fi

git -C "${ASSET_ROOT}/repos/RoboTwin" checkout --detach "${ROBOTWIN_REVISION}"

if [[ ! -f "${ASSET_ROOT}/repos/RoboTwin/assets/objects/objaverse/list.json" ]]; then
  # The clean baseline never reads randomized background textures. The
  # upstream all-assets helper also downloads that 10.97 GiB archive, so fetch
  # only the embodiments and objects required by clean ALOHA evaluation.
  "${PYTHON_BIN}" -m huggingface_hub.commands.huggingface_cli download \
    TianxingChen/RoboTwin2.0 \
    embodiments.zip objects.zip \
    --repo-type dataset \
    --revision "${ROBOTWIN_ASSETS_REVISION}" \
    --max-workers 2 \
    --local-dir "${ASSET_ROOT}/repos/RoboTwin/assets"
  (
    cd "${ASSET_ROOT}/repos/RoboTwin/assets"
    unzip -q -o embodiments.zip
    unzip -q -o objects.zip
  )
  (
    cd "${ASSET_ROOT}/repos/RoboTwin"
    "${PYTHON_BIN}" script/update_embodiment_config_path.py
  )
fi

# This host has the NVIDIA EGL vendor library but lacks the small GLVND
# dispatcher package (libEGL.so.1). Extract it locally when system packages
# cannot be installed without sudo; run_robotwin2.sh adds it to the loader path.
if ! ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
  egl_root="${ASSET_ROOT}/system_libs"
  mkdir -p "${egl_root}"
  if [[ ! -f "${egl_root}/root/usr/lib/x86_64-linux-gnu/libEGL.so.1" ]]; then
    (
      cd "${egl_root}"
      apt-get download libegl1
      dpkg-deb -x libegl1_*_amd64.deb root
    )
  fi
fi

printf 'Model: %s\nDataset: %s\nRoboTwin: %s\n' \
  "${MODEL_REVISION}" "${DATA_REVISION}" "${ROBOTWIN_REVISION}"
printf 'Active root: %s\nArchive root (not copied yet): %s\n' \
  "${ASSET_ROOT}" "${ARCHIVE_ROOT}"
