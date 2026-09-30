# ManiSkill RLT baseline and latent-world research

## Run a bounded pilot on another server

After installing the OpenPI/ManiSkill environment in the sibling `UPT_dev/.venv` and making the shared dataset, Stage 1 step-2000 checkpoint, tokenizer and simulator assets available, run this command from this branch:

```bash
bash run_rlt_overnight.sh world 0
```

The second argument is the physical GPU index. The launcher checks the worker's CUDA UUID and resets, renders and steps one RGB environment before loading the policy. It runs a two-iteration integration smoke with an adapter-weight audit, then starts a fresh pilot with 2 train / 4 fixed evaluation environments, 500 control steps, global batch 32, micro batch 8, 512 BC warmup updates and at most 64 updates per iteration. Evaluation, video recording and checkpoints run every 50 iterations. Automatic expert takeover is disabled; the memory run uses the attention reader.

The defaults stop at 1,000 outer iterations or 12 hours, whichever comes first. `RLT_NIGHT_HOURS` accepts 1–24; `RLT_NIGHT_STEPS` accepts multiples of 10 from 20–5,000. `RLT_NIGHT_INTERVAL` defaults to 50, must divide the step budget and must leave at least two checkpoints (for 20 steps, set it to 10). Exit code 124 means the wall-time limit was reached: keep the last periodic checkpoint, as the interrupted update is not saved. A failed probe, smoke or weight audit stops that branch. Logs and resolved configs are written below `$RLT_STORAGE/runs/inspur_world`; override `RLT_OUTPUT_ROOT` to change the destination. These pilots do not establish convergence or replace matched baseline comparisons.

Each job owns its Ray head, port range and physical-GPU lock. Each stage selects a free three-port block instead of reusing the smoke ports. The overnight wrapper ignores `RLT_SMOKE_RAY_PORT`; the standalone portable launcher still accepts an explicit port and rejects occupied ports. Busy GPUs are rejected. The renderer uses the selected GPU's queried PCI address and this host's NVIDIA EGL library. Set `SAPIEN_VULKAN_LIBRARY_PATH` and `RLT_NVIDIA_EGL_LIBRARY` if the loader or driver uses a different path. The launcher prefers `$HOME/.local/rlinf-vulkan/lib/libvulkan.so.1.4.357` when present, otherwise the system loader; it does not copy another host's NVIDIA driver or C++ libraries. The old GPU-2 launcher remains unchanged.

Validation: the September 30 A6000 run passed the CUDA UUID/RGB probe and two-iteration adapter-update smoke. The following pilot failed before Ray startup because the fixed ports were still occupied. The replacement port selection and smoke-to-pilot transition pass CPU regression tests; the corrected long pilot still needs to run on the destination server. Tests also cover occupied head/client/dashboard ports, unsafe budgets and the busy-GPU guard.



See the [2026-09-30 audit](AUDIT_2026-09-30.md) for current changes, diagnostic results and reproducible commands. Earlier design and verification history is retained below.

This directory preserves the reproduction boundary and provides the code-review/runbook entry point for the FLARE-inspired research branch. The [2026-09-26 pilot report](PILOT_RESULTS.md) adds the implemented architecture figure, independent real-data evaluation, GPU smoke and resume evidence; faster RL convergence is not established. [中文](README.zh-CN.md)

## Branches and review order

The initial baseline snapshot was `7db62813`, based on official-source snapshot `b85c07175b10017bf58ab83e1b1eee99666d0626`. Branch `baseline/maniskill-rlt-2026-09-25` now includes resume/isolation fixes through `ff566637`; this research branch cherry-picks that fix as `43a0e615`. Stage 1/2 source and launchers do not establish convergence. The original worktree remains `/home/luokz/rlinf_rlt/UPT_dev`.

The research branch is `research/rlt-flare-latent-dynamics`, in `/home/luokz/rlinf_rlt/UPT_flare_dev`. It shares the existing Python environment but has a separate source tree. Do not switch branches in the active training worktree.

Read [baseline provenance](BASELINE.md), then [algorithm design and ablations](DESIGN.md), and finally [verification and open gates](VERIFICATION.md). The implementation entry points are:

- `toolkits/rlt/cache_latents.py`: immutable episode feature cache.
- `toolkits/rlt/train_latent_world.py`: offline Stage 1B, validation and resume.
- `toolkits/rlt/evaluate_latent_world.py`: per-horizon action/uncertainty diagnostics.
- `rlinf/models/embodiment/modules/rlt_latent_world.py`: future representation model.
- `rlinf/algorithms/rlt/latent_world.py`: executed-chunk replay and provenance guards.
- `toolkits/rlt/preflight.py`: configuration/artifact checks without a training job.

The following generic commands are **for review and later explicit execution**: they use GPU 0 or a two-GPU configuration, so do not execute them while baseline owns GPUs 0/1. For concurrent isolated GPU 2 work, use [the pilot runbook](PILOT_RESULTS.md) instead. A saved intermediate checkpoint is not evidence of convergence.

## Prepare paths without starting a job

Use the research source tree and keep large artifacts on NAS. Choose an actually completed Stage 1 checkpoint; do not point at a file still being written or at `pi05_base`.

```bash
cd /home/luokz/rlinf_rlt/UPT_flare_dev
source /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EMBODIED_PATH="$PWD/examples/embodiment"
export RLT_STORAGE=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill
export RLT_DATASET="$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
export RLT_NORM_STATS="$RLT_DATASET/norm_stats.json"
export RLT_STAGE1_CHECKPOINT="/replace/with/completed/global_step_N/actor"
export RLT_LATENT_CACHE="$RLT_STORAGE/research/latent_cache_stage1_N"
export RLT_WORLD_RUN="$RLT_STORAGE/research/stage1b_seed2026_run01"
export RLT_WORLD_CHECKPOINT="$RLT_WORLD_RUN/best.pt"
export RLT_STAGE2_RUN="$RLT_STORAGE/research/stage2_seed1234_run01"
```

The deliberate placeholder must be replaced after checking the completed checkpoint. `norm_stats.json` must be the exact Stage 1/Stage 2 normalization file; adjust its path if it was saved elsewhere. Use a different cache name for another encoder checkpoint, and a different output directory for each experiment. Do not store venv, sockets or build caches on CIFS.

CPU review checks can run without touching any GPU or Ray session:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/unit_tests/test_models.py tests/unit_tests/test_data.py \
  tests/unit_tests/test_worker.py -k latent_world -q
CUDA_VISIBLE_DEVICES='' python toolkits/rlt/preflight.py --config-only
```

`--config-only` checks composition, not artifact existence or runtime readiness. After Stage 1B produces a checkpoint, omit that flag to hash/compare the actual feature contract. Reading the full Stage 1 weights can take time on NAS; this is intentional and does not allocate the VLA on a GPU.

## Stage 1B: later execution after resources are free

The exporter performs frozen VLA inference on one GPU, then the lightweight trainer uses only cached features. A bounded real-data run is recorded in [pilot results](PILOT_RESULTS.md); the generic commands below require the selected devices to be free.

```bash
CUDA_VISIBLE_DEVICES=0 python toolkits/rlt/cache_latents.py \
  --config experiments/maniskill_rlt/config/cache_latents.yaml --batch-size 2

# Offline W&B is the safe default until the intended account/entity is verified.
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python toolkits/rlt/train_latent_world.py \
  --config experiments/maniskill_rlt/config/stage1b.yaml

CUDA_VISIBLE_DEVICES='' python toolkits/rlt/evaluate_latent_world.py \
  --checkpoint "$RLT_WORLD_CHECKPOINT" --cache-dir "$RLT_LATENT_CACHE"

# Resume only the same run/config/cache after an interruption:
CUDA_VISIBLE_DEVICES=0 python toolkits/rlt/train_latent_world.py \
  --config experiments/maniskill_rlt/config/stage1b.yaml \
  --resume "$RLT_WORLD_RUN/last.pt"
```

Run long commands inside a named tmux session if desired; detach with Ctrl-b, d. For online W&B, authenticate the intended account yourself and set `WANDB_ENTITY` plus `WANDB_MODE=online` before training. Do not commit credentials. The previous baseline's W&B visibility issue is independent of GitHub authentication and is not silently changed by this branch.

Stage 1B writes `best.pt`, `last.pt` and `metrics.jsonl` under `RLT_WORLD_RUN`; its W&B files live there too. It refuses to overwrite a nonempty run without `--resume`. Resume requires identical settings, including `max_steps`; use a new run/config for a different experimental budget. Multiple writers to one cache/run directory are unsupported.

## Stage 2: later integration smoke test, then measured experiments

Stage 2 additionally needs the already verified headless Vulkan environment. Use the same server-local loader/ICD setup; no desktop is required. Preserve the working environment configuration, rather than sourcing the baseline launcher (which starts training).

```bash
export RLINF_VULKAN_PREFIX=/home/luokz/.local/rlinf-vulkan
export PATH="$RLINF_VULKAN_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/lib64"
export SAPIEN_VULKAN_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357"
export VK_DRIVER_FILES="$RLINF_VULKAN_PREFIX/share/vulkan/icd.d/nvidia_headless_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
unset DISPLAY WAYLAND_DISPLAY VK_LAYER_PATH VK_INSTANCE_LAYERS __NV_PRIME_RENDER_OFFLOAD

CUDA_VISIBLE_DEVICES='' python toolkits/rlt/preflight.py

# Only after the baseline has finished and no other task owns this Ray cluster:
CUDA_VISIBLE_DEVICES=0,1 python examples/embodiment/train_embodied_agent.py \
  --config-name maniskill_rlt_stage2_latent_world \
  runner.max_epochs=2 algorithm.rlt_schedule.enable=False \
  algorithm.replay_buffer.min_buffer_size=1 algorithm.update_epoch=1 \
  algorithm.train_actor_steps=2 algorithm.critic_actor_ratio=1 \
  runner.save_interval=1 runner.val_check_interval=1
```

This short run is an integration check, not an experiment: the normal RLT warmup is deliberately disabled. It still collects a rollout and may take time. Verify at least one optimizer update, finite Q/world losses, a saved checkpoint, and resume/rollout weight synchronization. If the critical-phase gate records too few transitions, investigate it instead of declaring the smoke test successful. Keep Ray temporary/socket storage on local disk or `/dev/shm`, with object spilling/checkpoints on NAS; do not stop a shared Ray service or reconnect to the ongoing baseline's cluster just to run this test.

For the actual experiment, use the unmodified research config without the smoke overrides. For its comparator, use `maniskill_rlt_stage2_matched_baseline` and a different `RLT_STAGE2_RUN`. Both disable expert takeover, use the same Stage 1 checkpoint and resource settings, and leave the original schedule intact. Evaluate intervention reduction only after configuring the same real expert checkpoint/takeover rule for both arms. Expert-free success and sample efficiency are the first available endpoints.

The shared `examples/embodiment/config/rlt_research/ac_baseline.yaml` is an intentional frozen copy of baseline settings without primary-only Hydra metadata. This avoids changing the official config and avoids inheriting its `hydra.searchpath` from a non-primary config. Tests compare cache/rollout feature settings and comparator resources.

## Tomorrow's acceptance boundary

Review code/tests and the Stage 1A/1B design before authorizing new jobs. The implementation includes neither explicit long-term memory, contact/force labels, new task gates nor an in-DiT FLARE implementation. No convergence, success-rate gain, intervention reduction or transfer result has been established. The design gives falsifiable experiments and lists the decisions left for the researcher.
