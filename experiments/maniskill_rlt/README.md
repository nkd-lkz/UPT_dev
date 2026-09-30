# Inspect the RLT Interaction-Memory Branch

## Run a bounded pilot on another server

After installing the OpenPI/ManiSkill environment in the sibling `UPT_dev/.venv` and making the shared dataset, Stage 1 step-2000 checkpoint, tokenizer and simulator assets available, run this command from this branch:

```bash
bash run_rlt_overnight.sh memory 1
```

The second argument is the physical GPU index. The launcher checks the worker's CUDA UUID and resets, renders and steps one RGB environment before loading the policy. It runs a two-iteration integration smoke with an adapter-weight audit, then starts a fresh pilot with 2 train / 4 fixed evaluation environments, 500 control steps, global batch 32, micro batch 8, 512 BC warmup updates and at most 64 updates per iteration. Evaluation, video recording and checkpoints run every 10 iterations. Automatic expert takeover is disabled; the memory run uses the attention reader.

The defaults stop at 1,000 outer iterations or 12 hours, whichever comes first. `RLT_NIGHT_HOURS` accepts 1–24; `RLT_NIGHT_STEPS` accepts multiples of 10 from 20–5,000. Exit code 124 means the wall-time limit was reached: keep the last periodic checkpoint, as the interrupted update is not saved. A failed probe, smoke or weight audit stops that branch. Logs and resolved configs are written below `$RLT_STORAGE/runs/inspur_memory`; override `RLT_OUTPUT_ROOT` to change the destination. These pilots do not establish convergence or replace matched baseline comparisons.

Each job owns its Ray head, port range and physical-GPU lock. Busy GPUs are rejected. The renderer uses the selected GPU's queried PCI address and this host's NVIDIA EGL library. Set `SAPIEN_VULKAN_LIBRARY_PATH` and `RLT_NVIDIA_EGL_LIBRARY` if the loader or driver uses a different path. The launcher prefers `$HOME/.local/rlinf-vulkan/lib/libvulkan.so.1.4.357` when present, otherwise the system loader; it does not copy another host's NVIDIA driver or C++ libraries. The old GPU-2 launcher remains unchanged.

Validation: CPU configuration tests cover physical GPUs 0/1/2 and budget rejection; read-only preflight passed using the shared inputs. Actual A6000 CUDA, Ray and rendering validation runs on the destination server as part of the launcher.



See the [2026-09-30 audit](AUDIT_2026-09-30.md) for current changes, diagnostic results and reproducible commands. Earlier design and verification history is retained below.

Use this entry point to inspect the independent memory branch without starting training. Read the [design and code map](DESIGN.md), then run CPU tests and read-only preflight. The [latest pilot results](PILOT_RESULTS.md) include the architecture figure, actual GPU updates, resume and negative predictive-probe results; [verification](VERIFICATION.md) preserves the initial implementation record.

Branch: `research/rlt-zeva-interaction-memory`. Baseline: `ff56663769fd00f6108c39195888f5d55cb8a737`. Stage 1, FLARE, and VR work remain separate.

## Checks Without Training

Reuse the existing Python for synthetic unit inputs and a fake simulator boundary. Gradient tests check backpropagation without creating training checkpoints.

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m pytest \
  tests/unit_tests/test_interaction_memory.py -q
```

Read-only preflight composes Hydra and checks Stage 1 weights, normalization, and Vulkan file paths. It does not deserialize weights, start Ray, allocate a GPU, or create a training directory. GPU 2 in its output describes a future launch, not observed GPU execution.

```bash
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory --check
```

Expect `Interaction memory: True` and `Preflight OK`. Set `RLT_STAGE1_ACTOR` for another fully saved Stage 1 actor directory; the inherited default is step 750.

## Smoke Only After Separate Authorization

The following command was not executed for this delivery. Once GPU 2 use is approved, the launcher retains the existing occupancy checks, private Ray cluster, and physical GPU UUID probe before two bounded update iterations. It is not a convergence experiment.

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory
```

Check GPU placement, finite losses, replay memory fields, and checkpoint save/load before larger experiments. Output is under NAS `runs/stage2_smoke/memory_stage2_gpu2_<time>_<PID>/`, with experiment name `stage2_memory_smoke`. Omitting `--memory` preserves baseline smoke behavior.
