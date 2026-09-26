# Interaction Memory: Pilot Results and Acceptance

This record reports the actual 2026-09-26 experiments: GPU rollout, TD updates and checkpoint resume now work, but faster RLT convergence has not been demonstrated. Read the method first, then separate engineering checks from evidence of benefit. [中文](PILOT_RESULTS.zh-CN.md)

## How Memory Extends RLT

The original actor uses the current RL token, joints and reference actions. This module adds completed command/outcome evidence: starting joints, the executed action prefix, observed joint change, duration and ending flags. The next decision reads four recent events and retrieves four from an archive of at most 32, encodes them into a 64-dimensional context, and concatenates it with the original actor/critic inputs.

![Implemented interaction-memory architecture](figures/architecture.svg)

Blue modules are frozen; green networks train; dashed arrows denote training signals. This is the implemented variant, not the original Zeva architecture. Editable [SVG](figures/architecture.svg), publication-layout [PDF](figures/architecture.pdf), and [PNG](figures/architecture.png) are available. Regenerate with `python experiments/maniskill_rlt/draw_architecture.py`.

Critic TD loss updates the reader. The actor receives detached context and keeps its Q+BC objective. Stage 1 and the VLA remain unchanged; no Stage 1B is needed. Replay stores contemporaneous memory snapshots, never hindsight-filled history. Evidence is joint-space control data, not historical vision, tactile/force sensing or inferred contact labels.

The borrowed ideas from [Zeva's official implementation](https://github.com/air-embodied-brain/Zeva) are action/change evidence and multiple memory timescales. Its visual causal tokens, phase/effect objectives, prompt interface, and frozen-network deployment with memory updates are not reproduced here. Our reader learns online through TD; this is neither a Zeva reproduction nor causal identification.

## What Real Training Caught

The first smoke exited successfully, but all 17 reader tensors remained unchanged across checkpoints. Default FSDP parameter flattening prevented the name-based optimizer partition from assigning the reader correctly to the critic. A CPU gradient test did not establish actual distributed optimizer updates.

The experiment overlay now requires `actor.fsdp_config.use_orig_params: True`; validation rejects incompatible settings. A regression exercises production optimizer partitioning. The launcher also compares adjacent checkpoints under `memory_encoder.*` and fails if no weights changed.

| Check | Result |
| --- | --- |
| Fixed real GPU smoke | Global steps 1–4, exit 0 |
| Step 1 versus 4 | 17/17 reader tensors changed; maximum absolute change 0.00057749 |
| Resume from 4 to 6 | Exit 0; step 5→6 changed 17/17 tensors |
| Both optimizer step counters | 8→12; existing state continued |
| CPU regression | 195 passed, 1 skipped, 1 deselected |

Resume restores actor, critic, target, optimizers, schedulers and replay, not an exact simulator/active-memory continuation. The skip is the existing optional Gemma3 check; the deselection is the baseline's incompatible `VideoMetadata` test. Dependencies were not modified to hide it.

GPU work used physical GPU 2 only, with CUDA visibility, RLinf placement and Vulkan PCI binding aligned. Baseline Stage 1 processes on GPUs 0/1 were not stopped or restarted. Each smoke used two training environments, one evaluation environment and 40-control-step episodes: an integration budget, not a success-rate experiment.

## Does History Provide Useful Evidence?

A separate supervised diagnostic predicted ten-tick joint changes from current joints, commands, and the production reader's historical context. It trained a fresh reader, not the online actor, and used no VLA. It does not measure RL success.

The no-memory control had the same architecture with context zeroed. Complete episodes were split between training and validation; seeds 2026/2027/2028 and optimization budgets matched. Snapshots preceded appending the current outcome. A diagnostic shuffled history while keeping current joints and commands fixed.

| Data and budget | Seed | No-memory MSE | Memory MSE | Shuffled-history MSE |
| --- | --- | --- | --- | --- |
| 32 episodes, 300 updates | 2026 | 0.00028224 | 0.00029212 | 0.00179913 |
| Same | 2027 | 0.00027725 | 0.00029476 | 0.00152981 |
| Same | 2028 | 0.00025943 | 0.00025684 | 0.00074894 |
| 64 episodes, 600 updates | 2026 | 0.00010699 | 0.00010731 | 0.00081252 |
| Same | 2027 | 0.00011660 | 0.00011120 | 0.00081112 |
| Same | 2028 | 0.00010306 | 0.00010496 | 0.00038523 |

There is **no consistent benefit across seeds**. Shuffling hurts, showing sensitivity to history, but also creates mismatched inputs; it does not establish useful causal reasoning. Increasing the bounded budget still gave no stable advantage, so the negative results are retained without further tuning to this validation set. Both rounds are exploratory validation, not independent final testing.

A hypothesis is that current joints and commands already explain much of the change in successful demonstrations under one controller. The next experiment should reserve new test episodes and vary hidden dynamics or use failure/recovery data, comparing no memory, recent history and retrieved history. Successful demonstrations alone cannot establish cross-task experience transfer.

## Reproduce and Locate Artifacts

Use explicit `--check` for a read-only preflight; the second command below runs a bounded smoke. `RLT_SMOKE_STEPS` is the stopping global step in 1–20, not additional updates. For resume, set `RLT_SMOKE_RESUME_DIR` to an existing `global_step_N` and increase the stopping step by at least two to audit adjacent new checkpoints.

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory --check
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  RLT_SMOKE_STEPS=4 RLT_SMOKE_RAY_PORT=6402 \
  bash run_rlt_stage2_smoke_gpu2.sh --memory
```

Preflight does not allocate CUDA; the run refuses a busy GPU 2. Cleanup never calls global `ray stop` or removes another job's files. Run the CPU diagnostic with `python -m toolkits.rlt.probe_interaction_memory --dataset DATASET --output NEW_OUTPUT --episodes 64 --steps 600`; the output directory must not exist.

Large artifacts remain outside Git under `/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill`:

| Relative directory | Purpose |
| --- | --- |
| `runs/stage2_smoke/memory_stage2_gpu2_20260926_222443_2996233` | Pre-fix run; reader did not update, not valid learning evidence |
| `runs/stage2_smoke/memory_stage2_gpu2_20260926_223451_3027410` | Fixed four-step smoke |
| `runs/stage2_smoke/memory_stage2_gpu2_20260926_223716_3036341` | Resume from four to six |
| `research/zeva_probe_20260926_2234/results.json` | First diagnostic; episode IDs, data hashes and per-seed metrics |
| `research/zeva_probe_20260926_2238/results.json` | Second diagnostic |

Read the core loop in order: `rlinf/algorithms/rlt/interaction_memory.py` stores/validates evidence, `rlinf/models/embodiment/modules/rlt_memory_encoder.py` reads it, `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` concatenates it, and `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` updates the model. `toolkits/rlt/audit_adapter.py` checks actual changes. Research branches stay separate; nothing is merged into baseline.
