# Action-Outcome Representations: Pilot Results and Acceptance

This record reports the 2026-09-26 real-data and Stage 2 checks: a new residual predictor passed independent-episode evaluation, and both predictor variants completed real GPU smoke runs. Faster online RL convergence is not established. Read the method and correction before interpreting the results and reproduction commands. [中文](PILOT_RESULTS.zh-CN.md)

## What the Added Network Learns

With the Stage 1 VLA frozen, current images, language and joints produce RL token `z_t` and proprioception `p_t`. A small encoder yields `e_t`; a Transformer reads `[e_t, executed action prefix, future query_h]` and predicts the actual token and joint change after `h` control steps. Targets come from real future observations processed by the same frozen VLA, not generated videos.

![Implemented action-outcome architecture](figures/architecture.svg)

This depicts our implementation, not FLARE's original architecture. Blue modules are frozen; green networks train; dashed arrows denote gradients. Editable [SVG](figures/architecture.svg), publication-layout [PDF](figures/architecture.pdf), and [PNG](figures/architecture.png) are included. Regenerate with `python experiments/maniskill_rlt/draw_architecture.py`.

Stage 1B trains on demonstrations, retaining its anchor/BC auxiliary terms. Stage 2 concatenates current `e_t` with actor/critic inputs; critic TD and real-replay future losses continue updating the small module. Actor context is detached. The VLA stays frozen. Online prediction supervision only uses valid nonterminal successors of complete ten-step chunks; reset observations are never future targets.

The idea borrowed from [FLARE's official method](https://research.nvidia.com/labs/gear/flare/) is future-representation supervision of action-related features. FLARE places future tokens inside the action-denoising model and trains jointly with flow matching. Here a frozen-RLT sidecar supplies current features to actor/critic, with future prediction as auxiliary supervision. It generates no video, does not jointly update the VLA, and is not a full FLARE reproduction.

## Why Add Residual Prediction?

The original predictor directly regressed future tokens. Its first validation gave a one-step cosine error of 0.07058, worse than predicting no change at 0.01450. Most information stays unchanged between neighboring frames; reconstructing the entire token consumes the small budget.

With `LatentWorldConfig.predict_residual=True`, prediction becomes:

```text
predicted future token = LayerNorm(current token) + predicted change
```

The change-output layer starts at zero, matching persistence initially; learning then captures deviations caused by execution. The input token is detached, so only the sidecar receives gradients. Joint-change targets, backbone, splits and budgets stay the same. The default remains `False` for old checkpoint compatibility. The flag is stored in the sidecar and read automatically by Stage 2.

## What Independent Episodes Show

All experiments use the frozen Stage 1 step 750 checkpoint. Full episodes 0–11 were cached: 1/6/9 for validation/model selection, the other nine for training. Training used 300 optimizer updates, batch 32, micro-batch 16 and seed 2026. Episodes 12–23 were exported separately for testing only. Cache contracts check the frozen model, normalization, preprocessing and source hashes; evaluation rejects test IDs overlapping the original cache.

Both original and residual variants trained on the same physical L40 GPU 2. Each used its validation-selected `best.pt`; both were evaluated on CPU using the same procedure. An earlier CPU residual run is retained as exploratory evidence, not the matched-backend comparison.

The table reports independent-test future-token cosine error, lower is better. Valid target counts are 905, 857 and 797; frames within an episode are correlated, not independent trials.

| Horizon | Persistence | Original direct prediction | Residual prediction | Residual, shuffled actions |
| --- | --- | --- | --- | --- |
| 1 tick | 0.012456 | 0.074895 | **0.010676** | 0.012794 |
| 5 ticks | 0.071462 | 0.074809 | **0.030102** | 0.084986 |
| 10 ticks | 0.158003 | 0.074135 | **0.045826** | 0.169292 |

These results support residual parameterization under this budget and sensitivity to action inputs. They do not establish physical laws, fewer interventions, higher success, or cross-task transfer. Shuffled actions introduce distribution shift: this is a sensitivity diagnostic, not causal identification. One seed, one task and a small set of successful demonstrations require subsequent multi-seed online comparisons under fixed interaction budgets.

Prediction-head disagreement is not calibrated safety confidence. Original validation ten-tick uncertainty/error correlation was about -0.0017; residual independent-test correlation was about 0.4644, still uncalibrated. Exploration bounds remain disabled; these values do not automatically constrain actions.

## Online Integration and Regression

The original sidecar completed two Stage 2 global steps with 50/54 `latent_world.*` tensors changing. The four unchanged tensors belong to the offline-only BC head, as expected. The residual sidecar also passed the real GPU smoke. The launcher now audits adjacent checkpoints rather than trusting exit codes; its overlay enforces FSDP `use_orig_params=True` for name-based optimizer partitioning.

The residual run also resumed from step 2 to 4 with exit 0. Both optimizer counters advanced from 4 to 8, confirming stateful continuation. The resumed checkpoint-update audit passed.

A two-step smoke with the module disabled on this same branch also exited 0, checking the opt-in off path. This is not a success-rate ablation: short episodes and very few updates cannot compare control quality.

CPU regression: **188 passed, 1 skipped, 1 deselected**, covering residual initialization, gradients, episode selection, independent-test leakage rejection, exact Stage 1B resume, and checkpoint audits. The skip is the existing optional Gemma3 check; the deselection is the baseline's incompatible `VideoMetadata` test. English and Chinese Sphinx builds both had zero warnings. RST trees were unchanged; existing static markup/symbol findings were not repaired in this task.

Only GPU 2 was used, with CUDA, Ray placement and Vulkan PCI binding aligned. Baseline processes on GPUs 0/1 were not stopped or restarted. Each Stage 2 smoke used two training environments, one evaluation environment and 40-control-step episodes. This verifies rollout, replay, updates, weight sync and checkpointing, not success rates.

## Reproduce and Find the Code

Check before explicitly running. Reuse the existing Python environment without dependency changes. `RLT_WORLD_CHECKPOINT` points to the Stage 1B sidecar, not VLA weights; the VLA remains the fixed step 750 checkpoint by default.

```bash
cd /home/luokz/rlinf_rlt/UPT_flare_dev
export RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv
export RLT_WORLD_CHECKPOINT=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill/research/flare_residual_gpu_20260926/stage1b/best.pt
bash run_rlt_stage2_smoke_gpu2.sh --world --check
RLT_SMOKE_RAY_PORT=6412 bash run_rlt_stage2_smoke_gpu2.sh --world
```

Only the last command starts training. A busy GPU 2 is rejected; cleanup never calls global `ray stop`. `RLT_SMOKE_STEPS` is the stopping global step, limited to 1–20. Set `RLT_SMOKE_RESUME_DIR` to an existing `global_step_N` and raise the stopping step by at least two. Resume restores models, optimizers, schedulers, targets and replay, not an exact simulator continuation.

For the first offline pilot, run `python -m toolkits.rlt.run_offline_pilot --output NEW_OUTPUT` with explicit `CUDA_VISIBLE_DEVICES=2` and `RLT_DATASET`, `RLT_NORM_STATS`, `RLT_STAGE1_CHECKPOINT` set. It exports 12 whole episodes and trains 300 steps; output must not exist. `python -m toolkits.rlt.compare_latent_variants --original-config TRAIN_CONFIG --output NEW_OUTPUT --device cuda:0` trains the residual variant on the same cache and budget. Independent evaluation uses `python -m toolkits.rlt.evaluate_latent_world --checkpoint SIDECAR --cache-dir TEST_CACHE --independent-test`.

Read the core code in order: `toolkits/rlt/cache_latents.py` extracts temporal labels; `rlinf/models/embodiment/modules/rlt_latent_world.py` defines the sidecar; `toolkits/rlt/train_latent_world.py` trains it offline; `rlinf/algorithms/rlt/latent_world.py` checks online temporal boundaries; and `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` combines TD/future losses. `toolkits/rlt/audit_adapter.py` checks real updates.

Large artifacts remain outside Git under `/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill`:

| Relative directory | Contents |
| --- | --- |
| `research/flare_pilot_20260926_2230` | Original cache, resolved configs, Stage 1B checkpoints, validation/test JSON |
| `research/flare_residual_20260926_2242` | Exploratory CPU residual run, excluded from matched-device comparison |
| `research/flare_residual_gpu_20260926` | GPU residual run and independent-test JSON |
| `research/flare_test_cache_20260926` | Independent cache, episodes 12–23 |
| `runs/stage2_smoke/stage2_gpu2_20260926_223122_3015384` | Original Stage 2 smoke |
| `runs/stage2_smoke/stage2_gpu2_20260926_224446_3058816` | Residual Stage 2 smoke |
| `runs/stage2_smoke/stage2_gpu2_20260926_224923_3072010` | Residual resume from step two to four |
| `runs/stage2_smoke/stage2_gpu2_20260926_225351_3086762` | Two-step smoke with the sidecar disabled |

The branch remains `research/rlt-flare-latent-dynamics`. Baseline resource-isolation/resume fix `ff566637` was cherry-picked as `43a0e615`; Zeva/VR were not merged, and the active baseline worktree was not modified.
