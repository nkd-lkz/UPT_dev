# Validate Baseline Stage 2 on GPU 2

Use this smoke test to check Stage 2 checkpoint loading, simulation, replay, actor–critic updates, evaluation, and checkpoint saving on physical GPU 2 while Stage 1 occupies GPUs 0/1. Run the read-only preflight, launch in tmux, then inspect learner metrics and output files. This tests the integration, not convergence or generalization.

## Configuration and Weights

The standalone config is `examples/embodiment/config/maniskill_rlt_stage2_smoke_gpu2.yaml`; launch it with `run_rlt_stage2_smoke_gpu2.sh` from the repository root. The original Stage 1 and Stage 2 configs remain unchanged. This smoke does not include FLARE modules.

The default checkpoint is `runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750/actor/model_state_dict/full_weights.pt`, relative to `/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill`. Step 750 is a saved checkpoint, not the current training step. Normalization uses `datasets/lerobot/maniskill_peginsertionside_joint/norm_stats.json` under the same root without recomputing statistics.

Stage 1 weights initialize the frozen `rollout.rlt_feature_model`. The small Stage 2 MLP actor and twin-Q start from scratch; do not set `actor.model.model_path` to Stage 1. The smoke budget is:

| Setting | Smoke Value |
| --- | --- |
| GPU | `CUDA_VISIBLE_DEVICES=2`; RLinf hardware rank `2-2`; CUDA ordinal 0 inside the worker; renderer pinned to GPU 2 by PCI address |
| Train / eval environments | 2 / 1 |
| Control steps per rollout / action chunk | 40 / 10 |
| Outer iterations | 2; evaluate and save every iteration |
| Global / micro batch | 4 / 2; accumulate 2 micro batches on one GPU |
| Minimum replay size | 4; up to 8 chunk transitions in the first rollout |
| Update budget | 2 initial warmup updates; at most 2 per iteration; actor/critic ratio 1:1 |
| Logging | Local TensorBoard and text logs |

The smoke forces critical-phase control with `trigger_mode: always_on`, so collection does not depend on an early checkpoint already grasping the peg. The baseline warmup logic still selects reference actions during the first rollout; the MLP actor can take over after updates. Expert intervention is disabled and `rollout.expert_model: null` prevents a second OpenPI from loading. Sensors remain 384×384, with baseline action and proprioception dimensions. These settings test integration and are not a full-task evaluation protocol. A zero success rate alone does not fail the smoke.

## Check and Launch

Validate checkpoint, normalization and Vulkan paths together with the Hydra config first. This preflight reads files without starting Ray, allocating GPU memory or creating output directories:

```bash
cd /home/luokz/rlinf_rlt/UPT_dev
bash run_rlt_stage2_smoke_gpu2.sh --check
```

After `Preflight OK`, start the actual test in tmux:

```bash
tmux new-session -s rlt_stage2_smoke
cd /home/luokz/rlinf_rlt/UPT_dev
bash run_rlt_stage2_smoke_gpu2.sh
```

The launcher activates `.venv` and sets the previously verified headless Vulkan paths. Detach with `Ctrl-b`, then `d`; return with `tmux attach -t rlt_stage2_smoke`. Launch refuses if GPU 2 uses more than 1024 MiB. This check does not reserve the GPU against other processes; do not launch another GPU 2 job concurrently.

Ray defaults to GCS 6382, client 6383 and dashboard 6384 on 127.0.0.1; agent and worker ports are assigned automatically to avoid Stage 1's defaults. The resource budget is 8 CPUs and a 2 GiB object store. Sockets and temporary files use a private `/dev/shm/rlt2.*` directory; spills and training output use NAS. An explicit `RAY_ADDRESS` selects this head, not Stage 1. Cleanup signals only this launcher's training process and Ray head. Do not use global `ray stop` or `pkill ray`. CPU, memory bandwidth and NAS throughput remain shared, so Stage 1 may slow slightly.

To select another completely saved checkpoint, set its actor directory before launch. The script does not follow the latest checkpoint automatically:

```bash
export RLT_STAGE1_ACTOR=/absolute/path/to/checkpoints/global_step_1000/actor
bash run_rlt_stage2_smoke_gpu2.sh --check
```

Replace the placeholder with an existing directory. If the default port is occupied, set `RLT_SMOKE_RAY_PORT=6390` and reserve its next two ports too. The launcher rejects Stage 1 port 6379 and recovery ports 6385–6387. `RLT_STORAGE`, `RLT_DATASET_DIR` and `RLINF_VULKAN_PREFIX` also accept environment overrides.

## Logs and Acceptance

The launcher prints a run directory under `$RLT_STORAGE/runs/stage2_smoke/stage2_gpu2_<timestamp>_<PID>/`. It contains `train.log`, `ray-head.log`, `resolved-config.yaml` and `tensorboard/`. Small actor–critic checkpoints go under `stage2_smoke/checkpoints/global_step_<N>/actor/`. Ray diagnostics remain in the printed `/dev/shm/rlt2.*` directory; the script does not delete diagnostic data automatically.

Watch the actual path printed by the launcher from another terminal:

```bash
tail -f /absolute/path/to/the/printed/run/directory/train.log
```

A passing smoke completes both iterations with exit code 0, collects replay samples, reports positive `train/rlt/critic_updates_run` and `train/rlt/actor_updates_run`, has finite losses, completes evaluation, and saves at least one Stage 2 checkpoint. A live process, GPU allocation, or evaluation output alone does not establish success. W&B is disabled by default to avoid account-permission failures; TensorBoard holds the metrics.

The GPU placement probe has passed; full simulation and training results are recorded below when available. After smoke, return to the full baseline config, restore `auto` phase switching and the training/evaluation budgets, and select an appropriate Stage 1 checkpoint. Do not treat an expanded smoke budget as a convergence experiment.

## GPU Isolation and Recovery Record

The first GPU smoke on 2026-09-26 exposed an incorrect placement assumption: RLinf independently enumerates all three physical GPUs, so `0-0` does not follow the outer `CUDA_VISIBLE_DEVICES=2` remapping. Workers entered GPU 0, causing smoke and Stage 1 OOM. Stage 1 last reported step 889; the latest saved checkpoint was 750. Disabled dashboard also broke the failure handler's State API call. CPU config checks did not establish GPU isolation.

The corrected launcher resolves all three placements and starts a real RLinf worker. Before allocating one scalar, it checks visibility is `2` and the CUDA UUID matches physical GPU 2. This gate runs before every training launch, or independently with:

```bash
bash run_rlt_stage2_smoke_gpu2.sh --probe
```

Require `GPU_PROBE_OK` and exit code 0. This hardware test passed on 2026-09-26 with GPU 2 UUID `4662787b-485a-0e8f-e4b2-dd47352ed69c`. It does not replace full simulation and training smoke.

Stage 1 uses `run_rlt_stage1_resume750.sh` to restore original step 750 into a new timestamped run, with Ray ports 6385–6387. DCP restores model, optimizer and scheduler state. The OpenPI iterator was not separately saved in this checkpoint, so bit-identical data-order continuation is not claimed. Startup refuses occupied GPUs 0/1; do not start recovery twice.

The first recovery attempt failed while copying Adam states to CUDA, before any new training step. DCP had initialized GPU optimizer tensors to build its load template, and `Optimizer.load_state_dict` allocated replacements before releasing those tensors. `Checkpoint.load_state_dict` now moves the existing optimizer state to CPU when `cpu_offload=True` before installing the checkpoint state. It preserves initialized entries rather than clearing them, which would trigger another synthetic optimizer initialization. Model weights, Adam moments and counters, scheduler state, and saved RNG state retain their checkpoint values. This changes restore-time memory placement, not the training algorithm or batch size; the `local_shard` path is unchanged.

`tests/unit_tests/test_checkpoint.py` covers single/multiple optimizer round trips, exact next-update parity on CPU, CUDA restore peak allocation, and a real FSDP `NO_SHARD` DCP round trip followed by an update. Four tests passed on GPU 2 on 2026-09-26. CPU-only execution skips the two CUDA tests. The recovery launcher verifies the personal W&B login `c6522513`, pins entity `c6522513-sustech`, and defaults HTTPS traffic to the user's existing localhost proxy on port 17891 while excluding local Ray endpoints. Keep that proxy available, or provide a working `HTTPS_PROXY` before launching. These host-specific settings belong to this launcher, not the shared model config.

## Verified Run on 2026-09-26

Stage 1 resumed from 750 and completed steps 751–754 during verification, with finite losses and gradients, on GPUs 0/1. The run is `runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016` under the storage root, in tmux `rlt_stage1_resume750_fixed:0`. Its W&B run is `c6522513-sustech/rlinf-rlt/8oblhtio`. Step 751 reported loss 0.447 and step 754 reported loss 0.449. The next scheduled checkpoint is step 1000; the verification did not wait for that save or for convergence.

Only after observing step 751, the full GPU 2 smoke was launched in tmux `rlt_stage2_smoke_verified`. Run `runs/stage2_smoke/stage2_gpu2_20260926_154350_2095986` completed two iterations with exit code 0. TensorBoard confirms:

| Metric | Iteration 1 | Iteration 2 |
| --- | --- | --- |
| Actor / critic updates | 2 / 2 | 2 / 2 |
| Replay size | 6 | 12 |
| Actor loss | 0.430630 | 0.711137 |
| Critic loss | 0.116520 | 0.010061 |
| Evaluation success | 0 | 0 |

Every logged scalar was finite. Both evaluations completed; checkpoints `global_step_1` and `global_step_2` contain nonempty `actor/model_state_dict/full_weights.pt`, DCP metadata, and a DCP shard. All three smoke components were observed on physical GPU 2. After smoke cleanup, only the two Stage 1 GPU processes remained, and Stage 1 had advanced to 754. This validates the engineering path, not task mastery; expert intervention was disabled. TensorBoard steps are zero-based (0/1), while checkpoint directory steps are one-based (1/2).

Validation also passed Ruff, shell syntax checks, 37 CPU tests (2 CUDA tests skipped and 2 unrelated Ray-starting tests excluded), and both Sphinx builds with zero build warnings. The repository-wide documentation scans still report 61 pre-existing inline-markup warnings and 26 unresolved names in unchanged RST pages; these are outside this recovery change.
