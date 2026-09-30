# RLT Atomic Decisions verification

This record separates verified CPU behavior from pending robot validation for `research/rlt-jev-atomic-decisions`, based on `d9ba471e`. Development used `/home/luokz/rlinf_rlt/UPT_jev_dev` and the existing baseline virtual environment. No new GPU process, Ray cluster, VLA inference or ManiSkill training was launched. No shared dependencies were installed or modified.

## 2026-09-30 bounded experiment queue

The user authorized GPU 2 experiments after the initial delivery. GPU 2 was occupied by another user's compute jobs at preparation time (about 14.8 GiB); no process was terminated or occupancy threshold relaxed. `toolkits.rlt.atomic_experiment` now supports waiting for release, running the two-step smoke with the UUID probe, inspecting finite learner metrics and final CPU-loadable atomic weights, and then starting a fresh 20-step pilot. Each phase uses the original launcher protection. A changed or dirty checkout, timeout, failed run or invalid report stops the queue.

The pilot keeps the same candidate bank and frozen Stage 1 weights, uses 2 train / 4 fixed-seed eval environments, micro/global batch 8/32, and up to 16 learner updates per outer step. The first 32 updates give Q zero weight in selector learning; Q weight then ramps over 128 updates. Evaluations, videos and checkpoints occur every five outer steps. These are diagnostic budgets, not a convergence prescription or matched baseline comparison.

Updated CPU regression: **239 passed, 4 skipped, 1 deselected**, including 33 atomic component/workflow cases. New coverage checks busy/unreadable GPU rejection, inherited pilot settings, finite update gating and reading actual TensorBoard event/checkpoint files in a temporary test directory. The same pre-existing video metadata dependency test remains excluded. Both launcher profiles pass read-only preflight; shell syntax, Ruff and whitespace checks pass. Regression testing caught and fixed a Hydra primary-searchpath inheritance error and a temporary TensorFlow import block leaking into unrelated model tests.

At this revision, GPU training results are still pending resource availability. Per-experiment `status.json` and per-phase `summary.json` record the actual outcome; queued or completed CPU tests must not be presented as successful GPU training. Checkpoint resume is not part of this queue.

## Checked implementation

The candidate bank preserves the clipped reference, bounds corrections, preserves gripper commands, masks exact duplicates and rejects non-finite observations. Actor sampling selects an intact candidate; metadata explicitly describes a student proposal, which routing may override.

Tests cover selector-only gradients, actual-intervention BC, detached candidate targets, exact categorical target expectations, reward horizons of one and two, terminal masking, optimizer partitioning, real replay round-trip, saturated-selector recovery, model/optimizer save-load and incompatible candidate radius rejection. The real worker update method is also exercised on CPU across two updates with two microbatches each: clipping sees only the active optimizer's gradients, not stale gradients from the other phase. Disabled factory construction has the same parameter keys and bitwise values as baseline under the same seed.

The smoke config composes with Hydra, validates model/controller contracts, and pins actor/env/rollout to physical rank 2. `use_orig_params=True` preserves actor/critic optimizer ownership. The launcher defaults to read-only preflight; explicit `--probe` or `--run` is required for GPU use. It checks GPU memory and compute processes, uses a cooperative lock and isolated Ray ports, and never runs broad `ray stop` or process-name cleanup.

## CPU regression results

With `CUDA_VISIBLE_DEVICES=''`, one-thread BLAS and the research worktree on `PYTHONPATH`:

```bash
python -m pytest tests/unit_tests/test_atomic_decision.py \
  tests/unit_tests/test_models.py tests/unit_tests/test_data.py tests/unit_tests/test_utils.py \
  -k 'not test_vlm_trend_batch_video_metadata_stays_nested_per_sample' -q
```

Result: **229 passed, 4 skipped, 1 deselected**, including **23 new component tests**. The excluded metadata test was run independently in the unchanged baseline and also failed: installed `transformers.video_utils.VideoMetadata` rejects `frames_indices`. Its dependencies were not changed to hide that failure. An earlier full-suite run also exposed missing `PYTHONPATH` in subprocess tests; setting the documented worktree path resolved those failures.

`bash -n run_rlt_atomic_gpu2.sh`, `bash run_rlt_atomic_gpu2.sh --check`, Ruff checks/formatting and `git diff --check` pass. Preflight checks file existence and normalization JSON; it does not deserialize or validate the Stage 1 model's numerical quality.

## Synthetic learning diagnostic

The CPU tool uses a one-step quadratic reward with a desired signed joint correction encoded in a small synthetic observation. It collects 1024 uniformly sampled action transitions, trains for 1000 critic and selector updates, and evaluates 256 independent contexts per seed. Only sampled-action rewards train the critic. It contains no simulator dynamics, contact sensor, image input or scarce exploration reward.

| Update | Seeds | Greedy critic selects optimum | Selector selects optimum | Mean reward |
| --- | --- | --- | --- | --- |
| Initial selector | 0, 1, 2 | Random initialization | 0% | −1 |
| Prototype expected-cost gradient | 0, 1, 2 | 100% each | 0% each | −2 |
| Detached improvement distillation | 0, 1, 2 | 100% each | 100% each | 0 |

Each paired run uses identical generated data and critic updates; matching final critic losses provide an additional consistency check. This diagnoses selector saturation in the prototype, not universal superiority of distillation. The failed 200-update exploratory check led to this longer paired test. Raw successful and failed paired reports are preserved in [atomic_cpu_results.json](atomic_cpu_results.json), and both updates remain reproducible in the CPU tool:

```bash
python -m toolkits.rlt.atomic_cpu_smoke --steps 1000 --actor-update expected-cost
python -m toolkits.rlt.atomic_cpu_smoke --steps 1000 --actor-update distill
```

The diagnostic took roughly 18 seconds per seed with one CPU thread on this machine. That is not a robot rollout throughput claim. Production exposes only the revised distillation objective.

## Documentation checks

The `docs-check` harness built the unchanged EN and ZH Sphinx trees with zero build warnings. The separate repository-wide markup and symbol scanners still flag existing issues outside this change, including 26 unknown/example symbol occurrences; those unrelated documents were not rewritten. The new bilingual research guides were manually cross-checked against code, config, paths and commands. Their SVG is a code-native architecture diagram; it does not depict measured speed or success.

## Required next validation

1. Schedule an idle GPU 2, then explicitly run the placement probe. No GPU smoke was performed during this delivery.
2. Run the two-iteration ManiSkill smoke. Require successful feature loading, finite Q/actor updates, candidate metadata transport, checkpoint save/load and unchanged GPU 0/1 processes.
3. Inspect complete videos and candidate distributions. Check reference clipping, intervention expressibility, gripper limitations and any accumulated joint motion.
4. Run matched multi-seed baselines before claiming sample efficiency, intervention reduction or cross-task transfer. Include a continuous residual control restricted to the same radius.

GPU/FSDP checkpoint resume, online robot learning, wall-clock latency, physical safety, transfer and VR integration remain unverified. In particular, unchanged Stage 1 does not make this branch equivalent to the baseline Stage 2 actor or compatible with its actor checkpoints.
