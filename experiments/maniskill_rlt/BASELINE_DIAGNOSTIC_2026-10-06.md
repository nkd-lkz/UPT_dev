# Diagnose RLT imitation and Q-guided action changes

This protocol compares BC-only and Q+BC on the same saved transitions before another online memory experiment. It uses the production small head and loss functions on CPU. Simulator evaluation follows when a GPU is available. The campaign manifest records actual progress and results.

## Continue after the baseline evaluation

The October 6 evening campaign extends GPU waiting to 24 hours and limits actual work to six hours per GPU. `toolkits/rlt/run_research_queue.py` keeps atomic job status, owned child-process timeouts, pinned checkpoint hashes and complete frozen-evaluation checks. Completed jobs can be resumed without repetition; failed or interrupted jobs require inspection before retrying. Independent detached checkouts retain the original baseline revision and the new diagnostic revision. An archive without Git metadata is unsuitable for the existing launcher, which calls `git rev-parse` after preflight.

After all 16 baseline runs pass the episode, weight, route and paired-initial-state checks, each GPU runs a small simulator smoke before its diagnostic. These checks establish usable evidence, not baseline convergence. They authorize response diagnosis only; no new RL or expert training starts automatically.

| GPU | First | After baseline validation and smoke |
|---|---|---|
| 0 | BC-only and reference-only, four seeds each | 56 paired response trials, four queries per condition |
| 1 | Q+BC and old zero head, four seeds each | Six paired stationary / A-B-A response streams |

The response trial collects eight real ten-tick command blocks under each arm stiffness, then restores the same seeded scene for each query while retaining calibration history. This is a privileged diagnostic reset, not a normal deployment reset. Full simulator-state hashes, current qpos, velocity and query commands must agree across conditions. Wrong history comes from the paired condition with the same historical commands and valid slots. The hidden stiffness label is never a predictor input. Pairs 0–31, 32–39 and 40–55 form training, validation and test partitions. Three seeds train equal-size no-history / fixed-response heads for 512 updates with identical initialization and sampling. Test uses the final update only. An engineering smoke uses two pairs and does not fit a model.

`rlinf/algorithms/rlt/response_context.py` implements four diagnostic readers: clear at every declared attempt start, retain, time decay, and error weighting against the last four completed responses. All retain at most 32 records. Error weights can recover, but recent agreement is not calibrated confidence. The module is not wired into the production actor. Real A-B-A and stationary controls share attempt boundaries; condition IDs and future outcomes stay outside the reader.

The first synthetic run uses a linear plant, six seeds and fixed thresholds. It is not robot simulation. At return to A, first-four-block MSE is 0.000389 for retention versus 0.000696 for error weighting. This counterexample does not justify promoting error weighting into RL. The simulator probe may reveal its limits, not automatically endorse it.

Run the CPU diagnostic with the existing environment and a new output path:

```bash
CUDA_VISIBLE_DEVICES='' "$RLINF_VENV/bin/python" -m toolkits.rlt.probe_memory_conditions synthetic --output "$CAMPAIGN/synthetic"
```

Once a physical GPU is free, invoke the same module with `matched --gpu 0 --output "$CAMPAIGN/matched"` or `shift --gpu 1 --output "$CAMPAIGN/shift"`. Remove the CPU example's `CUDA_VISIBLE_DEVICES=''`; the tool isolates its selected GPU. Add `--smoke` for integration validation. The tool takes the shared project GPU lease and rechecks memory/processes before simulator creation. Recorded units are raw joint-displacement squared error. None of these prediction diagnostics measure task success, correction savings or transfer.

## Why this diagnostic comes first

The October 4 frozen comparison found 16/64 successes for zero context, 15/64 for response memory, and 19/64 for reference-only. Only 26/64 episodes reached the learned actor. A memory condition at that gate cannot directly fix the other episodes. This diagnostic separates reference imitation error from the effect of the actor's Q term. It does not establish memory utility.

## Make an immutable development cache

Preparation uses only present, indexed simulator files, records missing counts and source hashes, and leaves the checkpoint unchanged. It rejects nonfinite data, unrecorded transitions, inconsistent terminal flags and intervention data. The October 3 checkpoint index contains more records than were saved.

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$RLINF_VENV/bin/python" \
  toolkits/rlt/diagnose_actor_learning.py prepare \
  --replay "$SOURCE_REPLAY" --output "$CAMPAIGN/cache" \
  --limit 2048 --split-seed 601
```

`SOURCE_REPLAY` is the zero-context checkpoint 275 directory `actor/sac_components/replay_buffer/rank_0`. Set `CAMPAIGN` to a new output directory and `RLINF_VENV` to the existing compatible environment. The cache retains current/next frozen features, reference actions, executed actions, rewards, terminal flags and decision-time memory. It does not run a VLA or simulator.

The split holds out one fifth of the collection model versions. Adjacent chunks are not split independently. This is a development split, not proof of unseen physical conditions. Old checkpoints already trained on these records; only fresh heads have a held-out split.

## Run the matched learning comparison

Use the resolved zero-context training config. Both arms start fresh with the same seed, architecture, data, sample sequence, microbatch size, critic updates and actor updates. Both train a critic. The only intended intervention is zeroing the actor's Q coefficient in BC-only; BC and reference dropout remain unchanged.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$RLINF_VENV/bin/python" \
  toolkits/rlt/diagnose_actor_learning.py train \
  --cache "$CAMPAIGN/cache" --config "$CAMPAIGN/source-config.yaml" \
  --output "$CAMPAIGN/training" --steps 2048 --seed 1234 --threads 2
```

The pilot config uses batch 32, microbatch 8 and one critic update per actor update: 2,048 critic updates and 2,048 actor updates per arm. The resolved pilot overrides the base config's 4:1 ratio. Q weight is zero for 512 critic updates, ramps to 0.05 over the next 512, then stays at 0.05. BC weight stays at 7.0. The arms should remain identical during the common zero-Q warmup.

The tool calls `RLTACLossMixin`'s production losses, bypassing only worker timing decorators. It preserves optimizer ownership, Adam defaults, stochastic sampling, microbatch ordering, whole-model gradient clipping and target EMA. It does not reproduce FSDP, growing replay or original optimizer state.

Reports contain deterministic reference MSE/MAE, action saturation, twin-Q disagreement and Q1 preference between actor and reference on the same states. Q preference is not a measured counterfactual return. Checkpoints every 256 critic updates are diagnostic records; final step 2048 is selected in advance for control evaluation.

## Evaluate control after resources are available

Two bounded queues each wait at most 12 hours for an unused GPU and permit at most four hours of evaluation runtime. `wait_for_gpu.py` requires both low memory use and no compute process. The portable launcher then acquires its lease and rechecks memory. Existing processes are not stopped; an acquisition race fails closed.

GPU 0 evaluates BC-only and reference-only. GPU 1 evaluates Q+BC and the old zero-context checkpoint as a descriptive anchor. Fixed seeds 4101–4104 each use 16 lanes and 500 ticks, giving 64 episodes per arm. No expert is used; context is zero and weights are frozen. Only BC-only versus Q+BC is a matched fresh-training comparison. The old checkpoint has a different training history and budget.

The campaign launchers call `run_rlt_portable.sh --memory --eval` with `RLT_MEMORY_READER=zero`, `RLT_EVAL_CHECKPOINT`, `RLT_EVAL_VARIANT`, `RLT_EVAL_SEED`, `RLT_EVAL_ENVS=16` and `RLT_EPISODE_STEPS=500`. Preflight does not count as GPU execution. A busy-device queue is reported as waiting.

## Use results to select the next experiment

If Q+BC worsens imitation, test whether that change affects real success before changing its coefficient. If both heads fail to imitate, inspect conditioning, dropout and data support. Reference failures before the actor gate require separate investigation. Do not select checkpoints on the final seeds.

After understanding baseline learning, test correct/no/matched-wrong history; then A→B→A conditions with clearing, retention, time decay and residual-based downweighting; then memory × a recovery-validated fixed expert. These later experiments do not run automatically. Report per-attempt unassisted success and correction duration, not only cumulative retry success.

CPU tests cover input rejection, version separation, missing-file accounting, unchanged source indices, matched sampling/initialization, identical zero-Q warmup, and divergence when Q is enabled. GPU checks reject occupied or unreadable devices. Simulator validation remains pending while GPUs are occupied.
