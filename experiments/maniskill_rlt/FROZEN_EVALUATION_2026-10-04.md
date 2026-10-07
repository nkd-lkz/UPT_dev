# Evaluate the saved memory heads

This experiment checks whether a saved RLT head uses past responses during control. It loads model weights only. It does not train or restore replay.

The October 3 pair reached its eight-hour limit. At the last shared checkpoint, iteration 275, both heads succeeded in three of eight evaluation episodes. These observations do not establish a memory benefit. Select checkpoint 275 because it is the last shared save, not because it has the best score.

| Arm | Saved head | Context at each decision | Action source |
|---|---|---|---|
| zero | zero, iteration 275 | zero | learned head after the existing automatic gate |
| response | response, iteration 275 | completed response history | learned head after the same gate |
| response-empty | response, iteration 275 | zero | learned head after the same gate |
| reference | zero, iteration 275 | unused for control | frozen VLA reference throughout |

Use evaluation seeds 4001, 4002, 4003 and 4004. Each seed has 16 lanes and one episode per lane, with a 500-tick limit. Keep the seed, lane count, gate and reference checkpoint identical across arms. Save the initial observation fingerprint and each episode result. These are new evaluation seeds, not a claim of unseen physics or disjoint training states. Evaluate the two-episode integration check separately from this set.

`response-empty` changes the input distribution of the response-trained head. It measures reliance on context. It is not the retrained zero-context control, and it is not a retain/clear test across retries.

## Run one evaluation

Set the asset and output paths. Use a new output directory for each run.

```bash
export RLINF_VENV=/path/to/venv
export RLT_STORAGE=/path/to/assets
export RLT_STAGE1_ACTOR=/path/to/stage1/actor
export RLT_DATASET_DIR=/path/to/lerobot/dataset
export RLT_OUTPUT_ROOT=/path/to/new/evaluation
export RLT_PHYSICAL_GPU=0
export RLT_MEMORY_READER=response
export RLT_SMOKE_PROFILE=matched
export RLT_EPISODE_STEPS=500
export RLT_EVAL_CHECKPOINT=/path/to/global_step_275/actor/model_state_dict/full_weights.pt
export RLT_EVAL_VARIANT=native
export RLT_EVAL_ENVS=16
export RLT_EVAL_SEED=4001
bash run_rlt_portable.sh --memory --eval
```

Use `zero_context` for `response-empty`. Use `reference` for reference-only control. The zero-trained arm uses `RLT_MEMORY_READER=zero` and `native`.

The launcher checks GPU identity and RGB rendering. It then starts only rollout and environment workers. It does not start an actor learner. The original training configuration stays unchanged.

## Check what actually ran

Read `evaluation-config.yaml`, `episode-records.json` and `route-audit.json` in the run directory. The last file records weight equality, the first observation fingerprint, routed actor slots, nonempty history and the action difference after masking history. The difference holds the current state and reference action fixed. It is a dependency measure, not a success metric.

Independent evaluation starts with rollout version zero. Training warmup would suppress the learned head at this version. The evaluator disables that training gate for `native` and `zero_context`; the environment's automatic phase gate still applies. The `reference` arm explicitly retains the warmup gate and verifies that no learned action was routed.

Routing fractions count scheduled chunk slots, including finished lanes. They do not equal executed control-tick fractions. Episode records identify which lanes reached the critical phase. Keep these denominators distinct.

The replay checkpoint may contain stale index entries after cache eviction. This evaluator does not load that replay. Do not infer full training-resume safety from a successful model-only evaluation.

## Decide the next change

Compare paired episode outcomes before changing the network. If the response head barely reacts to history, inspect its input scale and learning signal. If it reacts but control does not improve, test whether the history describes useful execution conditions. If all learned heads trail the reference, repair the baseline learning setup first. Require repeatable control evidence before adding Stage 1B or claiming fewer corrections.

## Completed results: October 4

All 256 planned episodes completed. Initial observation fingerprints matched across arms for every seed. All weights remained unchanged. The reference arm routed no learned actions.

| Arm | Successes / 64 | Actor-routed / scheduled chunk slots |
|---|---:|---:|
| zero | 16 / 64 | 502 / 3200 |
| response | 15 / 64 | 550 / 3200 |
| response-empty | 18 / 64 | 450 / 3200 |
| reference | 19 / 64 | 0 / 3200 |

All arms reached the automatic critical-phase gate in 26 of 64 episodes. The other 38 episodes failed before that gate. Relative to reference, zero lost three successes and gained none; response lost four and gained none. These are paired development observations from one training seed, not a significant treatment effect or evidence that memory is universally harmful.

Masking response history changed the head's normalized action components by a mean absolute 0.00454 across all scheduled predictions. The head uses context, but this run does not show useful control effects. Completed lanes are included in that sensitivity statistic.

Validate a completed monitor snapshot with:

```bash
python -m toolkits.rlt.summarize_memory_evaluation status.json public-results.json
```

The summary rejects incomplete queues, changed weights, missing or duplicate lanes, and mismatched initial observations. It retains each episode result without private paths.

The next experiment should isolate baseline learning. First measure actor/reference action error at matched gate states. Then train BC-only and Q+BC zero-context heads from the same initialization on the same frozen transition set, with equal updates. Evaluate all saved checkpoints on a prespecified new set. This is an offline diagnostic, not a replacement online baseline. Do not select the best checkpoint using the current 64 episodes and report them again as a final test. Require a new online comparison before expanding the memory encoder or adding expert-cost claims.
