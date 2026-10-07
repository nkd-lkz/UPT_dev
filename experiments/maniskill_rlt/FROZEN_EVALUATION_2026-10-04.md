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
