# Two-GPU comparison of completed-response context

This campaign tests whether the branch's existing response descriptor helps online control under matched settings. It does not implement the proposed Stage 1B, an expert correction protocol or transfer. Both jobs run from one frozen revision of the Zeva development branch.

GPU 0 runs `reader_type: zero`; GPU 1 runs `reader_type: response`. Both collect the same bounded history and use the same 64-dimensional context slot. Zero context is a capacity-matched RLT control, not the unmodified head architecture. Both readers have no trainable parameters, and with seed 1234 every initial actor/critic tensor is identical. The response model reads seven empirical command-response slopes and seven support values, padded to 64; the control always receives zero. Both use critic/actor optimization already present in the branch.

The shared Stage 1 reference is the existing ManiSkill joint-control step-2000 checkpoint. Both main pilots use `PegInsertionSideWideClearance-v1`, 4 train lanes, 8 fixed evaluation lanes, 500 control ticks per episode, the same automatic critical-phase gate, no expert takeover, batch 32/microbatch 8, 512 BC warmup updates and at most 64 learner updates per outer iteration. Seeds are actor 1234 and environment 2026. Original always-on overnight results are not directly comparable.

Each job first performs GPU identity and RGB simulation probes, then a two-iteration, 100-tick integration smoke with checkpoints and a critic-weight update audit. A successful smoke starts a fresh main run. The main budget is 1000 outer iterations or 8 hours including startup and smoke, whichever comes first; checkpoint, evaluation and video intervals are 25. This is a single paired development pilot, not a convergence or statistical significance claim. Compare only common completed training budgets; repeated evaluation of eight initial states does not create independent samples.

```bash
# Use separate fresh output directories; each process owns its own GPU and Ray head.
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  RLT_OUTPUT_ROOT=/path/to/campaign/zero \
  bash run_rlt_memory_comparison.sh zero 0
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  RLT_OUTPUT_ROOT=/path/to/campaign/response \
  bash run_rlt_memory_comparison.sh response 1
```

`comparison.log`, `phase.txt`, start/end timestamps and final `exit_code.txt` live in each output root. Exit 124 denotes the wall-time limit; use the last completed checkpoint. Per-run directories contain the actual resolved config, TensorBoard, videos, Ray logs, periodic model/optimizer/replay checkpoints and the runner state. Do not automatically restart a failed job or pool smoke and main metrics. Exact simulator continuation is not promised by checkpointing.

At launch, inspect nonzero learner updates, finite losses, policy routing and checkpoint contents. Main evaluation after the 512-update warmup is needed before interpreting an actor's success; the smoke only checks integration. A performance comparison needs the same reset IDs, code and step counts and must report the actor's actual control fraction. A later seed campaign should preserve this protocol before adding an attention reader or consequence pretraining.

Official RLinf was checked on 2026-10-03: main `c70606f08cdca259b8dec03d4430926b5b8fac9d`; concurrent group creation PR #1645 merged; retained replay checkpoint PR #1623 remains open; replay/FSDP PR #1626 merged upstream. No upstream change was incorporated into these runs. Separate Ray heads isolate this pair. Checkpoint size/resume semantics remain an audit item; do not treat an open PR as an applied fix.

The first October 3 launch (`c4d38d5f`, campaign `zeva_matched_20261003_163012`) stopped at the first learner call: FSDP original-parameter writeback expected a flattened tensor but encountered the full matrix shape. Neither member completed a checkpoint. Its logs remain separate. For these two parameter-free readers, the portable launcher now selects the existing `use_orig_params=False` baseline path; the Q head remains independently wrapped and critic-owned. Trainable attention still requires `use_orig_params=True`. This is a scoped compatibility choice, not a general FSDP fix or an upstream merge. Both members must be rerun from scratch under the same new revision.
