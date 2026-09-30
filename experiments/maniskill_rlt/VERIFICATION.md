# Interaction-Memory Verification — 2026-09-26

See the [2026-09-30 audit](AUDIT_2026-09-30.md) for current changes, diagnostic results and reproducible commands. Earlier design and verification history is retained below.

This record preserves the initial CPU/configuration checks. The subsequent [pilot report](PILOT_RESULTS.md) / [中文实验记录](PILOT_RESULTS.zh-CN.md) supersedes the pending GPU gates below: real smoke, optimizer updates and resume passed; 195 CPU tests now pass. Predictive benefit remains unproven. Baseline parent: `ff56663769fd00f6108c39195888f5d55cb8a737`; branch: `research/rlt-zeva-interaction-memory`.

At the initial implementation milestone, no training, Ray cluster, GPU probe, real simulator, or weight download was started. The existing baseline virtualenv was reused without installing or upgrading packages. Synthetic backward passes verify gradient routing; they are not learning experiments.

## Verified

The following command completed with **192 passed, 1 skipped, 1 deselected**. It includes 23 interaction-memory cases, 113 data cases, and 56 passing model cases. The model skip is the existing optional Gemma3 import check.

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH="$PWD" \
  /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m pytest \
  tests/unit_tests/test_interaction_memory.py \
  tests/unit_tests/test_data.py \
  tests/unit_tests/test_models.py -q \
  -k 'not vlm_trend_batch_video_metadata_stays_nested_per_sample'
```

| Contract | Evidence |
| --- | --- |
| Past-only, bounded memory | Capacity, recent/retrieved deduplication, owned snapshots, contradictory records preserved |
| Reset ownership | Different instances clear; explicit same-instance retry retains archive; vector lanes and partial resets stay isolated |
| Real wrapper boundary | `ManiskillRLTEnv` runs against a fake external simulator; checks early termination, frozen tails, auto-reset, retention and NumPy/Torch actions |
| Effective commands | Controller-equivalent [-1,1] clipping in records; original environment commands not changed |
| Numerical behavior | Empty reader is exactly zero; masked invalid padding does not affect results; finite critic gradients |
| Gradient ownership | Actor conditioning detaches reader; critic can update it; target parameters receive no gradients |
| Checkpoints | Neural weights and standalone runtime-memory state each roundtrip with `weights_only=True` |
| Disabled behavior | Parameter schema, initialization RNG and deterministic actions match feature-off construction |
| Replay/transport | Current and terminal snapshots survive `EnvOutput`, rollout, and actual trajectory collector; no reset-state substitution |
| Configuration | Baseline AC and baseline GPU-2 smoke compose with the same overlay; invalid target mode rejected |

The extended launcher passed both `--check` and `--memory --check`, using `RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv`. Checks located the step-750 weights (9,933,605,356 bytes), normalization JSON, and Vulkan files. They did not deserialize weights or test GPU placement. Bash syntax and Ruff checks/formatting passed on changed code; `git diff --check` passed.

## Known Existing Test Failure

An unfiltered `test_data.py` run produced 113 passes and one failure: `test_vlm_trend_batch_video_metadata_stays_nested_per_sample`. The installed OpenPI transformers fork rejects `VideoMetadata(..., frames_indices=...)`.

The identical test was run in the untouched `/home/luokz/rlinf_rlt/UPT_dev` baseline and failed identically. It is excluded explicitly in the combined regression command above; no test was silently modified or dependency upgraded to hide it. This does not establish that the full repository test suite passes.

## Documentation Checks

The docs-check build harness completed both EN and ZH builds with zero reported warnings under its standard optional-autodoc-noise filtering. The existing RST trees were not edited. Repository-wide static checks still report 61 existing inline-markup warnings and 26 existing symbol findings; these are not reported as newly fixed.

The new experiment Markdown pages were reviewed separately for code paths, configuration fields, chronology, command scope, and English/Chinese parity. They are research worktree guides, not new Sphinx gallery pages. The refine-docs and docs-check workflows guided the explanation order and code/document consistency checks.

## Pending Before Claiming a Working Training Integration

The read-only preflight does not replace a real GPU smoke. After separate authorization, verify the following with the bounded memory smoke launcher:

1. GPU UUID isolation, real ManiSkill rendering and physics, and Stage 1 feature extraction.
2. FSDP optimizer grouping, actor/critic updates, reader weight synchronization and target updates on the real distributed path.
3. Replay memory occupancy metrics, finite losses, checkpoint save/load, and measured CPU/GPU overhead.
4. Frozen-parameter retries under audited identical physical conditions; ensure archives actually persist when intended.

Runtime archives are not automatically restored with Ray environment checkpoints. Old baseline Stage 2 checkpoints/replay are not plug-compatible with the enlarged memory-enabled model. No success-rate, faster-convergence, fewer-intervention, cross-task-transfer, or causal-identification result is claimed. These require the controlled experiments described in the design.
