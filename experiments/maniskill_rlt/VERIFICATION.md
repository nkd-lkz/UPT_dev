# Verification / 验收记录

This records implementation checks on 2026-09-25, not a robotics experiment result. 本文记录代码验证，不代表真实训练、成功率或泛化结果。

## Scope / 范围

- Baseline: `baseline/maniskill-rlt-2026-09-25`, `7db62813`; research worktree: `/home/luokz/rlinf_rlt/UPT_flare_dev`.
- Implementation commit / 实现提交: `cfc503a7` (`feat(rlt): add action-conditioned latent world adaptation`).
- Python: original `/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python`, 3.11.14; imports verified to resolve `rlinf` from the research worktree.
- CPU tests: `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
- No real-data training, feature-cache generation, Ray job or GPU simulation was started. Tiny synthetic optimizer steps belong only to CPU unit tests. 没有启动新的实际训练；合成数据上的两步更新仅用于测试续跑和梯度路径。
- The existing baseline process and its workers remained active in the original worktree; its code/configuration was not switched to the research branch. No existing data, weights, checkpoints or logs were deleted.

## Passing checks / 已通过

The focused command passed **19 tests**:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/unit_tests/test_models.py tests/unit_tests/test_data.py \
  tests/unit_tests/test_worker.py -k latent_world -q
```

Coverage includes / 覆盖：

- Future queries cannot inspect later actions; targets receive no gradients.
- All-terminal online batches yield zero auxiliary loss and gradients.
- Strict sidecar reload, feature-contract mismatch rejection, and shape rejection.
- Disabled extension preserves original RLT state-dict compatibility and deterministic output.
- Actor gradients do not enter the world model; critic/world loss reaches prediction heads.
- Production optimizer partition assigns every world parameter to the critic, and a real CPU critic update changes predictor weights.
- Sidecar parameters are present in checkpoint and selective rollout-sync parameter lists.
- No cross-episode windows or padded-future supervision; corrupt/missing-frame caches are rejected.
- Online horizon matches the complete executed chunk; any done masks the transition.
- Two-step synthetic Stage 1B workflow produces checkpoints; resuming from step one reproduces every final model tensor **bit-for-bit on CPU**.
- Held-out diagnostic tool reads the saved sidecar and reports horizon metrics.
- Hydra research/comparator composition, resource matching, and cache/rollout feature-config equality.
- Actual-content provenance hashing and preflight rejection of changed norm stats/nonzero entropy.

Additional checks / 其他检查：

- A direct CPU comparison against the policy source at historical commit `7db62813` confirms exact state-dict tensors, stochastic/deterministic actions and Q outputs when the extension is disabled. This checks the historical baseline, not only two configurations of the new class.
- All four toolkit `--help` entry points exit successfully.
- `preflight.py --config-only` resolves actor GPU `0-0`, env/rollout `1-1`, global batch 256 and online world weight 0.1 without starting training.
- Read-only inspection of a real parquet frame confirms main/wrist images decode to uint8 `(384, 384, 3)`, state dimension 9 and action dimension 8. This is not a full GPU export test.
- Default `RLTLatentWorld` has **1,685,739 parameters**; this excludes the frozen VLA and extra Stage 2 input-layer weights. No VRAM or latency estimate is presented as a measurement.
- Ruff lint/format and Git whitespace checks pass for the changed Python/source files.
- EN/ZH design/runbook counterparts were checked for equivalent scope, settings, commands and limitations using `refine-docs` and `docs-check` guidance. Research Markdown is not part of the public Sphinx gallery.
- Both EN and ZH Sphinx builds pass the repository's default docs-check build harness with zero reported warnings, using an isolated docs-only venv at `/home/luokz/.cache/rlinf-docs-venv`. The harness filters expected autodoc import noise in a docs-only environment; this is not `--strict-autodoc` validation in the full training environment.

## Broader checks with inherited failures / 继承的非本次问题

The full `test_models.py` + `test_data.py` run reports **185 passed, 1 skipped, 1 failed**. The failing test is `test_vlm_trend_batch_video_metadata_stays_nested_per_sample`: installed `transformers.video_utils.VideoMetadata` rejects `frames_indices`. Running that exact test in the untouched baseline worktree reproduces the same failure. It concerns VLM Trend video metadata, not the RLT future model, and was intentionally not repaired in this branch.

全量相关测试不是全部通过：上面的 VLM Trend 兼容性失败已经在 baseline 独立复现。本次没有改动无关功能，也没有改动共享训练 venv 来掩盖这个问题。

Global documentation scans also retain baseline issues:

- `check_rst_markup.py`: 0 build errors and 61 silent-render warnings. Baseline/research warning sets are identical; only their output order differs.
- `check_doc_symbols.py`: exit 1 with 26 unresolved code-font names. Baseline/research output is identical.

These scans inspect existing Sphinx RST, not the new Markdown research notes. No unrelated repository-wide documentation cleanup is included. 全局文档历史问题保留原状，不与研究实现混合提交。

## Not yet validated / 尚未验证

1. Real frozen-VLA feature extraction from a completed Stage 1A checkpoint on GPU; throughput, numerical consistency across batch sizes and NAS performance.
2. Real-data Stage 1B optimization, held-out action sensitivity, uncertainty calibration and best-checkpoint selection quality.
3. Full multi-process Ray/FSDP Stage 2 execution, optimizer stepping through the distributed wrapper, target/rollout synchronization, GPU memory and checkpoint resume.
4. Success-rate curves, sample-efficiency gains, recovery performance or reduction of expert/human interventions.
5. Contact/force labels, explicit persistent memory retrieval, new-task phase gates, sequential-task retention or held-out-task transfer.

这些都是后续验收门槛，不是已实现收益。运行指引提供了显式后续命令，本次没有自动执行。当前可验收的是源码、CPU contract 测试和研究设计；只有完成对应实验后，才能提出收敛加速、物理理解或成长能力的结论。
