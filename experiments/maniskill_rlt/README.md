# Inspect the RLT Interaction-Memory Branch

See the [2026-09-30 audit](AUDIT_2026-09-30.md) for current changes, diagnostic results and reproducible commands. Earlier design and verification history is retained below.

Use this entry point to inspect the independent memory branch without starting training. Read the [design and code map](DESIGN.md), then run CPU tests and read-only preflight. The [latest pilot results](PILOT_RESULTS.md) include the architecture figure, actual GPU updates, resume and negative predictive-probe results; [verification](VERIFICATION.md) preserves the initial implementation record.

Branch: `research/rlt-zeva-interaction-memory`. Baseline: `ff56663769fd00f6108c39195888f5d55cb8a737`. Stage 1, FLARE, and VR work remain separate.

## Checks Without Training

Reuse the existing Python for synthetic unit inputs and a fake simulator boundary. Gradient tests check backpropagation without creating training checkpoints.

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m pytest \
  tests/unit_tests/test_interaction_memory.py -q
```

Read-only preflight composes Hydra and checks Stage 1 weights, normalization, and Vulkan file paths. It does not deserialize weights, start Ray, allocate a GPU, or create a training directory. GPU 2 in its output describes a future launch, not observed GPU execution.

```bash
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory --check
```

Expect `Interaction memory: True` and `Preflight OK`. Set `RLT_STAGE1_ACTOR` for another fully saved Stage 1 actor directory; the inherited default is step 750.

## Smoke Only After Separate Authorization

The following command was not executed for this delivery. Once GPU 2 use is approved, the launcher retains the existing occupancy checks, private Ray cluster, and physical GPU UUID probe before two bounded update iterations. It is not a convergence experiment.

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory
```

Check GPU placement, finite losses, replay memory fields, and checkpoint save/load before larger experiments. Output is under NAS `runs/stage2_smoke/memory_stage2_gpu2_<time>_<PID>/`, with experiment name `stage2_memory_smoke`. Omitting `--memory` preserves baseline smoke behavior.
