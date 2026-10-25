# RoboTwin 2.0 AlphaBrain RLT baseline

This experiment evaluates a frozen task-specific π₀.₅ reference before training the AlphaBrain RLT heads. It keeps the published 50-step flow-matching horizon intact and treats the shorter horizon as an execution or learner protocol, so an `H=10` result is not mislabeled as the official `H=50` reference.

The pinned baseline is `adjust_bottle` with the public `RLinf-Pi05-RoboTwin-SFT-adjust_bottle` checkpoint, the clean ALOHA environment, three RGB views, 14-dimensional joint control, five flow steps, and fixed evaluation seeds. The development smoke uses one environment on one visible GPU. Final success rates require at least the fixed 50-episode evaluation declared in `adjust_bottle.yaml`.

## Validated development result (2026-10-25)

The native `H=50` reference completed 9 of 10 fixed-seed development episodes (90%, not a final benchmark estimate), with a mean episode length of 160 ticks. The sole failure reached the bottle but displaced it during closure and never lifted it, so it was classified as `grasp_or_lift`. On seed `100100052`, all three replanning protocols succeeded: `H=50` at 150 ticks, `H=25` at 125 ticks, and `H=10` at 150 ticks. All three generated the unchanged 50-step flow trajectory and differed only in how much of it was executed before replanning.

The official 50-demo dataset loaded 7,188 frames. A one-step Stage-1 reconstruction smoke produced loss 4.1131 and gradient norm 9.1827 at 11.4 GiB peak allocated GPU memory. A separate real-rollout learner smoke passed reconstruction, actor BC warmup, and twin-Q plus Q+BC updates. These are connectivity and gradient-flow checks, not evidence that online RLT improves success.

Auditable outputs are under `/mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/runs`: `dev10_summary.json`, per-seed `episode.json` files, `learner-smoke.json`, `model_contract.json`, and `stage1-smoke/status.json`.

## Prepare assets

The setup script downloads to the local data disk because the NAS filesystem does not implement all lock and permission operations used by Hugging Face. It verifies immutable revisions, merges the three safetensors shards into the format consumed by RLinf OpenPI, and pins RoboTwin's `RLinf_support` commit.

```bash
bash toolkits/rlt/setup_robotwin2_assets.sh
```

## Verify the frozen reference

Run `H=50` first because it matches the published reference protocol. `H=25` and `H=10` replan more frequently while sampling the same 50-step flow trajectory.

```bash
export CUDA_VISIBLE_DEVICES=1

for horizon in 50 25 10; do
  toolkits/rlt/run_robotwin2.sh python toolkits/rlt/robotwin2_reference_rollout.py \
    --repo "$PWD" \
    --robotwin /mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/repos/RoboTwin \
    --checkpoint /mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
    --output /mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/runs/reference \
    --execution-horizon "$horizon"
done
```

Each decision cache contains the frozen `[768, 2048]` three-view prefix, its validity mask, the 14-dimensional state, and the full `[50, 14]` physical reference. `episode.json` records the seed, issued tick count, reward, terminal flags, success, and cache provenance.

The distributed RLinf evaluation path is independently represented by `evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_rlt_gate.yaml`. Use it to separate a custom-runner error from a model or environment error.

## Verify the AlphaBrain learner

The learner consumes the real `H=10` decision ledger. One smoke step verifies all three boundaries: new-RLT autoregressive reconstruction, actor BC initialization, and a twin-Q plus delayed Q+BC update.

```bash
toolkits/rlt/run_robotwin2.sh python toolkits/rlt/robotwin2_learner_smoke.py \
  --episode /mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/runs/reference/h10/episode.json \
  --norm-stats /mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/models/RLinf-Pi05-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json \
  --output /mnt/hdd_toshiba_1/luokz/robotwin2_rlt_20261025/runs/learner-smoke.json
```

Passing this smoke establishes protocol and gradient flow only. It does not establish online RL improvement. The training comparison remains frozen reference, BC-only, and Q+BC on identical seeds, with success rate, success-versus-environment-step AUC, wall-clock throughput, and intervention cost reported separately.
