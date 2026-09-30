# ManiSkill RLT Baseline — 2026-09-25

This snapshot pins the code used for the ManiSkill RLT reproduction before
experiments with future latent prediction. Treat it as the comparison branch,
not as a claim that Stage 2 results or convergence have been reproduced.

## Code and Branches

For a ManiSkill-only experiment on a host without Hugging Face access, add
`--skip-libero-assets` to the existing `embodied --model openpi --env maniskill_libero`
installation command. This keeps the package installation and all subsequent
ManiSkill/OpenPI setup, but skips both supported LIBERO asset download paths.
It is accepted only for the combined `maniskill_libero` target; LIBERO tasks
still require their assets. Omitting the flag preserves download failures.
After restoring the network, activate the venv and run
`libero-download-assets --skip-existing` to fetch the standard LIBERO assets.

- Upstream code snapshot: `b85c07175b10017bf58ab83e1b1eee99666d0626`.
- Baseline branch: `baseline/maniskill-rlt-2026-09-25`.
- Follow-up experiment branch: `research/rlt-flare-latent-dynamics`.
- Baseline Stage 1: `examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml`.
- Baseline Stage 2 AC: `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`.
- Baseline Stage 2 TD3: `examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml`.

Keep the running training checkout on the baseline branch. Develop experiments
in a separate worktree. Do not merge experiment changes back into this snapshot;
create a new dated baseline if the reference implementation must change.

## Recorded Stage 1 Run

`run_rlt_stage1_2gpu.sh` is the exact host-specific launcher used on 2026-09-25;
its paths refer to this lab machine. It has no embedded credentials. Adapt the
paths before using it elsewhere, and authenticate W&B to the intended account.
The original run used the machine's existing W&B login.

| Setting | Value |
| --- | --- |
| Run | `maniskill_rlt_stage1_2xl40_20260925_163418` |
| GPUs | L40, physical devices 0 and 1 |
| Per-rank micro batch / global batch | 1 / 256 (128 accumulation steps) |
| Max steps / save interval | 2000 / 250 |
| Learning rate / warmup | 2.5e-5 / 500 steps |
| Scheduler | cosine, horizon 10000, minimum 2.5e-6 |
| Precision | OpenPI native mixed parameter dtypes; AMP disabled |
| FSDP | no_shard, gradient checkpointing disabled |
| Allocator | `expandable_segments:True` |
| Ray | 2 GPU resources, 8 GiB object store, temp in /dev/shm, spill on NAS |

The scheduler horizon is intentionally recorded as executed: it exceeds the
2000-step stopping point, so this run does not complete the full cosine decay.
At approximately step 200, training was alive with loss about 0.583 (initially
3.25). These are training diagnostics, not held-out manipulation success rates.
Checkpoints, datasets, caches and credentials are external artifacts and must
not be committed. The first expected checkpoint is `global_step_250/actor`.

## Runtime and Data

Python 3.11; torch 2.11.0+cu128; Ray 2.58.0; ManiSkill 3.0.0b22;
SAPIEN 3.0.1; rlinf-openpi 0.1.1; transformers 4.57.6; W&B 0.25.0.
Use `requirements/install.sh` to recreate dependencies; these versions describe
the local reproduction, not a portable lockfile.

Data: `RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ`, locally named
`maniskill_peginsertionside_joint`, with its calculated `norm_stats.json`.
Base weights: `pi05_base`. The task uses 8-D joint-delta actions and 9-D Panda
proprioception. The shipped Stage 2 task is
`PegInsertionSideWideClearance-v1`; it should not be confused with the narrower
`PegInsertionSide-v1` renderer smoke test.

## Evaluate the Completed Stage 1 Checkpoint

The resumed run reached step 2000 and exported `global_step_2000/actor/model_state_dict/full_weights.pt`. The launcher's final shell parse error happened after training and export; validate the checkpoint through the evaluation preflight rather than interpreting that shell exit code as a missing model.

Run the dedicated GPU 2 launcher to evaluate exactly 20 fixed reset IDs. It loads the Stage 1 OpenPI policy directly, disables RLT phase switching and expert takeover, and records the complete 500-control-step window as one synchronized tiled MP4 containing all 20 environments.

```bash
cd /home/luokz/rlinf_rlt/UPT_dev
bash run_rlt_stage1_eval20_gpu2.sh --check
tmux new-session -d -s rlt_stage1_eval20 \
  "cd /home/luokz/rlinf_rlt/UPT_dev && bash run_rlt_stage1_eval20_gpu2.sh"
```

The launcher refuses a busy physical GPU 2, starts a private Ray head, and writes results under NAS `runs/stage1_eval20/eval20_<timestamp>_<pid>/`. Read `eval/success_once` as the success rate over 20 episodes and verify `eval/num_trajectories=20`. The MP4 is under `video/eval/seed_2026/0.mp4`; it is one tiled recording, not 20 separately encoded files.

## Stage 2 Acceptance Still Required

Before launching Stage 2, select and evaluate a saved Stage 1 checkpoint,
configure the stronger expert checkpoint or explicitly disable takeover, and
size rollout environments for the available GPUs. Stage 2 includes simulator
state-based phase switching and optional expert intervention. Count both when
comparing reduced-intervention methods. No Stage 2 run is certified by this
snapshot, and no new training was launched while making the branches.

See [中文记录](BASELINE.zh-CN.md) and the existing bilingual RLT documentation.
