# RoboTwin 2.0 online RLT and LIBERO-Pro reference evaluation

These entry points provide explicit single-GPU contracts for the local AlphaBrain RLT baseline. RoboTwin 2.0 separates training and evaluation seeds, and LIBERO-Pro requires every task, reset state, perturbation, and policy seed to be named on the command line.

## RoboTwin 2.0

Train the Stage-1 encoder with `robotwin2_pretrain.py`, then warm up the residual actor from immutable reference caches:

```bash
python toolkits/rlt/robotwin2_online.py warmup \
  --stage1 /path/to/encoder.pt \
  --norm-stats /path/to/physical-intelligence/robotwin/norm_stats.json \
  --reference-root /path/to/reference-runs \
  --output /path/to/warmup.pt
```

Online Q+BC consumes only RoboTwin's training seed file. Evaluation consumes only its evaluation seed file and executes the same 10-action chunks used by the learner:

```bash
python toolkits/rlt/robotwin2_online.py online \
  --repo "$PWD" --robotwin /path/to/RoboTwin \
  --checkpoint /path/to/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --norm-stats /path/to/physical-intelligence/robotwin/norm_stats.json \
  --learner /path/to/warmup.pt --seeds 0,1,2,3,4 \
  --output /path/to/online-run

python toolkits/rlt/robotwin2_online.py evaluate \
  --repo "$PWD" --robotwin /path/to/RoboTwin \
  --checkpoint /path/to/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --norm-stats /path/to/physical-intelligence/robotwin/norm_stats.json \
  --learner /path/to/online-run/online.pt --policy actor \
  --seeds 0,1,2,3,4 --output /path/to/heldout-eval
```

The frozen Pi0.5 model proposes 50 actions, while both the actor and environment exchange exactly 10 actions. Every episode records the requested seed, resolved environment seed, geometric failure stage, and physical action correction.

## LIBERO-Pro

Install `rlinf-liberopro` in an isolated dependency directory and set `LIBERO_PRO_RLT_ROOT` to the asset root. The wrapper refuses a silent fallback to standard LIBERO and uses EGL rendering:

```bash
toolkits/rlt/run_libero_pro.sh python toolkits/rlt/libero_pro_reference.py \
  --checkpoint /path/to/RLinf-Pi05-LIBERO-SFT \
  --output /path/to/libero-pro-run \
  --cases 0:0:lan:42 \
  --cases 0:0:object:42
```

Each case has the form `TASK:STATE:PERTURBATION:SEED`; valid perturbations are `lan`, `object`, `swap`, and `task`. Successful simulator episodes and benchmark-infrastructure failures are persisted separately. An invalid BDDL file or missing perturbation reset state is therefore not converted into a policy failure.
