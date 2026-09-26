# Action-conditioned future representations for RLT

This design tests whether predicting interaction outcomes in a frozen RLT feature space makes online actor–critic learning more sample-efficient. It starts from the official ManiSkill reproduction, adds a lightweight offline adaptation stage, and carries that module into real-transition learning. The [pilot results](PILOT_RESULTS.md) add a residual-prediction variant, an architecture figure and real-data/GPU evidence; they do not establish improved RL, physical laws or complete continual learning. [中文](DESIGN.zh-CN.md)

## 1. The question we can test first

The primary hypothesis is: with the same demonstrations, frozen VLA, environment budget, reward, switching rule and evaluation protocol, an action-conditioned outcome representation helps RLT reach a fixed success-rate threshold in fewer environment control steps. The mechanism is supervision of what a commanded action actually changes, not a replacement reward.

The first experiment is still joint-control peg insertion. Contact-aware recovery, fewer interventions and cross-task transfer are subsequent hypotheses, not conclusions from the auxiliary loss. We deliberately postpone explicit memory retrieval and automatic phase switching so that one mechanism can be isolated.

## 2. What comes from FLARE, and what changes

[FLARE's paper](https://arxiv.org/abs/2505.15659) and [project](https://research.nvidia.com/labs/gear/flare/) motivate predicting future observation embeddings while learning actions. Its learnable future tokens interact with action tokens inside a diffusion/flow transformer; intermediate future-token features are aligned with a future visual-language embedding. Its action-aware embedding model and EMA targets belong to its original training recipe.

Our implementation is **FLARE-inspired, not a FLARE reproduction**. It does not insert tokens into Pi0's action denoiser or reproduce FLARE's architecture, dataset or reported results. Instead, a separate small transformer learns action-conditioned futures in the existing frozen RLT coordinate system. This makes the same targets available to Stage 2 without retraining a multi-billion-parameter visual encoder on every replay update. Whether this adaptation is novel relative to other latent-dynamics RL work still needs a broader literature review.

| Decision | This branch | Reason / limitation |
| --- | --- | --- |
| Visual backbone | Completed Stage 1A, frozen | Stable cached/replay coordinates; cannot recover contact information absent from its features |
| Future targets | Frozen RL token + normalized proprioception | Reuses Stage 2 observations; no image decoder |
| Action conditioning | Executed action prefix and a learned horizon query | Predictions depend on commanded actions, not time alone |
| Action relevance | Auxiliary chunk BC from the current adapter | Prevents a purely visual objective; not a guarantee of control relevance |
| Online update | TD loss plus outcome supervision on actual replay | No imagined transitions or learned reward substitution |
| Persistence | Model parameters, checkpoints and offline cache | Not a searchable long-term experience library |

## 3. Stage 1 becomes two explicit sub-stages

Stage 1A remains official OpenPI RLT SFT, including its original VLA and token-reconstruction objectives. The currently running baseline is untouched. Stage 1B is the new lightweight adapter training stage; it needs a completed, immutable Stage 1A checkpoint. Calling Stage 1B an end-to-end modification of Pi0 would be incorrect.

The data path is:

```text
demonstration RGB + language + joints
  -> frozen Stage 1A / exact Stage 2 preprocessing
  -> episode cache: z_t, p_t, a_t, frame and task IDs
  -> Stage 1B: current-state adapter + action-prefix transformer
  -> sidecar checkpoint with weights / preprocessing / control-unit hashes
  -> Stage 2: actor/critic inputs + online future loss on real replay
```

The exporter calls `Pi0Eval.extract_rlt_obs(include_reference=False)`. Only unused reference-action decoding is skipped; prefix extraction and proprioception normalization are shared with rollout. Its default remains `True`, preserving existing inference behavior and RNG consumption. Images and original weights never enter Git.

The supplied LeRobot v2.1 dataset contains 400 episodes, 28,681 observations, RGB/wrist RGB, nine state values and eight action values at 10 Hz. It does **not** contain explicit contact, force or friction annotations. Dataset actions remain raw `pd_joint_delta_pos` environment commands in [-1, 1]; they must not be replaced with the normalized actions used internally by SFT. Proprioception is already normalized by the frozen OpenPI transform.

## 4. The learned module and its losses

Let `z_t` be the 2048-dimensional frozen RL token, `p_t` the nine-dimensional normalized proprioception and `A[t:t+h]` the executed eight-dimensional actions. Write `LN` for non-learned per-token layer normalization. A two-layer MLP builds `e_t = encoder([LN(z_t), p_t])`, with 128 dimensions by default.

For each horizon `h` in {1, 5, 10}, a two-layer, four-head transformer reads `[e_t, action_0 + position_0, ..., action_(h-1) + position_(h-1), query_h]`. Separate horizon calls ensure that a one-step prediction cannot inspect actions after that step. Three independently initialized prediction heads output `pred_z_h` and `pred_delta_p_h`. The heads share a trunk; they are not three independent world models.

The default horizon loss is:

```text
L_h = masked_mean(
    1 - cosine(pred_z_h, stopgrad(LN(z_(t+h))))
    + 0.1 * MSE(pred_z_h, stopgrad(LN(z_(t+h))))
    + 0.5 * SmoothL1(pred_delta_p_h, stopgrad(p_(t+h) - p_t))
)
L_offline = future_weight * mean_h(L_h)
            + 0.1 * MSE(anchor(e_t), stopgrad(LN(z_t)))
            + 0.1 * masked_MSE(tanh(behavior(e_t)), demonstrated_chunk)
```

All masks are normalized by their valid count, with a denominator floor of one. Each ensemble member uses an independent bootstrap mask with probability 0.8; the first head always sees all valid rows. Evaluation disables bootstrap masking. A terminal-only online minibatch produces a differentiable zero auxiliary loss.

The anchor preserves current-state information, and BC ties the adapter to actions. Frozen nonconstant targets reduce collapse risk but do not prove the model uses actions: a static scene predictor can still be competitive. The evaluation tool therefore reports an unchanged-state predictor, shuffled-action predictions and per-horizon proprioception errors. Success requires downstream control improvement, not merely a declining training loss.

Proprioceptive response prediction is the initial physical-interaction signal. It can reflect motion under constraints, but visual occlusion, gripper compliance or contact may not be observable in `z_t,p_t`. No “contact label” is inferred from slow motion, and no force law is claimed. Adding real simulator contact/force labels is a separate, privileged-information ablation; inference must remain on the declared sensor inputs.

## 5. Transfer into Stage 2

The sidecar extends existing observations rather than replacing them. Actor input is `[reference_chunk, z_t, p_t, stopgrad(e_t)]`; critic input is `[z_t, p_t, e_t]` plus its candidate action. Original raw features remain available if the learned representation is unhelpful. The auxiliary BC head is **not** copied into the Stage 2 actor; otherwise BC warm-start would confound the representation comparison.

Stage 2 optimizes the original RLT objectives, with:

```text
L_critic_total = L_original_RLT_TD + 0.1 * L_online_world
L_online_world = L_H + 0.1 * L_anchor       # no auxiliary online BC
L_actor = original RLT Q/BC objective      # existing intervention targets retained
```

`future_weight` also multiplies the online future term. The critic optimizer owns **all** sidecar parameters, including future queries and prediction heads. The actor reads a detached adapter feature. TD gradients may change the adapter; auxiliary gradients maintain its outcome-prediction role. Target updates must use `target_update_type: all`, so the target critic's adapter follows the same Polyak update. Parameters are ordinary registered policy parameters: the existing checkpoint and weight-sync paths include them. A single-GPU multi-process FSDP/Ray smoke and resume passed; multi-GPU scaling remains untested.

The implemented ManiSkill collector produces a transition after the entire action chunk. With `H=10` and 10 Hz control, the online prediction spans **one second**, not 0.1 seconds. `replay_world_batch` checks reward-slot duration and action count against H. It uses the actual routed/executed chunk, including expert replacements, not the proposed student action. Any termination or truncation masks the whole row: RLT terminal rows may carry the current observation as a placeholder successor, and treating that as physics would teach false zero motion.

Only recorded critical-phase transitions enter the existing RLT replay schedule. This is not an all-task world-model update. Updating the shared trunk online at H can degrade shorter-horizon predictions; retain offline validation and measure that forgetting. The frozen feature coordinate system solves stale feature encoding, not all catastrophic forgetting.

## 6. Optional exploration constraints

The default experiment leaves the original action parameterization unchanged. An optional `bounded_residual` mode uses:

```text
center = clip(reference_chunk, -1, 1)
radius = configured_radius                             # fixed-bound ablation
radius = max(min_radius, radius / (1 + variance/scale)) # optional heuristic
action = clip(center + radius * tanh(actor_sample), -1, 1)
```

Variance is ensemble disagreement predicting the reference chunk's future, evaluated without gradients. A scale of zero disables uncertainty adaptation; it does not mean division by zero. The defaults are a 0.2 radius, 0.02 minimum, adaptation off. Neither mode is enabled in the main config.

This limits deviation from the base action; it does not certify safety or identify contact. Shared-trunk ensembles may be confidently wrong. A poor reference can prevent necessary recovery when the radius is too small. Test a fixed radius before attributing gains to uncertainty. The diagnostic tool reports error/disagreement correlation and a p95 scale candidate, **not an automatic calibration or activation decision**. Recalibration is needed under online distribution shift.

Clipping changes action densities. The implementation retains baseline pre-tanh logprobs only for interface compatibility, forbids nonzero entropy in this research path, and must not be reused in entropy-regularized SAC/PPO without a correct distribution treatment.

## 7. Provenance, splitting and training defaults

Cache manifests bind checkpoint bytes, norm-stat bytes, resolved feature-model configuration, controller and control frequency. Episode files have content hashes. The exporter resumes only identical sources, rejects missing/nonconsecutive frames and timestamps, and publishes a complete manifest only after all episodes succeed. Checkpoints and manifests use same-directory temporary files and atomic rename, without requiring NAS symlinks.

Whole episode IDs are assigned deterministically by a seeded hash to a 90/10 split. Windows cannot cross episode boundaries; unavailable futures and actions are masked. This is **adapter validation**, not an independent policy test: the existing Stage 1A may have seen all 400 episodes. An end-to-end generalization claim needs an independent demonstration/task split established before Stage 1A, plus held-out environment seeds.

Stage 1B defaults: fp32 small model, global batch 256, microbatch 32 (eight accumulation steps), AdamW 3e-4, weight decay 0.01, 500-step warmup, cosine decay to 10% of peak, at most 10,000 optimizer updates. Validate every 250 updates; stop after 12 unimproved validations. These are starting settings, not a convergence guarantee. `best.pt` is selected by validation auxiliary loss; also inspect action sensitivity and control evaluations. `last.pt` stores optimizer, scheduler, CPU/CUDA RNG, sampler, split/config and cache manifest for exact same-configuration resume.

Stage 1B uses one process/GPU; the small cached-feature learner does not need DDP. Research Stage 2 uses two visible GPUs, actor on 0 and rollout/simulation on 1, eight train/eval environments, actor global batch 256 and microbatch 64. Scheduling, reward and switching thresholds are inherited from the frozen baseline. Neither resource settings nor GPU memory/throughput have been measured for this new path yet.

## 8. Experiments that distinguish the mechanism

Before expensive RL, verify action shuffling hurts prediction, compare with persistence, inspect feature variation and diagnose per-horizon errors. A low average loss dominated by stationary frames is not enough.

| Arm | Stage 1B | Online outcome loss | Exploration |
| --- | --- | --- | --- |
| A | No sidecar: matched original RLT | Off | Original |
| B | Same-size adapter, `future_weight=0`, anchor + BC | Off | Original |
| C | Full future pretraining | Off; TD still adapts encoder | Original |
| D | Full future pretraining | On | Original |
| E | D | On | Fixed bounded residual |
| F | D | On | Bounded residual + validated uncertainty scale |

Do not interpret C as a frozen adapter: disabling `latent_world_weight` disables self-supervised replay loss, not encoder TD gradients. B controls added capacity and extra BC exposure. A/B/C/D must share Stage 1A weights, demonstrations, environment seeds/budgets and intervention rules. E/F change action geometry and require their own comparison. The provided matched-baseline config reproduces the research resource settings without the sidecar. Expert takeover is off in **both** supplied configs because no stronger expert checkpoint is assumed; this setting cannot measure a reduction in interventions.

For each arm, use at least three paired seeds for initial evidence, preferably five for final reporting. Primary measures are success-vs-control-steps AUC and steps to a preregistered sustained success threshold (for example 80% over three evaluations). Count all environment control steps, including base/expert-controlled phases, not only critical replay rows. Report seeds failing to reach the threshold as censored failures rather than dropping them. Also report wall-clock time, gradient updates, peak VRAM, inference latency, collisions if explicitly instrumented, and recovery from held-out perturbations. Evaluate enough independent episodes to report uncertainty intervals; eight parallel evaluation environments is only concurrency, not a sufficient sample size.

To study interventions, enable the **same** expert checkpoint and takeover rule across arms; count takeover events and takeover control steps per episode and per successful episode, and preserve a no-intervention success evaluation. Do not improve the metric by relaxing the takeover threshold or accepting more failures. The current expert is a simulated stronger policy, not necessarily a human.

## 9. From one task to accumulating experience

Start with contact-sensitive shifts inside peg insertion: held-out clearances, pose offsets, approach errors and controlled physical parameter changes. Distinguish in-distribution learning from perturbation robustness. Simulator-only variables may define evaluation strata, but should not silently become actor inputs.

Then pretrain across several tasks with compatible embodiment/control/observation contracts, train sequentially, and evaluate both new-task sample efficiency and old-task retention. Use held-out task identities, not just new seeds of the same task. The cache retains task identity for splitting/analysis, but this release does not implement task-balanced sampling, a generic multi-task gate or task-specific heads. ManiSkill RLT's peg-specific critical-phase and takeover rules cannot be transferred merely by changing an environment ID.

An explicit memory extension would store `(feature-version, task, context, executed action, outcome, error, confidence, recovery)` and compare no memory, recent history, fixed archive and incrementally updated archive. It needs retrieval latency budgets, conflict handling, evidence thresholds and held-out follow-up attempts. That extension is not implemented here; current persistent learning lives in model parameters and standard replay/checkpoints. The default replay is finite and does not preserve an unlimited experience archive.

## 10. Review gates and limitations

Accept the implementation only if baseline-off compatibility, episode boundaries, future masking, target stop-grad, chunk-duration matching, terminal masking, optimizer ownership, deterministic CPU resume and config parity tests pass. GPU 2 cache extraction and short Stage 2 checkpoint/restart tests are now recorded in [pilot results](PILOT_RESULTS.md). Unit tests alone are not a substitute for these integration checks; convergence still requires controlled experiments.

Stop or revise the hypothesis if future prediction is insensitive to actions, worse than persistence, improves only teacher loss, or accelerates updates but not interaction-budget success. Do not claim fewer interventions from an intervention-free run. Do not claim physical understanding, cross-task generalization, lifelong growth or guaranteed convergence before the corresponding experiments.

Tomorrow's decisions: approve the Stage 1A/1B factorization versus a later in-DiT implementation; choose the primary success threshold and held-out perturbations; choose expert/no-expert protocol; decide whether actual contact sensing or an explicit memory archive is the next independent variable. The implementation does not require those decisions to be inspected or unit-tested.
