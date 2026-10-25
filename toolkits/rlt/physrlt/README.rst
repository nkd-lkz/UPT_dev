PhysRLT Development Experiments
==============================

Run small LIBERO-Pro experiments to test whether measured execution and geometry supervision improve an action-side representation. This isolated toolkit freezes AlphaBrain's released Pi0.5 model, captures its existing action-expert forward, and fits equal-capacity readouts before attempting short BC/Q+BC control pilots. Use the prediction results as development evidence, then inspect closed-loop results separately.

Prepare a Fixed Pilot
---------------------

The local host profile reuses the installed simulator and verified weights. From this worktree, prepare an immutable plan at a new output directory:

.. code-block:: bash

   bash toolkits/rlt/physrlt/run_local.sh prepare \
     --root /mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill/research/physrlt_20261025/pilot-v1 \
     --source /home/luokz/rlinf_rlt/third_party/AlphaBrain \
     --assets /mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill/research/public_models_20261025/libero/alphabrain-pi05-all4suite

This writes explicit task/layout/split cases and a dependency graph. Task 0 uses the object perturbation; task 3 uses language perturbation. Each task has six training, two validation and two development layouts. These published layouts were used in earlier baseline work: this is a development pilot, not an independent paper test set.

The plan first requires a 20-tick real-simulator capture/parity smoke. It then collects ten episodes per task, up to 320 ticks each. Alternate chunks receive small translation noise for response identification; collection success is therefore not a frozen-reference benchmark.

Run and Inspect the Pilot
------------------------

After preparing the cases, start the supervisor in a dedicated tmux session:

.. code-block:: bash

   tmux new-session -d -s physrlt_liberopro_1026 \
     'bash /home/luokz/rlinf_rlt/UPT_physrlt_liberopro_dev/toolkits/rlt/physrlt/run_local.sh run --root /mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill/research/physrlt_20261025/pilot-v1 --gpus 0 1'

The supervisor waits for a card to be idle for 30 seconds and launches only on GPUs 0 or 1. Each job has a hang-detection timeout; the finite campaign has no 12-hour global cutoff. A failed job blocks its descendants, while independent jobs continue. A supervisor restart with a recorded running job requires an explicit process/output audit.

Read ``status.json`` for job state, ``stdout.log`` for failures, and each job's result JSON for completion. Offline ``table.csv`` files compare prediction errors. ``control_summary.csv`` separates reference, warmup, online training and frozen evaluation; each entry retains its split and denominator. No W&B/network service is required to retain these local records.

Interpret the Comparisons
------------------------

Seven readout variants use the same 128-dimensional bottleneck and decoder capacity: action-feature reconstruction, geometry supervision, response supervision, joint supervision, shuffled joint labels, prefix-feature joint supervision, and current-state/reference-only joint supervision. Each receives 400 updates for seeds 31, 47 and 59. A common frozen linear probe tests the resulting representations against state-plus-action and persistence baselines. Its ridge parameter is selected using validation layouts only.

Four readouts enter the control pilot with 1,500 actor BC updates and seeds 31 and 47. The same AlphaBrain actor/critic architecture and hyperparameters are used for all four. A validation gate requires arm MSE below 0.02, gripper disagreement below 0.05 and at least one validation success before up to 1,040 online ticks. Passing different gates does not establish a fair large-scale RL comparison; failed gates are recorded as diagnostics.

The compact feature projection and 128-dimensional readout differ from the original 2048-dimensional RLT encoder. These are matched research variants, not an exact upstream algorithm reproduction. The existing baseline worktree remains unchanged.

Physical and Causal Contracts
----------------------------

The encoder receives only pre-execution features, current proprioception and reference actions. Its consequence decoders receive the actual accepted command prefix, with horizons at three and ten control ticks. Future measurements supervise training only. Unexecuted terminal horizons are masked. Measured joint angles, end-effector/gripper positions and calibrated metric depth have explicit units; object targets use world positions, not object-to-end-effector observations that can change without object motion.

No memory retrieval, external agent or language-model API is added. The implementation enforces native action bounds and measurement validity; it does not assume unknown friction laws or rigid coupling between end-effector and object displacement. Depth and object states are privileged training targets, not extra deployment sensors. The geometry head is a compact short-horizon predictor, not PrismWAM's depth diffusion model.

Prediction improvements must be followed by repeated closed-loop gains. This pilot cannot establish reduced human intervention, cross-generator generality or throughput gains. Report wall time and inference overhead separately from interaction efficiency, and create a fresh split before formal paper experiments.

Targeted Followups
------------------

The first pilot motivates a second development plan: action-side features are not consistently better than prefix or state/action inputs. Use ``prepare-followup`` with ``--base-root`` pointing to the first pilot and a new ``--root``. Its CPU probes run immediately; its GPU pilots wait until the first campaign finishes. All variants reuse the recorded data, so these are iterative development comparisons rather than independent confirmation.

``hybrid_recon`` and ``hybrid_joint`` allocate half the fixed feature width to each pathway. ``joint_recon`` retains a 0.1-weight reconstruction loss. ``factorized_joint`` gives geometry and response decoders separate latent halves while retaining the full latent for the policy. ``residual_joint`` predicts the response left unexplained by an empirical state/action predictor, fitted without the target training episode. Its residual prior is not a known physics law. ``state_action_joint`` is repeated as the simple-input control. Followups retain 400 representation updates, three offline seeds and a one-seed control screen.

The diagnostic ``token_without_candidate`` omits the accepted command from the probe, but its token still contains reference intent. It is not a fully action-free baseline. Correct counterfactual effects require matched interventions; observational prediction errors alone cannot establish them.

Corrected Control and Physical Outputs
-------------------------------------

Control protocol version 2 keeps the critic's replay, actor and target inputs in the same policy-proposal action domain. Accepted native commands are retained separately for physical supervision. The first pilot's online results were invalidated after this mismatch was found; its transition data, offline probes, reference and BC evaluations remain usable. Check INVALIDATION.json before interpreting any earlier online result.

Use ``prepare-control-check`` with ``--base-root`` and ``--after-root`` to queue corrected matched controls after the followup campaign. In addition to the four basic readouts, ``joint_outcomes`` exposes predicted normalized joint/EE/gripper motion, pooled depth change and object motion directly to the policy; ``joint_mixed`` combines these with half the latent. Both use reference-conditioned predictions, never future measurements. They keep inference decoders, so any latency claim must count those decoders.

The final reference screen changes translation-command gain or delays command execution on training layouts. This is custom software actuation stress, not an official LIBERO-Pro score or a friction/stiffness change. Delayed online RL is rejected until pending commands are included in its state. A job with no log progress for ten minutes is stopped; a launch-time GPU occupancy race receives a bounded retry.

Preserving the Action Prior
---------------------------

Use ``prepare-anchor-check`` with the same ``--base-root`` and an ``--after-root`` pointing to the corrected campaign to test whether physical supervision discarded useful action cues. Four arms combine a frozen reconstruction readout with reconstruction, geometry, response or joint readouts. A fixed projection maps the concatenated latents back to 128 dimensions. Every arm uses two 400-update encoders and the same actor; the dual-reconstruction control uses different initialization seeds to avoid a duplicated, rank-deficient token. This tests complementary representations, not extra policy capacity or a guaranteed physics benefit.
