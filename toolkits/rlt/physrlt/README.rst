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

Joint angles and end-effector positions are related by forward kinematics under ideal calibration; they are not presented as independent information sources. Collection retains seven arm-joint positions, three end-effector coordinates and two gripper positions for controller and calibration diagnostics. A matched ablation changes only response supervision: all 12 fields, arm joints plus gripper, or end effector plus gripper. ``arm_q_gripper_joint`` and ``eef_gripper_joint`` retain the same inputs, 128-dimensional bottleneck, geometry loss and update budget.

After the offline ablation, use ``prepare-response-check`` to create a separate closed-loop screen. All three arms share the training seed, validation/development layouts and 1,500 actor BC updates. ``online-ticks`` is fixed to zero so Q+BC cannot amplify an unverified target difference. This screen checks whether redundant response supervision changes BC control; it is not a final success-rate experiment.

``balanced_joint`` retains all three response views but computes arm-joint, end-effector and gripper errors separately before averaging the three groups. It tests whether dimension-wise averaging overweights the seven joint coordinates. This remains multi-view auxiliary supervision; it does not present FK-related quantities as independent physical information.

Explicit ``outcomes`` and ``mixed`` policy inputs use only response fields supervised by the selected ablation. Omitted decoder rows are replaced by their normalized training mean, so random, untrained outputs cannot enter the actor or critic. Latent-only deployment is unchanged.

No memory retrieval, external agent, language-model API or contact-sensor supervision is added. The implementation enforces native action bounds and measurement validity; it does not use simulator-only contact pairs, impulses or friction parameters, assume unknown friction laws, or impose rigid coupling between end-effector and object displacement. Depth and object states are privileged training targets, not extra deployment sensors. The geometry head is a compact short-horizon predictor, not PrismWAM's depth diffusion model. Here, consolidating experience means learning from consequence-prediction targets; it is not teacher-student knowledge distillation or sim-to-real by itself.

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

Repeating on New Development Layouts
-----------------------------------

``prepare-repeat`` queues two training seeds (47 and 59), six new development layouts (10 through 15), and nominal or 0.7 translation-command gain. It compares reconstruction against joint physical supervision separately for action-only and fused features. This finite plan has 16 jobs, each with two methods, 1,500 BC updates and at most 1,040 online ticks per method. Its readouts remain trained on nominal data: the gain comparison tests transfer and short RL adaptation, not a representation already trained on changed physics. Each method uses the same task, initial states, training seed and budget within its matched comparison.

The new layout identities must not overlap earlier pilot layouts. Suite, task, published state and perturbation define that identity; collection reset hashes are retained as provenance because byte-exact simulator states are not portable across runtime or asset revisions. Within one control run, the first reset of each case is the paired reference and every later method must match its numeric simulator state within an absolute tolerance of ``1e-9``. Validation retains the original validation split. These additional published layouts are development data, not a certified untouched paper test. Repeated seeds on the same layouts do not increase the count of unique initial states. Report all methods and conditions, including failed validation gates and regressions.

Reading the Evidence Ledger
---------------------------

Run ``python -m toolkits.rlt.physrlt.report --root <campaign-parent>`` to generate ``overview/index.html``, a JSON ledger and CSV tables from raw artifacts. Add ``--watch`` in a separate tmux session to refresh every minute until the prepared campaigns are terminal. Use only one report writer for a given parent directory. The report retains partial jobs, training splits, unique-layout counts and invalidated online results. It never replaces missing runs with expected scores or combines repeated reference evaluations into independent samples.

Checking Candidate-Action Effects
--------------------------------

``prepare-paired-actions`` requires completed first-pilot data and readouts, then queues a no-training diagnostic on idle authorized cards. It replays the recorded command prefix, checks proprioception and geometry parity, and executes two opposite, bounded translation changes from the same state. It compares predicted outcome differences against measured differences and a zero-effect control. Raw measurements and predictions are retained in ``measurements.pt``. Joint, shuffled-label and state/action decoders are compared; the untrained decoder of a reconstruction-only model is deliberately excluded. At most twelve training-layout states are used, so this tests local action sensitivity, not unseen-condition generalization or task success.

The default ``--paired-split train`` reproduces that diagnostic. Set ``--paired-split development`` when preparing a new campaign to reserve the original development layouts for paired-action collection. These layouts were excluded from readout fitting, but remain development data rather than a final paper test.

``python -m toolkits.rlt.physrlt.effect_probe`` consumes the immutable transition data, ``measurements.pt``, its completed measurement-only ``result.json`` and one ``joint`` checkpoint. It cross-fits the paired labels by layout: one paired layout updates the model and the other evaluates it, then the roles reverse. The base model, matched pointwise supervision, correct effect-difference supervision and sign-reversed effect control start from the same checkpoint. ``--steps`` is capped at 400. The output includes source hashes and every fold; it is paired-label development evidence only because the base readout had already seen ordinary transitions from both training layouts.

After a separate development-split collection completes, pass ``--evaluation-measurements`` and ``--evaluation-paired-result`` together. The module then fits all training-layout pairs and evaluates only on development layouts that were excluded from readout fitting and paired-effect updates. It rejects incomplete artifacts, split mismatches, layout overlap and interventions whose declared magnitude changed under action clipping. This is stronger outcome-prediction evidence, but still does not establish task success or online-RL benefit.

An independent-development run also writes pointwise, correct-effect and reversed-effect readouts under ``checkpoints/``. Each checkpoint records the transition-data digest, paired-measurement digest, supervision arm, update count and seed. Pass selected files to ``control`` with explicit ``--external-readouts LABEL=PATH`` entries. Labels listed in ``--candidate-critics`` keep the actor input unchanged but replace the critic state with a 128-D mixed view of the current latent and the frozen decoder's predicted consequences for each replay, target or actor-proposed action. The environment still receives hard binary gripper commands; actor gradients use a bounded straight-through surrogate only inside the frozen consequence model. Each control output records the readout digest and conditioning protocol.

Labels listed in ``--relative-critics`` instead condition the critic on the predicted consequence difference between its candidate and the frozen reference action. The same frozen decoder evaluates both actions, so this cancels action-independent level bias and aligns the deployment interface with paired-effect supervision. The critic still receives the candidate action separately, and the actor remains unchanged. Absolute and relative labels are mutually exclusive; compare pointwise and paired-effect decoders under the same relative interface before attributing a gain to effect supervision.

``--correction-limit`` enables a separate reference-anchored control variant. The actor still predicts a full chunk, but the command used by the environment and critic is transformed to ``reference + limit * tanh((raw - reference) / limit)``. BC remains on the raw actor output, and target-policy smoothing is bounded by the same transform. The option is disabled by default, so baseline behavior is unchanged. Report it as a matched actor-parameterization control, not as evidence for the physical readout: a gain shared by pointwise and paired-effect arms belongs to reference anchoring, while an effect-supervision claim additionally requires the paired-effect arm to beat its matched pointwise arm.

Set ``--parameter-scope decoders`` in ``effect_probe`` for the first control gate. Pointwise, effect and reversed-effect updates then freeze the encoder and reconstruction head and modify only the response and geometry decoders. Their actor tokens remain exactly equal to the starting ``joint`` readout, so a later control difference cannot be attributed to changing the actor representation. The default ``all`` scope is retained as a representation-learning diagnostic and must be reported separately.

This candidate-aware critic is a bounded development test, not a new result by construction. Compare the original latent critic, candidate conditioning with the original joint readout, matched pointwise conditioning and paired-effect conditioning under the same warmup state, online interactions and evaluation layouts. BC warmup must pass the existing action-fidelity and validation-success gate before any online updates. A prediction improvement alone does not authorize a larger training budget.

Each supervisor now claims the shared research GPU lease before launching, and the child inherits it until exit. The progress watchdog accepts episode/effect artifacts as well as stdout. Earlier pilot-v3 supervisors loaded the stdout-only watchdog before this correction and stopped healthy partial runs; the completed protocol-v2 episodes remain valid development data. ``prepare-control-recovery`` repeats only the incomplete mixed-token arm and the blocked reference stress screen at a new output directory. A process interruption is not an algorithmic negative result.

The old pilot-v4 supervisor also interrupted the task-0 anchor run. ``prepare-anchor-recovery`` recovers its response and joint arms only; the completed task-3 run and task-0 reconstruction/geometry arms remain available. Recovery always uses a new directory and preserves the original failure record.
