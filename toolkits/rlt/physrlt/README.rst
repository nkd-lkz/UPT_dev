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

The optional ``flow_recon`` and ``flow_joint`` variants retain the same 512-dimensional input and 128-dimensional bottleneck, but summarize four evenly spaced action-expert states from the frozen Pi0.5 flow-matching trajectory instead of reading only its final forward. They require a newly collected dataset containing ``flow_feature``; old artifacts fail closed rather than silently substituting the final-state feature. This is a representation ablation, not a claim that denoising time is physical execution time.

Four readouts enter the control pilot with 1,500 actor BC updates and seeds 31 and 47. The same AlphaBrain actor/critic architecture and hyperparameters are used for all four. A validation gate requires arm MSE below 0.02, gripper disagreement below 0.05 and at least one validation success before up to 1,040 online ticks. Passing different gates does not establish a fair large-scale RL comparison; failed gates are recorded as diagnostics.

The compact feature projection and 128-dimensional readout differ from the original 2048-dimensional RLT encoder. These are matched research variants, not an exact upstream algorithm reproduction. The existing baseline worktree remains unchanged.

Physical and Causal Contracts
----------------------------

The encoder receives only pre-execution features, current proprioception and reference actions. Its consequence decoders receive the actual accepted command prefix, with horizons at three and ten control ticks. Future measurements supervise training only. Unexecuted terminal horizons are masked. Measured joint angles, end-effector/gripper positions and calibrated metric depth have explicit units; object targets use world positions, not object-to-end-effector observations that can change without object motion.

Joint angles and end-effector positions are related by forward kinematics under ideal calibration; they are not presented as independent information sources. Collection retains seven arm-joint positions, three end-effector coordinates and two gripper positions for controller and calibration diagnostics. A matched ablation changes only response supervision: all 12 fields, arm joints plus gripper, or end effector plus gripper. ``arm_q_gripper_joint`` and ``eef_gripper_joint`` retain the same inputs, 128-dimensional bottleneck, geometry loss and update budget.

Before conditioning a policy on measured execution, run ``gain_probe`` as a no-training identifiability check. It restores the same recorded simulator state, issues the same command chunk, and changes only the executed translation gain. The score projects measured end-effector displacement onto the issued translation direction. The held-out development gate requires at least six pairs per horizon, 75% paired ordering accuracy, and a positive mean nominal-minus-reduced margin at both three and ten ticks. This establishes that a completed transition carries local actuation information; it does not establish a control benefit or permit future measurements in the current decision.

``response_adaptation`` is the next closed-loop screen, not the final RLT method. It calibrates a bounded translation correction from preceding completed chunks and never reads the configured execution gain. The matched reference applies no correction, ``fixed_high`` always applies the maximum correction without reading response, and ``delayed_high`` applies that same correction only after the adaptive method's minimum observation count. ``oracle`` is only a bounded fixed gain-compensation control rather than an optimal policy. ``delayed_oracle`` waits for the same number of completed chunks as the adaptive method and then applies the declared gain compensation without reading their responses; it isolates response estimation from correction timing. The ``reversed`` arm deliberately applies the response estimate in the wrong direction. The nominal response score must come from separate training layouts. Run two development layouts first, freeze the settings, and only then evaluate the remaining layouts by explicit offset.

Passing ``--nominal-dataset`` replaces the global nominal score with a command- and proprioception-conditioned ridge prediction of the nominal 3-D end-effector displacement. Ridge strength is selected on validation layouts; development and confirmation rows are excluded before feature construction. The run fails closed unless this predictor beats the zero-motion baseline on validation and has positive displacement-direction cosine. Each completed chunk compares measured displacement with the prediction scaled by the known issued-action correction, yielding independent x/y/z execution-gain estimates without feeding corrected actions back through the regressor. ``--execution-gains`` can apply a declared three-axis stress, while the adaptive arm observes only completed responses. These ratios and their bounded corrections remain a diagnostic Physical Token, not a task-success result.

``--nominal-model diagonal`` replaces the flexible ridge with three fitted command-to-displacement coefficients. It is less expressive, but it must recover an absolute validation gain between 0.75 and 1.25 on every axis. This physically constrained control tests whether per-axis response adaptation beats uniform action scaling when only one execution axis changes.

The causal calibrator accumulates excitation instead of forgetting earlier chunks. A validation-declared 0.8--1.2 deadband keeps nominal axes unchanged; only a sufficiently large, persistent response deviation can alter an axis. This prevents low-motion or contact phases from turning measurement noise into three-axis action amplification.

``object_motion_probe`` tests the sparse object-evolution bottleneck without policy training. It selects the most action-sensitive object from training pairs only, chooses ridge strengths by leave-one-training-layout-out cross-validation, and evaluates once on disjoint development pairs. Four matched inputs are compared: state plus reference action, the final action-expert feature, the four-point flow-path feature, and the VLM prefix feature. It reports direct displacement regression, a motion-event-gated regression, the zero-motion baseline and a sign-reversed prediction control. The motion threshold is declared in metres and should be checked for sensitivity; neither object world positions nor paired outcomes enter deployment features.

``sparse_motion`` turns that diagnostic into a small trainable head over a frozen physical readout. It separates whether an object moves from its conditional displacement and can add paired candidate-effect supervision. Direct regression, event gating, pointwise supervision, paired supervision and a sign-reversed pair control share the same training and development layouts, optimizer budget and target object. This remains offline candidate-consequence evidence: it is not a policy or online-RL gain until the frozen head passes a declared closed-loop evaluation.

Choose paired-intervention timestamps before opening evaluation outcomes. ``tick_selection`` ranks timestamps using only training-layout object displacement, requires a declared number of covered layouts, and enforces temporal spacing between selected points. Pass its ``selected_ticks`` to ``counterfactual --ticks``. The resulting schedule records how measurements were acquired; it is not evidence that the representation or controller improved.

After confirmation, ``control --sparse-motion-heads LABEL=PATH`` can append the frozen head's candidate-minus-reference event and displacement quantities to a critic. ``--sparse-zero-controls LABEL`` keeps the same 256-dimensional critic capacity but replaces that physical channel with zeros. The loader verifies that the sparse head was trained from the exact physical-readout checkpoint used by the control arm; a mismatched digest fails before rollout.

After every bounded control screen, run ``control_report --runs PATH ...`` to preserve each correction axis, method, split denominator, mean ticks, positive replay count and skipped-update reason in one CSV and JSON report. The reporter accepts only complete immutable runs and keeps validation separate from development. It summarizes the declared experiment; it does not select a winning method or merge different axes into a success claim.

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

``prepare-paired-actions`` requires completed first-pilot data and readouts, then queues a no-training diagnostic on idle authorized cards. It replays the recorded command prefix, checks proprioception and geometry parity, and executes two opposite, bounded translation changes from the same state. It compares predicted outcome differences against measured differences and a zero-effect control. Raw measurements and predictions are retained in ``measurements.pt``. Joint, shuffled-label and state/action decoders are compared; the untrained decoder of a reconstruction-only model is deliberately excluded. The default pilot uses at most twelve states from two training layouts. A declared contact-phase extension may cover all six training layouts, but it remains a local action-sensitivity test rather than evidence of task success.

Uniform execution ticks can overrepresent free-space motion and make zero object displacement a strong baseline. Contact-phase acquisition must declare its ticks and retain disjoint-layout evaluation. ``object_effect`` updates only the geometry decoder from measured differences in object displacement; ``reversed_object_effect`` supplies the sign-reversed negative control. The encoder, reconstruction head and response decoder remain fixed. This arm advances to control only if the correct direction beats the joint effect, reversed direction and zero-effect baselines.

The default ``--paired-split train`` reproduces that diagnostic. Set ``--paired-split development`` when preparing a new campaign to reserve the original development layouts for paired-action collection. These layouts were excluded from readout fitting, but remain development data rather than a final paper test.

``python -m toolkits.rlt.physrlt.effect_probe`` consumes the immutable transition data, ``measurements.pt``, its completed measurement-only ``result.json`` and one ``joint`` checkpoint. It cross-fits the paired labels by layout: one paired layout updates the model and the other evaluates it, then the roles reverse. The base model, matched pointwise supervision, correct effect-difference supervision and sign-reversed effect control start from the same checkpoint. ``--steps`` is capped at 400. The output includes source hashes and every fold; it is paired-label development evidence only because the base readout had already seen ordinary transitions from both training layouts.

After a separate development-split collection completes, pass ``--evaluation-measurements`` and ``--evaluation-paired-result`` together. The module then fits all training-layout pairs and evaluates only on development layouts that were excluded from readout fitting and paired-effect updates. It rejects incomplete artifacts, split mismatches, layout overlap and interventions whose declared magnitude changed under action clipping. This is stronger outcome-prediction evidence, but still does not establish task success or online-RL benefit.

An independent-development run also writes pointwise, correct-effect and reversed-effect readouts under ``checkpoints/``. Each checkpoint records the transition-data digest, paired-measurement digest, supervision arm, update count and seed. Pass selected files to ``control`` with explicit ``--external-readouts LABEL=PATH`` entries. Labels listed in ``--candidate-critics`` keep the actor input unchanged but replace the critic state with a 128-D mixed view of the current latent and the frozen decoder's predicted consequences for each replay, target or actor-proposed action. The environment still receives hard binary gripper commands; actor gradients use a bounded straight-through surrogate only inside the frozen consequence model. Each control output records the readout digest and conditioning protocol.

Labels listed in ``--relative-critics`` instead condition the critic on the predicted consequence difference between its candidate and the frozen reference action. The same frozen decoder evaluates both actions, so this cancels action-independent level bias and aligns the deployment interface with paired-effect supervision. The critic still receives the candidate action separately, and the actor remains unchanged. Absolute and relative labels are mutually exclusive; compare pointwise and paired-effect decoders under the same relative interface before attributing a gain to effect supervision.

``--correction-limit`` enables a separate reference-anchored control variant. The actor still predicts a full chunk, but the command used by the environment and critic is transformed to ``reference + limit * tanh((raw - reference) / limit)``. BC remains on the raw actor output, and target-policy smoothing is bounded by the same transform. The option is disabled by default, so baseline behavior is unchanged. Report it as a matched actor-parameterization control, not as evidence for the physical readout: a gain shared by pointwise and paired-effect arms belongs to reference anchoring, while an effect-supervision claim additionally requires the paired-effect arm to beat its matched pointwise arm.

``--correction-axes`` can further restrict that correction to explicitly listed action dimensions; all other dimensions equal the reference exactly and receive no Q gradient through the actor. This is a support-matching diagnostic, not the intended general method. In particular, paired actions collected only on dimension zero do not justify an effect decoder guiding arbitrary rotation, translation or gripper corrections. A full method must collect paired evidence across the action subspace it permits the policy to change.

``--residual-output`` is a separate actor-parameterization control. It zero-initializes the final layer and interprets the actor's native output as a bounded residual around the reference; BC therefore targets a zero residual. Every representation starts from exactly the same reference behavior, which prevents random error from learning a full-action copy from obscuring token differences. This is not the original RLT actor and must be reported separately from the default direct-action output.

``--deferred-online-updates`` collects the bounded online rollout before applying one update per eligible transition. Together with ``--residual-output``, matched representations therefore collect the same initial policy distribution before their critics diverge. Each run records a hash over proposed actions, accepted commands, rewards and termination fields so accidental data-distribution differences remain visible. This diagnostic separates representation utility from immediate on-policy drift; it is not the default online RLT schedule.

Sparse success rewards need one additional safety distinction. ``--require-positive-replay`` skips the deferred actor and critic updates when the matched replay contains no positive transition. The resulting policy remains the zero-residual reference instead of following an ungrounded critic initialized from all-zero returns. Treat this as a separate protocol and rerun its action-reconstruction control; do not compare it directly with an ungated arm.

``--fixed-replays LABEL=PATH ...`` is an offline optimization-seed diagnostic. It requires deferred updates and one replay for every requested method. The loader rejects incomplete chunks, non-finite tensors and missing transition fields, records the source digest, and keeps the replay trace visible in the comparison audit. Use representation-specific re-encodings of one matched physical trajectory; never reuse another representation's latent tensors. This protocol tests whether representations learn differently from identical positive evidence, not online exploration performance.

When closed-loop success is sparse, ``critic_audit`` checks whether the critic learned to rank returns at all. It loads an immutable online checkpoint and a held-out replay, reconstructs sparse success returns at episode boundaries, and reports Q-ranking AUC, Monte Carlo return error, twin-Q disagreement, and the actor's Q advantage over the reference. An all-zero replay can reveal unjustified optimism but cannot provide an AUC. The audit never treats training-replay fit as generalization evidence or updates the model from validation or development outcomes.

The original ``candidate_relative`` interface replaces half of the 128-D latent with a 64-D relative consequence, changing both physical information and representation capacity. ``--concat-control-critics`` and ``--concat-relative-critics`` provide a capacity-matched diagnostic: both critics receive the complete 128-D token plus a 128-D channel. The control channel is zero, while the physical channel contains the predicted consequence difference between a candidate and the reference. The actor still reads only the base token, and the replay and update budget remain matched. Explicit consequences support a critic claim only when the physical arm improves held-out Q ranking and closed-loop results across seeds relative to the zero-channel control.

``--unit-concat-relative-critics`` additionally normalizes the direction of the 68 active consequence coordinates while the critic continues to read action magnitude from the candidate itself. A sign-reversed decoder is required as a negative control. To prevent physical supervision from also changing the actor's base token, ``--critic-readouts LABEL=PATH`` selects a separate Physical Token encoder, and ``--fixed-critic-replays`` attaches its row-matched view to fixed replay. The loader requires identical hashes for actions, accepted commands, rewards, and terminations across the two views.

When actor gradients are too sensitive to a sparse-reward critic, ``--conservative-selectors`` evaluates a frozen checkpoint without collecting more training data. It interpolates a fixed set of candidates between the reference and actor proposal, scores them with the lower Q minus a twin-Q disagreement penalty, and executes a correction only when the best candidate beats the reference by ``--candidate-selection-margin``. The action control and Physical Token arms must share the candidate count, margin, disagreement penalty, and checkpoint. This diagnostic tests candidate ranking, not a benefit from extra interaction.

``--actor-zero-concat`` and ``--actor-physical-concat`` test whether the grounded readout must enter the action-generation pathway rather than only the critic. Both variants use a 256-D actor input: the capacity control concatenates the complete 128-D base token with zeros, while the physical arm concatenates the same base token with a separate 128-D Physical Token. The executed replay, actor and critic update counts, and base token remain matched. A physical-representation claim requires the physical arm to outperform the zero-channel control across optimizer seeds; a gain over the original 128-D actor alone cannot separate physical information from added model capacity.

``--actor-physical-outcomes-concat`` keeps the same 256-D contract but replaces the physical encoder latent with the frozen decoder's explicit normalized consequences for the reference action. This separates a supervision-shaped latent from a WAM-like predicted-outcome representation. It uses only the current observation, proprioception, reference chunk, and frozen model prediction at deployment; measured future response and privileged geometry remain training labels. Compare it with both ``--actor-zero-concat`` and ``--actor-physical-concat`` before attributing a result to explicit physical quantities.

``--actor-physical-gates`` applies a learned scalar gate to the appended half of selected actor tokens. The gate is zero at initialization, so physical and zero-channel arms produce exactly the same initial action under matched actor weights; Q+BC updates can admit the frozen physical channel only when its gradient is useful. Apply the gate to every arm in a matched comparison, including the zero-channel capacity control, and report the learned gate alongside frozen-evaluation outcomes. This is a protection for the reference action prior, not independent evidence that the physical quantity is useful.

Set ``--parameter-scope decoders`` in ``effect_probe`` for the first control gate. Pointwise, effect and reversed-effect updates then freeze the encoder and reconstruction head and modify only the response and geometry decoders. Their actor tokens remain exactly equal to the starting ``joint`` readout, so a later control difference cannot be attributed to changing the actor representation. The default ``all`` scope is retained as a representation-learning diagnostic and must be reported separately.

This candidate-aware critic is a bounded development test, not a new result by construction. Compare the original latent critic, candidate conditioning with the original joint readout, matched pointwise conditioning and paired-effect conditioning under the same warmup state, online interactions and evaluation layouts. BC warmup must pass the existing action-fidelity and validation-success gate before any online updates. A prediction improvement alone does not authorize a larger training budget.

Each supervisor now claims the shared research GPU lease before launching, and the child inherits it until exit. The progress watchdog accepts episode/effect artifacts as well as stdout. Earlier pilot-v3 supervisors loaded the stdout-only watchdog before this correction and stopped healthy partial runs; the completed protocol-v2 episodes remain valid development data. ``prepare-control-recovery`` repeats only the incomplete mixed-token arm and the blocked reference stress screen at a new output directory. A process interruption is not an algorithmic negative result.

The old pilot-v4 supervisor also interrupted the task-0 anchor run. ``prepare-anchor-recovery`` recovers its response and joint arms only; the completed task-3 run and task-0 reconstruction/geometry arms remain available. Recovery always uses a new directory and preserves the original failure record.

Run a Medium-Difficulty Gate
----------------------------

Use one nonzero, non-saturated LIBERO-Pro cell before increasing seeds or tasks. ``medium_gate`` prepares task 5 with the object perturbation by default, matching the 40% frozen-reference cell in the baseline audit. It reserves states 0--5 for training, 6--9 for validation, 30--31 for paired-effect development, and 32--35 for closed-loop development. States 10--29 remain untouched until the configuration passes the development rule.

Prepare the immutable cases and reviewed queue, then validate the generated commands before launching it:

.. code-block:: bash

   python -m toolkits.rlt.physrlt.medium_gate prepare \
     --root <experiment-root> \
     --source <alphabrain-source> \
     --assets <pi05-checkpoint> \
     --python <libero-python> \
     --code "$PWD" \
     --gpu 2

   python -m toolkits.rlt.experiment_queue \
     --plan <experiment-root>/plan.json \
     --output <experiment-root>/queue \
     --check

The plan first reproduces the frozen π₀.₅ and AlphaBrain RLT controls with 400 encoder updates, 1,500 actor BC updates and 1,040 online control ticks. It then compares a matched action-reconstruction readout, a physical latent, a pointwise consequence decoder and a paired-effect decoder under the same bounded residual actor. The paired-effect arm advances only when it gains a development success over the pointwise arm, or ties its successes while reducing capped mean ticks by at least 5% without a validation regression.

After the queue finishes, build the common table from recorded episodes:

.. code-block:: bash

   python -m toolkits.rlt.physrlt.medium_gate summarize \
     --root <experiment-root>

``comparison.csv`` and ``comparison.md`` retain pending and negative rows as well as gains. A positive one-seed development gate authorizes seeds 47 and 59; it does not authorize a paper claim. Freeze the configuration before evaluating states 10--29 once.
