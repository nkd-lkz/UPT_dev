Verify Planner-Assisted RLT
===========================

Verify whether a bounded simulator planner can supply correctly recorded RLT corrections before starting a larger assisted-training experiment. This research branch separates planner feasibility, offline actor fitting, online integration, and autonomous evaluation. It does not replace the unassisted baseline or represent human intervention.

When Control Changes
--------------------

The original reference policy handles the approach. With the baseline auto gate, the environment requests the actor only when the peg is consecutively grasped, the task is not successful, the peg-head hole-frame x coordinate is at least -0.16 m, and the absolute y and z coordinates are each within 1.5 hole radii. Entry normally latches until episode end. The rollout also checks the learner schedule: the formal baseline requires 30,000 post-collection learner updates before actor control. This is not 30,000 environment steps.

The pilot adds a separate training-only supervisor. It samples current and past evidence at action-chunk boundaries. Its defaults are:

.. list-table::
   :header-rows: 1

   * - Trigger
     - Criterion and recipe
   * - Approach timeout
     - No critical-phase entry by 180 control ticks: regrasp if needed, align, insert.
   * - Lost grasp
     - A previously observed grasp disappears: attempt regrasp inside the supported workspace.
   * - Insertion stall
     - In the critical phase with a grasp, x minus transverse error fails to improve by 0.003 m for 30 ticks: withdraw to a pre-insertion pose, realign, insert.

At 10 Hz these are simulation-time budgets of 18 and 3 seconds, not wall-clock latency. Each episode permits one attempt and at most 300 planner ticks. Invalid states, unsupported controllers, lost grasp during the recipe, planning failure and budget exhaustion are not silently retried. The recipe returns control after a recorded failure; ordinary environment termination and timeout remain in force. Collision alone is not a failure detector. There is no rewind.

Check Actions Before Training
-----------------------------

The planner reads privileged object poses and geometry, but its commands go through the existing normalized ``pd_joint_delta_pos`` controller. Each command is the clipped difference between the next planned joint target and the actual current joints, divided by 0.1. No hidden ``env.step`` or reset occurs inside the planner.

Run a small training-side feasibility probe from the branch root with the shared environment activated. A working Vulkan loader/ICD is needed even without rendered observations; CPU tests used Lavapipe. The output directory must be new.

.. code-block:: bash

   python -m toolkits.rlt.planner_probe \
     --output /path/to/new/planner-reset --seeds 2026 2027 2028
   python -m toolkits.rlt.planner_probe \
     --output /path/to/new/planner-held --case held_recovery \
     --seeds 2026 2027 2028 2029 2030 2031
   python -m toolkits.rlt.planner_probe \
     --output /path/to/new/planner-drop --case dropped_recovery \
     --seeds 2026 2027 2028 2029 2030 2031

The latter cases first execute a planner prefix. Held recovery creates a fresh planner at the insertion boundary; dropped recovery opens the gripper for 15 real ticks before replanning. They are controlled recovery fixtures, not failures sampled from the baseline. NPZ traces store actual commands, joint endpoints, rewards, terminal flags and command source; perturbation actions are not expert labels. ``--video --gpu 2`` requires an idle rendering GPU and selects its PCI address. Additional held-recovery video probes succeeded on training-side seeds 2032, 2033 and 2034 in 128, 123 and 129 ticks, with MP4 recordings including the executed prefix.

Verify the Actor With Existing Demonstrations
---------------------------------------------

Actor input contains an independently predicted VLA reference chunk as well as the RL token and proprioception. Export that reference from the pre-action observation; substituting the demonstration action would leak the label into the input. Run the exporter only after the selected GPU is free:

.. code-block:: bash

   python -m toolkits.rlt.cache_actor_demos \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --output /path/to/new/actor-cache --gpu 2 \
     --episodes 0 1 2 3 4 5 6 7 8 9 10 11
   python -m toolkits.rlt.actor_bc \
     --cache /path/to/new/actor-cache --output /path/to/new/actor-bc --steps 500

The CPU fitter splits complete episodes 3:1, uses bounded deterministic actor outputs, and excludes chunks crossing episode ends. It reports total, arm and gripper errors. ``best_actor.pt`` contains actor parameters only and is deliberately not a resumable Stage 2 checkpoint. Offline validation does not certify closed-loop acceptance or initialize the pilot automatically.

Old FLARE caches have no reference predictions. ``--reference-mode zero-diagnostic`` permits an explicit zero-reference ablation, not a substitute for normal actor acceptance. The new ``reference_cache_v1`` contains genuine pre-action predictions from Stage 1 step 2000. With 9 training and 3 validation episodes, 500 updates reduced validation MSE from 0.191056 to a best 0.025444. The reference itself scored 0.015863 against the same demonstration actions, still better than the small actor. Extending fitting to 3,000 updates only improved the best MSE to 0.025000; final validation MSE rose to 0.029288. Successful fitting therefore does not establish actor acceptance or justify scaling training.

Check Reference Distillation Separately
---------------------------------------

To distinguish imitation of demonstrations from imitation of the reference, use ``--target-source reference`` with the same cached inputs. This replaces only the supervised target, not the input provenance. A 3,000-update CPU distillation run on the same episode split achieved best validation MSE 0.010878 against the reference. That number has a different target from the demonstration MSE above and is not directly comparable. It neither authorizes Stage 2 initialization nor measures autonomous success.

.. code-block:: bash

   python -m toolkits.rlt.actor_bc \
     --cache /path/to/new/actor-cache --output /path/to/new/reference-distill \
     --steps 3000 --target-source reference

Run a Matched Online Pilot
--------------------------

Only after those checks should you compare assistance off and on. Both use ``rlt_planner_pilot``: one training environment, 20 outer steps, 512 warmup updates, and at most 128 updates per outer step. Evaluation uses 20 distinct seeds, 12026 through 12045, without assistance; it does not repeat a single fixed initial state 20 times. The explicit seed sequence cycles for later evaluations. These engineering-scale settings differ from the formal 64-environment baseline and must not be compared as a matched reproduction of it. Check the learner's warmup completion and actual actor-control counts before interpreting any score as a trained-actor result.

.. code-block:: bash

   python -m toolkits.rlt.planner_experiment --check --gpu 2 --arm planner \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --output /path/to/new/stage2-planner

Remove ``--check`` to execute after the selected GPU and storage are available. Use ``--arm none`` with a separate new output directory for the matched unassisted run. ``--steps`` and ``--eval-episodes`` both default to 20 and must match across arms. The launcher sets physical RLinf placement as well as the CUDA mask, refuses busy GPUs and creates a private Ray head. Temporary files use ``--scratch-root /dev/shm`` by default, requiring 4 GiB free; the output filesystem needs 5 GiB. Cleanup targets only owned processes, never another Ray cluster. Evaluation disables both planner assistance and the optional model expert.

After both runs complete, check their actual configurations, distinct seed counts and warmup status with the reporting entry. A reference-dominated rollout before warmup completion is not reported as a comparison of two trained actors.

.. code-block:: bash

   python -m toolkits.rlt.planner_report \
     --none /path/to/completed/stage2-none \
     --planner /path/to/completed/stage2-planner \
     --output /path/to/new/comparison.json

Executed planner actions and provenance survive transition transport and replay. They supervise BC and enter the critic as actual actions. The original pre-action reference stays unchanged. ``planner_mask_ratio`` and ``bc_planner_loss`` are separate from human metrics. Failed attempts and their real transitions remain available; they are not relabeled as success.

Separate Approach Failures From Insertion
------------------------------------------------------------

Keep complete-task evaluation as the main outcome. A separate insertion diagnostic executes a planner prefix after reset, checks that the peg is held near the hole and the task is not already successful, then evaluates the checkpoint without further expert actions. It does not rewind the simulator or reset the episode clock. Prefix ticks consume the original 500-tick horizon and are reported as ``fixture_prefix_ticks``; ``insertion_policy_ticks`` counts subsequent policy execution. An invalid fixture aborts the run rather than silently dropping a seed.

Evaluate the same weights and explicit initial-state seeds in two new directories:

.. code-block:: bash

   python -m toolkits.rlt.planner_experiment --gpu 2 --arm none \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --checkpoint /path/to/final/model.pt --eval-episodes 50 \
     --eval-scope full_task --output /path/to/new/full-task
   python -m toolkits.rlt.planner_experiment --gpu 2 --arm none \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --checkpoint /path/to/final/model.pt --eval-episodes 50 \
     --eval-scope insertion --output /path/to/new/insertion

Checkpoint-only evaluation bypasses learner warmup: full-task evaluation still uses the environment phase gate; insertion evaluation requests the actor immediately. It measures these weights, not training readiness. The latter score is an insertion-subtask result, never a full-task autonomous success rate. The fixture uses privileged geometry, and its distribution need not match states reached by a failing reference policy.

Check Whether Corrections Change the Actor
------------------------------------------------------------

Add ``--export-corrections`` to a fresh training launch to save the transitions actually admitted to replay, with their original pre-action reference, submitted commands, terminal masks and planner provenance. Episodes with no admitted transitions remain in the inventory. The launcher seals the cache only after collection succeeds. Checksums and the frozen-feature contract are verified before fitting; an interrupted run's cache is rejected. For the verified Panda controller only, the diagnostic retains original submissions and separately reconstructs their normalized values after controller clipping to [-1, 1]. It reports clipping counts, never rewrites the sealed cache, and does not change online replay. Other controller mappings are rejected.

Use one sealed cache for both actor objectives:

.. code-block:: bash

   python -m toolkits.rlt.correction_fit \
     --cache /path/to/completed/planner/corrections \
     --output /path/to/new/correction-fit \
     --updates 2000 --warmup-updates 128 --seed 1234

To test whether an existing actor can use corrections, add
``--initial-weights /path/to/actor/model_state_dict/full_weights.pt``.
Both arms strictly load the same finite weights, create fresh optimizers and
targets, and record the initial SHA-256. This is not a resume. The actor must
use the cache's Stage 1 features and normalization. Random initialization and
warm starts are separate experiments; do not combine their MSE results.

The CPU diagnostic uses a chronological 3:1 split of complete episodes, common initialization, identical minibatch indices and matched random draws. Both arms train critics to match optimizer work; only Q+BC uses Q gradients in the actor objective after warmup. There is one actor update per four critic updates. BC targets use actual planner commands in successful episodes and the original reference on non-planner ticks. Failed planner ticks remain in TD data but are excluded from BC; padding after termination is excluded from BC and reward sums. Successful-episode filtering is a coarse quality rule, not proof that every action was optimal. The existing online loss is unchanged and continues to use its original intervention labels.

The report includes held-out correction, arm, gripper and reference MSE, rejected correction counts, command saturation, data and minibatch hashes, and exact optimizer counts. Eligibility requires positive reward in an exported transition: a successful ending outside the recorded phase is conservatively ineligible. Final weights are selected by the declared update budget, not test success. Evaluate ``bc_only/model.pt`` and ``q_bc/model.pt`` with the commands above, keeping the same Stage 1 features and normalization data. Lower offline MSE is not a closed-loop gain. These weights are evaluation artifacts, not resumable distributed checkpoints. At least four nonempty episodes and successful corrections in both splits are required; otherwise collect more training-side data without borrowing evaluation failures.

Compare Real Interaction Budgets Across Seeds
------------------------------------------------------------

Equal episode counts gave the old planner arm fewer environment ticks and more learner updates. New experiments therefore set either ``--control-budget`` or ``--update-budget``. The former counts every real training tick, including reference, learner, planner and handoff holds, and truncates the final episode exactly at the limit. The latter caps cumulative critic updates; actor updates are reported separately. Final evaluation and checkpoint saving are forced at the boundary. ``--steps`` is an episode safety cap in these modes, and reaching it first fails the run. These modes require fresh, synchronous, single-environment training, without resume or overlapping bootstrap.

The campaign entry prints six serial training commands by default. Add ``--execute`` only to start them. Each completed training run is followed by a separate insertion evaluation of its final checkpoint. Existing GPU jobs are never killed; a busy GPU causes a clear refusal, not automatic waiting.

.. code-block:: bash

   python -m toolkits.rlt.planner_campaign \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" --gpu 2 \
     --seeds 1234 1235 1236 --eval-episodes 50 \
     --budget-basis control --budget 30000 --max-episodes 1000 \
     --protocol complete --output /path/to/new/control-budget-campaign

Run a separate campaign with ``--budget-basis updates --budget 2000`` to match optimizer work. Check both actor and critic counts before calling that comparison optimizer-matched. Reports retain each training seed, mean success, across-training-seed standard deviation, paired differences, failure categories, planner attempts/ticks and update costs. The same 50 distinct evaluation seeds, 12026 through 12075, are reused across arms, not added to training. Three training seeds and 50 initial states remain a development experiment, not a significance or generalization guarantee. Insertion results are stored separately. The launcher now sets both training-environment and actor seeds explicitly; old pilots with environment seed 2026 must not be silently pooled with these runs.

Test Bounded Correction as a Separate Protocol
------------------------------------------------------------

The default ``--protocol complete`` retains the existing planner-completes-task behavior. The new ``--protocol preinsert_handoff`` regrasp/realignment recipe stops before insertion, requiring a grasp, an unfinished task, hole-frame x at least -0.16 m and transverse norm at most 0.045 m. It does not promise arbitrary contact recovery. The trigger and 300-tick recipe budget remain bounded.

If the recipe ends or fails mid-chunk, the supervisor holds the current arm position until the next chunk boundary, then returns control to a policy computed from a fresh observation. It never executes the stale remainder proposed before intervention. Up to one chunk of holds is charged to planner and interaction costs, separately recorded as ``planner_handoff_hold_ticks``; ``planner_handoffs`` counts completed handoff recipes, not autonomous successes. The recipient still depends on the reference/actor gate and learner readiness: a handoff alone does not prove learner execution. Use a separate campaign directory and compare against assistance-off under this same declared protocol. Do not combine it with historical complete-recovery results.

Warm Start on Two Available GPUs
--------------------------------

To test whether corrections improve an existing small actor, use a separate weights-only training run. The two-GPU launcher puts the learner on physical GPU 0 and the frozen VLA, rollout and single CPU-physics environment on GPU 1. It requires both GPUs to be idle, leaves GPU 2 alone, uses a private Ray instance, and freezes the runtime source into an archive and a local snapshot before launching.

Select a known readable Stage 2 ``actor/model_state_dict/full_weights.pt`` and a new output directory. Run the command in a tmux session to keep training alive after disconnection:

.. code-block:: bash

   python -m toolkits.rlt.planner_stage2 \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --initial-weights /path/to/stage2/actor/model_state_dict/full_weights.pt \
     --output /path/to/new/planner-stage2 \
     --steps 400 --eval-interval 50 --eval-episodes 50 --save-interval 100 \
     --wandb-entity YOUR_ENTITY

Add ``--check`` to validate composition and the initial actor/critic weights on CPU without launching training. The normal command enables online W&B and TensorBoard, retaining ``launch.json``, ``resolved.yaml``, ``source.tar.gz``, the initial weights and ``train.log`` in the output directory. Frequent backend writes use a separate persistent local directory, ``$HOME/rlinf_rlt/run_metrics/<output-name>``, to avoid remote-filesystem stalls; override it with ``--metrics-dir``. The resolved ``runner.logger.backend_log_path`` and manifest record this location, and the planner report reads it. Keep this directory as well as the checkpoint output. Use filesystems with room for the outputs; videos stay disabled. The default private Ray port is 6575; use ``--port`` for a different free three-port range.

This is not a resume of the original no-assistance baseline: only actor/critic weights carry over; replay, optimizer, target initialization and counters start anew. The target is initialized from those same weights. Warmup first collects 256 replay samples and runs 4096 critic updates, with at most 128 updates per outer round. The existing actor-weight schedule also uses a 4096-update warmup and ramp. Training uses the complete-recovery planner protocol and records its cost. Every 50 rounds it evaluates without planner or model expert on seeds 12026 through 12075, and saves every 100 rounds (also at completion). Evaluation before learner readiness still follows the reference fallback. Report readiness alongside success, and use held-out initial states for a final result after selecting a checkpoint on this development set.

Read Results and Limits
-----------------------

The first 20-step distributed comparison scored 5/20 without assistance and 3/20 after planner-assisted training. The subsequent 60-episode pair scored 4/20 and 7/20, respectively. However, it used 24,091 versus 14,203 real training ticks and 173 versus 257 actor updates; the assisted arm used 4,392 planner ticks. These single-seed, unequal-work pilots do not establish a reliable autonomous gain. The budgeted protocols above are new experiments and must not inherit these scores as evidence of effectiveness.

On 2026-10-06, CPU PhysX/MPLib probes completed 3/3 reset fixtures, 6/6 held fixtures and 6/6 controlled-drop fixtures. The GPU reference export and three additional videos described above also completed. Tests cover Torch/NumPy actions, actual-action replay, provenance, reference leakage, terminal freezing and reset. Distributed probes exposed and corrected NumPy action handling and an inconsistent replay schema at terminal transitions: planner provenance now remains metadata rather than a policy observation. These small development counts do not establish arbitrary recovery or learner improvement. Distributed comparison results and autonomous gains after removing assistance require separate acceptance.

Use ``failure_before_actor_phase``, ``failure_after_actor_phase``, ``success_before_actor_phase`` and ``success_with_actor_phase`` to localize the baseline bottleneck. Planner-assisted approach transitions can improve training-state coverage; they do not change the evaluation gate. An actor that never controls the approach cannot directly repair the frozen reference policy's approach failures. Report autonomous success, planner steps and attempts, failure categories and extra expert-data budget separately. Never train on held-out evaluation failures and reuse them as an untouched test set.

Development acceptance on 2026-10-07 verified exact stopping at 220 and 3,000 real control ticks. The latter collected 12 episodes, 118 admitted transitions and 827 planner ticks, with no online learner update because replay remained below the 128-transition threshold. On this sealed dataset, paired CPU fitting used 500 critic and 125 actor updates per arm, with 8 nonempty training episodes and 3 validation episodes. Held-out correction MSE changed from 0.17130 initially to 0.07227 for BC-only and 0.07894 for Q+BC. However, arm-only MSE worsened from 0.05263 to 0.06772 and 0.07641: the total improvement was driven by the gripper. Validation contained only 65 eligible planner ticks. This does not pass actor capability acceptance or demonstrate autonomous improvement. A separate single-episode insertion smoke test measured 83 prefix ticks plus 417 policy ticks, confirming scope accounting rather than performance.

Both fitted heads subsequently completed 2/2 insertion fixtures on seeds 12026 and 12027, without planner actions after the prefix. Mean policy execution was 26 ticks for BC-only and 21 for Q+BC, following the same mean 87-tick prefix. A constant zero-arm command with the gripper closed did not complete any of three fixtures in the physics regression (seeds 12026–12028). These are small integration checks, not evidence that Q improves success or that complete-task capability improved. The three-training-seed campaigns have not been run. Code verification passed 37 planner tests including real physics, plus a broader CPU regression with 240 passed and 12 skipped. One unrelated Transformers video-metadata compatibility failure, also reproduced on the unchanged baseline, was excluded from that broader run. Both language documentation builds passed.

Evening Continuation: Fixed Weights and Bounded Corrections
------------------------------------------------------------

The 2026-10-07 run ``planner_warm1100_1007_150132_retry2`` finished 400 episodes:
102,648 training ticks, including 31,819 planner ticks, 281 attempts and 83
planner failures; 8,986 critic and 2,247 actor updates. Assisted training success
of 327/400 is not autonomous capability. Expert-free evaluations on 50 fixed
seeds ranged from 30% to 42%, finishing at 18/50. There were 28/50 failures
before actor phase and 4/50 after it. A matched reevaluation of the initial
checkpoint is still required before claiming an improvement.

A same-cache diagnostic starting from baseline step 1100 used the earlier
3,000-tick cache and 2,000 critic/500 actor updates per arm. Held-out correction
MSE was 0.12085 initially, 0.11575 for BC-only and 0.21194 for Q+BC; arm MSE was
0.13782, 0.13126 and 0.23839 respectively. Only three held-out episodes and 65
eligible correction ticks support these numbers. Q degraded following in this
setting; closed-loop benefit remains unmeasured. More online Q updates are not
justified by these diagnostic losses alone.

Repeating the diagnostic with optimizer/minibatch seeds 1234, 1235 and 1236
gave correction MSEs of 0.11575, 0.11504 and 0.11902 for BC-only, versus
0.21194, 0.16522 and 0.17791 for Q+BC. These reuse the same data and initial
weights; they are not three independent online training seeds. The scheduled
closed-loop comparison uses seed 1234, selected before these repeats, rather
than choosing the lowest validation error.

Recovering episode groups from old replay needs additional checks.
``toolkits.rlt.correction_export`` accepts only completed, unevicted,
synchronous single-environment transition checkpoints, groups by actual done
flags and checks adjacent endpoints. A nonterminal recording gap leaves the
export unsealed. The 400-episode replay has such a gap between transitions 109
and 110, so its full episode grouping cannot be reconstructed reliably. This
does not establish a TD endpoint bug, but guessed boundaries cannot support
episode-disjoint validation. New runs use collection-time
``--export-corrections`` to retain actual episode grouping, including empty
replay episodes.

Expand insertion-fixture checks with CPU physics:

.. code-block:: bash

   python -m toolkits.rlt.planner_probe --case insertion_fixture \
     --seeds 12026 12027 12028 --output /path/to/new/fixture-control

The probe executes a planner prefix, checks a held and unsuccessful state, then
holds zero arm increments with the gripper closed until the original episode
ends. Every executed command is saved. ``fixture_ready_rate`` and
``hold_success_rate`` are separate; neither is learner success. A working Vulkan
loader is still required. Use an isolated software Vulkan ICD for CPU checks
rather than another user's occupied GPU.

The expanded CPU probe on seeds 12026–12075 prepared 48/50 valid fixtures.
Seed 12067 exhausted the 300-tick prefix budget; seed 12070 failed grasp
planning. No constant-command continuation succeeded. The overnight insertion
diagnostic therefore declares a conditional 48-seed population using
``--fixture-report /path/to/fixtures50_cpu/summary.json`` with
``--eval-episodes 50``. The latter is the candidate population size in this
mode. All checkpoints use the same 48 selected seeds, and launch metadata
retains the original 50 seeds, both exclusions and the probe hash. Selection
uses only planner feasibility before any learner is evaluated. Runtime fixture
failures still abort the run. Complete-task evaluation continues to use all
50 seeds; its denominator must not be replaced by 48.

The single-GPU pilot also supports ``--initial-weights`` for weights-only
training, ``--wandb`` for W&B logging and ``--checkpoint-run`` to choose the last
budget checkpoint of completed training. Fixed-weight evaluation bypasses
training-readiness counters. Evaluation and initial weights are SHA-256 tagged.
Each run archives runtime sources, including the evaluation entry, and executes
a private snapshot. Metrics go to the local persistent path
``$HOME/rlinf_rlt/run_metrics/<output-name>_<path-hash>``; different seeds never
share event files. NAS retains checkpoints, resolved configuration and launch
metadata.

The machine-specific ``experiments/maniskill_rlt/overnight_gpu0_20261007.json``
plan evaluates initial, planner400, BC-only and Q+BC weights on 50 full tasks
and the 48 verified insertion fixtures each. ``overnight_gpu1_20261007.json`` compares no
assistance with ``preinsert_handoff`` over three training seeds, the same initial
weights and 15,000 actual control ticks per arm, with W&B enabled. Separate
insertion evaluations follow the primary comparisons. These are new protocols;
previous scores are not their results.

The bounded queue is provided by the LIBERO worktree:

.. code-block:: bash

   python ../UPT_libero_dev/toolkits/rlt/experiment_queue.py \
     --plan experiments/maniskill_rlt/overnight_gpu0_20261007.json \
     --output /path/to/new/queue-gpu0 --check

Remove ``--check`` to begin waiting. Replace every output path before reusing a
dated machine-specific plan. The queue uses the explicitly selected physical
GPU, requires two consecutive idle observations and holds a per-GPU lease. Its
twelve-hour total budget includes waiting; each job also has a timeout. Cleanup
targets only owned process groups. This run uses tmux sessions
``rlt_planner_audit_1007`` and ``rlt_planner_handoff_1007``. Execution state is in
the NAS files ``research/planner_overnight_20261007/queue_gpu0_v2/status.json`` and
``queue_gpu1_v2/status.json``. The original queues were interrupted while waiting,
before any GPU job started, to declare fixture coverage. Their diagnostics remain.
Use ``planner_report --evaluation-runs ... --output ...``
for separately scoped checkpoint results or ``--comparisons`` for three completed
training pairs. Queuing, configuration checks and fitting losses do not replace
autonomous evaluation.

The checked-in ``experiments/maniskill_rlt/overnight_results_20261007.json``
indexes completed evidence separately from live queue state. The evening
revision passed 46 planner tests, including CPU physics, and 223 broader CPU
tests with 6 skips. The pre-existing Transformers video-metadata compatibility
case remains explicitly excluded. Offline quality masking is a diagnostic
contract; it does not silently change the distributed online BC rule.
