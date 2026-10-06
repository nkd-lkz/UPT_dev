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

The latter cases first execute a planner prefix. Held recovery creates a fresh planner at the insertion boundary; dropped recovery opens the gripper for 15 real ticks before replanning. They are controlled recovery fixtures, not failures sampled from the baseline. NPZ traces store actual commands, joint endpoints, rewards, terminal flags and command source; perturbation actions are not expert labels. ``--video --gpu 2`` requires an idle rendering GPU. CPU video rendering was unsupported in the tested SAPIEN build and video validation is pending.

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

Old FLARE caches have no reference predictions. ``--reference-mode zero-diagnostic`` permits an explicit zero-reference ablation, not a substitute for normal actor acceptance. On the development cache, 9 training and 3 validation episodes gave validation MSE 0.191001 at initialization and best MSE 0.028592 over 500 updates. This establishes learnability of the diagnostic mapping, not autonomous task success.

Run a Matched Online Pilot
--------------------------

Only after those checks should you compare assistance off and on. Both use ``rlt_planner_pilot``: one training environment, 20 outer steps, 512 warmup updates, and at most 128 updates per outer step. Evaluation uses 20 distinct seeds, 12026 through 12045, without assistance; it does not repeat a single fixed initial state 20 times. The explicit seed sequence cycles for later evaluations. These engineering-scale settings differ from the formal 64-environment baseline and must not be compared as a matched reproduction of it. Check the learner's warmup completion and actual actor-control counts before interpreting any score as a trained-actor result.

.. code-block:: bash

   python -m toolkits.rlt.planner_experiment --check --gpu 2 --arm planner \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --output /path/to/new/stage2-planner

Remove ``--check`` to execute after the selected GPU and storage are available. Use ``--arm none`` with a separate new output directory for the matched unassisted run. The launcher refuses busy GPUs, requires free disk space, creates a private Ray head, and cleans up only its own child processes. It never stops another Ray cluster. Evaluation disables both planner assistance and the optional model expert.

Executed planner actions and provenance survive transition transport and replay. They supervise BC and enter the critic as actual actions. The original pre-action reference stays unchanged. ``planner_mask_ratio`` and ``bc_planner_loss`` are separate from human metrics. Failed attempts and their real transitions remain available; they are not relabeled as success.

Read Results and Limits
-----------------------

On 2026-10-06, CPU PhysX/MPLib probes completed 3/3 reset fixtures, 6/6 held fixtures and 6/6 controlled-drop fixtures. Component and real single-environment tests cover intervention triggers, actual-action replay, provenance, reference leakage, terminal freezing and reset. These small development counts do not establish arbitrary recovery or learner improvement. GPU reference export, distributed online training and post-training autonomous improvement remain unverified.

Use ``failure_before_actor_phase``, ``failure_after_actor_phase``, ``success_before_actor_phase`` and ``success_with_actor_phase`` to localize the baseline bottleneck. Planner-assisted approach transitions can improve training-state coverage; they do not change the evaluation gate. An actor that never controls the approach cannot directly repair the frozen reference policy's approach failures. Report autonomous success, planner steps and attempts, failure categories and extra expert-data budget separately. Never train on held-out evaluation failures and reuse them as an untouched test set.
