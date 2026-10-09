Accept the Pi0.5 RLT Baseline Before Migration
============================================================

This guide runs a bounded AlphaBrain RLT/Pi0.5 development campaign on LIBERO and defines the action boundary for a subsequent RoboDojo migration. Use the independent ``research/rlt-pi05-libero-robodojo`` branch in ``UPT_rlt_pi05_dev``. First verify the released VLA, then accept the new RLT encoder and small actor, and only then run matched online-learning pilots. The older Qwen/RLT_a experiments remain separate.

Choose the Checkpoint and Protocol
----------------------------------

The new route freezes a Pi0.5 VLA with a flow-matching action expert. AlphaBrain's ``RLT/encoder_decoder.py`` compresses final VLM features to one 2048-wide token using its encoder and cross-attention reconstruction decoder. The small actor outputs a full 10-by-7 action chunk; online updates do not train the flow-matching expert.

The source is pinned to ``604924beb77b04b0da49326dfae6ea423a27d28a``. The public model is `AlphaBrainGroup/paligemma-pi0-libero-all4suite <https://huggingface.co/AlphaBrainGroup/paligemma-pi0-libero-all4suite>`_ at ``482fcfea4d6a5cee97456f50b1f5aeeae539a1c8``. Despite its legacy class name, its configuration sets ``pi05: true``. It was fine-tuned jointly on four LIBERO suites, so it is an engineering starting point rather than the few-demonstration RLT checkpoint or evidence of unseen-task transfer.

The adapter maps the legacy class name explicitly and loads every weight strictly, skipping random initialization of weights that the checkpoint replaces. It applies the model's MEAN_STD normalization once. The training dataset stores gripper openness in [0, 1], whereas LIBERO executes +1 for closed and -1 for open: protocol v3 enables the upstream ``gripper_remap`` conversion ``1 - 2 * openness`` before command binarization. The dataset's `episode statistics <https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot/blob/main/meta/episodes_stats.jsonl>`_ and state/action rows establish this convention; the release's normalization constants alone do not establish the gripper range. The encoder receives 512 real-image hidden states from the reference's own inference forward. A fixed-noise parity check requires captured and uncaptured reference actions to match exactly.

Every prefix forward must also fill the complete attention KV cache before generating actions. The pinned upstream adaRMS decoder passes ``past_key_values``, while Transformers 4.53.2 attention consumes ``past_key_value``. Without adaptation, the unused keyword silently leaves the cache empty and the action expert cannot attend to the images or instruction. The adapter translates this keyword on the loaded model's legacy attention instances; it changes neither installed dependencies nor the pinned checkout. Tests compare cached suffix inference with joint prefix-and-suffix attention and verify that changing the prefix changes the output. Earlier v1/v2 results contain cache or gripper-interface errors and do not qualify as valid reference baselines.

.. warning::

   This is a new-RLT architecture pilot, not an unmodified upstream training reproduction. Stage 1 uses reference-rollout observations; the actor warmup uses zero reference dropout. Heads share the existing acceptance learner with 2048-wide inputs, ten-step chunks and a signed-gripper threshold. Rewards are scaled by five and discounted by actual executed simulator ticks. These choices are recorded in ``PROTOCOL`` and must stay distinct from paper scores.

Prepare Verified Assets
-----------------------

Use the prepared LIBERO acceptance environment without changing dependencies under existing jobs. ``run_rlt_pi05.sh`` selects the sibling ``.venv-libero-acceptance`` and isolated EGL/dependency directories. Override ``RLT_PI05_VENV`` and ``RLT_EXTERNAL_ROOT`` on another host. Hugging Face lookups remain offline during inference. W&B is disabled by default; ``--wandb online`` explicitly enables it and requires authenticated run creation before computation starts.

Download to a large filesystem; the immutable release contains roughly 9.4 GB of weights. The downloader resumes partial files and verifies Git blob or LFS hashes before writing ``verified.json``:

.. code-block:: bash

   export RLT_PI05_ASSETS=/path/to/large-disk/pi05-assets
   export RLT_PI05_SOURCE=/path/to/AlphaBrain
   python -m toolkits.rlt.pi05_assets --destination "$RLT_PI05_ASSETS"

The source directory must be clean at the pinned revision. The assets directory is read-only to inference; the adapter does not rewrite the released configuration or silently tolerate missing weights.

Run the Gated Campaign
----------------------

The campaign reserves a research queue lease, observes GPU memory and compute processes twice, and runs only when the selected card is idle. It never stops another user's process. Choose an unused output directory and an explicit physical GPU:

.. code-block:: bash

   export RLT_PI05_RUN=/path/to/large-disk/new-pi05-pilot
   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign run \
     --root "$RLT_PI05_RUN" --assets "$RLT_PI05_ASSETS" \
     --source "$RLT_PI05_SOURCE" --gpu 2

The queue has a twelve-hour wall-clock budget, including resource waits. Each GPU job has a two-hour timeout. A failed prerequisite skips its dependants; completion of a process alone does not certify its semantic manifest.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Stage
     - Budget and data
     - Acceptance
   * - Reference collection
     - Task 0, published states 0–9, at most 320 ticks each
     - At least five successes; exact feature-capture action parity
   * - Encoder and BC warmup
     - 300 reconstruction updates, 3000 BC updates; whole-episode 80/20 split
     - Reconstruction and following error decrease; held-out gripper disagreement at most 10%
   * - Closed-loop warmup check
     - Reference and actor on states 10–14
     - Same state hashes; reference succeeds at least three times; actor loses at most one success
   * - Online pilots
     - BC-only and Q+BC restart from the same warmup; states 20–29; five updates per chunk
     - Finite updates and explicit actor/critic update counts
   * - Pilot comparison
     - Reference, warmup-only, BC-only and Q+BC on states 15–19
     - Report every outcome, including zero success

These small development sets detect integration failures. They are not the independent frozen-200 evaluation, a full-suite acceptance, or a statistically powered improvement claim. Preserve a separate test set before choosing hyperparameters or reporting research results.

Expand to Matched Acceptance
----------------------------------------

For a larger comparison, select ``--profile acceptance``. This fixes reset seeds before running and saves generated simulator snapshots for reuse across arms; it does not cycle the 50 published initial states and call them 100 independent cases. ``--task`` supports the preselected Goal tasks 0, 1 and 3. Each campaign covers one task and one training seed, so completion is not a multi-seed or full-suite claim.

The acceptance profile first checks the reference on 20 development initial states, requiring at least 12 successes. It then collects 64 reference rollouts, holds out the last 12 whole episodes for reconstruction and BC diagnostics, and trains the encoder for 1000 updates and the actor for 6000 BC updates. The warmed actor must succeed on at least as many development cases as the reference minus two before online training starts. These are engineering gates, not targets chosen from the final test set.

BC-only and Q+BC start from the identical warmup checkpoint and each execute 20,000 control ticks, with five learner updates per chunk. Settling ticks, episode counts and actual actor/critic updates are recorded separately because successful episodes can end mid-chunk. The reference's stochastic flow sampling is seeded by case and control tick in an isolated RNG context; learner random-number consumption cannot change it. No human, planner or memory assistance is enabled.

Only after both online jobs complete does the queue evaluate reference, warmup-only, BC-only and Q+BC on the same 100 held-out generated initial states. It uses final-budget checkpoints, not the best test score, and rejects mismatched reset hashes in the comparison. ``summary.json`` reports Wilson 95% intervals and paired gained/lost successes. Freeze these results; any test-informed change requires a newly declared holdout before another confirmatory claim.

Start a single task with online logging as follows. Run another task in a separate output directory only on a separately authorized GPU:

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign run \
     --profile acceptance --task 0 --seed 42 --gpu 0 \
     --wandb online --wandb-entity your-team \
     --root "$RLT_PI05_RUN" --assets "$RLT_PI05_ASSETS" \
     --source "$RLT_PI05_SOURCE"

Each acceptance queue retains the twelve-hour total limit, including waits; individual jobs have a four-hour limit within it. Failed gates stop their dependants, and deadline-expired work stays explicitly incomplete. Local ``metrics.jsonl`` files accompany W&B runs in project ``rlt-pi05-libero``; ``wandb_run.json`` stores each run link. Losses, success rates, control ticks and durations are logged alongside the pinned configuration. An online-authentication failure does not silently switch to disabled logging.

Inspect the Results
-------------------

Read ``queue/status.json`` first to distinguish waiting, failed and completed jobs. Each run writes ``manifest.json`` with source, model revision, adapter hashes, package versions, seeds, device mapping and exit status. ``episodes.json`` records reset hashes, success and executed ticks. A passed warmup produces ``warmup/warmup.pt``; successful online execution saves ``q_bc/online.pt`` and ``bc_only/online.pt``.

For a compact report, run the following command. It keeps missing scores distinct from completed zero-success evaluations. When the queue exits, it also writes ``summary.json``:

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign status \
     --root "$RLT_PI05_RUN"

On the development host, ``rlt_pi05_v3_gpu0_1009`` and ``rlt_pi05_v3_gpu1_1009`` manage the v3 queues under ``research/rlt_pi05_20261009``: ``acceptance-v3-task0-seed42`` on GPU 0 and ``acceptance-v3-task1-seed42`` on GPU 1. Read their live status rather than inferring progress from a tmux session. They wait for existing compute processes to exit, not merely for utilization to reach zero. Task 3 and additional training seeds remain follow-up work. Earlier v1/v2 acceptance and CPU/pilot outputs are preserved as compatibility-failure evidence, not accepted baseline scores. The old RLT_a/planner queues were paused between jobs to give this baseline priority; their completed artifacts were not removed. Failure manifests and skipped jobs remain available rather than being relabelled as completed training.

When no GPU is available, ``cpu-reference`` can check the actual simulator loop with CPU inference and an installed OSMesa software renderer. Set ``LD_LIBRARY_PATH`` if OSMesa is unpacked outside the system paths. This mode never reserves a CUDA device; the required ``--gpu`` argument is ignored. It is slow and uses CPU BF16 autocast, so its results do not replace matched GPU evaluations:

.. code-block:: bash

   bash run_rlt_pi05.sh cpu-reference --source "$RLT_PI05_SOURCE" \
     --assets "$RLT_PI05_ASSETS" --output /path/to/new-cpu-smoke \
     --gpu 2 --episodes 1 --state-start 49 --seed 45000

Prepare the RoboDojo Boundary
-----------------------------

Once the LIBERO pilot passes, migrate the model's observation and action contract before adding more training. `RoboDojo <https://github.com/RoboDojo-Benchmark/RoboDojo>`_ currently provides an eval-only simulator/task release, while XPolicyLab owns policy servers. The inspected revisions are RoboDojo ``cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c`` and XPolicyLab ``53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2``.

The initial target is the ARX X5 joint-control interface: head and two wrist RGB cameras, 14 state coordinates, and absolute arm-joint radians plus gripper openings in [0, 1]. Gripper state in the inspected implementation represents commanded openings, not independent measured joint response. A seven-dimensional LIBERO end-effector checkpoint cannot be padded or relabelled into this interface.

``toolkits/rlt/robodojo_boundary.py`` tests the policy-side boundary without Isaac Sim. Construct ``RoboDojoBoundary`` with a compatible backend and identified normalization; call ``reset`` at an episode boundary, ``update_obs_batch`` with explicit environment IDs, then ``get_action_batch`` with the current active IDs. It returns native command dictionaries in the requested order, including after a batch shrinks. Stale observations, invalid grippers and LIBERO-shaped outputs are rejected. This component does not yet load a RoboDojo policy or provide a simulator training loop.

The migration snapshot on this host contains earlier source and bridge prototypes but lacks its original simulator environment and assets. The next acceptance steps are a pinned Isaac runtime and one task's assets, an embodiment-compatible Pi0.5 checkpoint or supervised adaptation, native reference evaluation, and only then an online transition contract with explicit reset, reward, termination and truncation. Keep human/planner assistance and memory disabled for the initial baseline and record their cost separately when added.
