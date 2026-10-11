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

For a larger comparison, select ``--profile acceptance``. This fixes reset seeds before running and saves generated simulator snapshots for reuse across arms; it does not cycle the 50 published initial states and call them 100 independent cases. ``--suite`` selects ``libero_spatial``, ``libero_object``, ``libero_goal`` (default), or ``libero_10`` (Long); ``--task`` selects task 0–9 within that suite. Each campaign covers one task and one training seed, so completion is not a multi-seed or full-suite claim.

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

Schedule All Four Suites
------------------------

To extend the accepted single-task pipeline, use ``toolkits.rlt.pi05_matrix`` from ``research/rlt-pi05-recovery`` in ``UPT_rlt_pi05_recovery_dev``. The earlier ``research/rlt-pi05-four-suite`` worktree stays unchanged so its sealed run remains auditable. The matrix freezes all 40 tasks, training seeds 42/43/44, and four evaluation arms before execution. It retains the budgets and engineering gates above; Spatial, Object, Goal and Long use the pinned upstream horizons of 220, 280, 320 and 520 control ticks, respectively.

Every task/seed trains its own encoder and heads against the same frozen public VLA. Collection, development and online reset seeds differ between training seeds; the 100 held-out reset seeds per task are shared across seeds and arms. Three repetitions on those cases measure training-seed variation, not 300 independent physical initial states. The complete matrix contains 120 task/seed cells and 48,000 evaluation episodes. Final-budget checkpoints and a fixed recipe make it a matched adapted baseline, not a reproduction of the upstream few-demonstration paper setting.

Create a new output directory on a large filesystem. Preparation seals runtime code hashes, case lists and job plans; it does not load a policy or start GPU work:

.. code-block:: bash

   export RLT_PI05_MATRIX=/path/to/large-disk/four-suite-run
   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_matrix prepare \
     --root "$RLT_PI05_MATRIX" --assets "$RLT_PI05_ASSETS" \
     --source "$RLT_PI05_SOURCE" --gpus 0 1 --wandb-entity your-team

Launch one bounded worker per declared GPU, for example in two tmux sessions. Each worker waits for an existing research lease and then requires two idle GPU observations. Existing users' processes are neither stopped nor treated as available capacity. Work rotates through suites and tasks for seed 42 before seeds 43 and 44:

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_matrix run \
     --root "$RLT_PI05_MATRIX" --gpu 0 --hours 12
   # In the second terminal:
   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_matrix run \
     --root "$RLT_PI05_MATRIX" --gpu 1 --hours 12

The twelve-hour limit includes lease and GPU waits. It is an execution window, not a promise to finish 120 cells overnight. Each worker owns ``queue-gpu0`` or ``queue-gpu1`` and refuses to overwrite it; interrupted stages retain their artifacts and require explicit recovery before another run. The existing single-task launcher remains independent. Keep declared runtime code and plans unchanged until the workers exit.

Before model loading, each worker runs a 90-second bounded W&B authentication and CUDA arithmetic check under its GPU lease. The result is saved in ``runtime-gpuN/manifest.json``; it contains no policy evaluation. New matrix plans set ``stop_on_failure: true``: any failed or timed-out job, including a development gate, halts that worker as ``halted_failure`` and leaves later jobs ``not_started_halted_failure`` for review. Legacy plans retain dependency-only skipping unless they explicitly enable this option. An explicit error in an incomplete job manifest also triggers process-group cleanup while the child is still running, so logger shutdown retries cannot consume the remaining stage budget.

If authentication fails, check the proxy inherited by the actual tmux worker, not only the current shell. When direct connectivity is available and the proxy is obsolete, unset its uppercase and lowercase environment variables for that worker explicitly. Keep online logging required. If NVIDIA reports a driver/library mismatch after a system update, coordinate the host repair; do not reboot a shared training server automatically. A process-local library override must match the loaded kernel driver and pass CUDA and rendering checks before use. Record any override and preserve failed attempts, checkpoint hashes and case manifests when preparing a recovery; an infrastructure failure with zero episodes is not a zero-success result.

Inspect ``queue-gpu0/status.json`` and ``queue-gpu1/status.json`` for live scheduling. The following CPU-only command rechecks each cell's provenance and matched resets, including physical-state consistency across training seeds:

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_matrix status \
     --root "$RLT_PI05_MATRIX"

The report lists completed, failed and pending cells separately. A suite mean requires all ten tasks and all three seeds; the four-suite mean requires all four complete suites. It reports per-seed means and sample standard deviation, giving tasks and suites equal weight. Failed gates are not removed to inflate the mean. Earlier exploratory Goal pilots remain outside this new matrix. Neither a high training success rate nor completion of the queue certifies a complete matrix.

For a simulator-only check while GPUs are busy, ``preflight --root "$RLT_PI05_MATRIX"`` creates and restores one development reset for each of the 40 tasks using OSMesa. Configure the external dependency and OSMesa library paths first. Diagnostic snapshots live under ``reset-preflight``, independently of the GPU jobs; the check executes zero control ticks and no policy inference, and therefore produces no success-rate evidence.

Inspect the Results
-------------------

Read ``queue/status.json`` first to distinguish waiting, failed and completed jobs. Each run writes ``manifest.json`` with source, model revision, adapter hashes, package versions, seeds, device mapping and exit status. ``episodes.json`` records reset hashes, success and executed ticks. A passed warmup produces ``warmup/warmup.pt``; successful online execution saves ``q_bc/online.pt`` and ``bc_only/online.pt``.

For a compact report, run the following command. It keeps missing scores distinct from completed zero-success evaluations. When the queue exits, it also writes ``summary.json``:

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign status \
     --root "$RLT_PI05_RUN"

The live status recomputes ``acceptance_audit`` from the artifacts. For acceptance campaigns, ``engineering_pipeline_complete`` requires both completed evaluations and a passed audit. To save a separate audit snapshot without loading a model or reserving a GPU, choose a new report filename:

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign audit \
     --root "$RLT_PI05_RUN" --report "$RLT_PI05_RUN/audit-01.json"

The command checks the declared case files and their digests, task and training seed, recorded protocol/code/package versions, whole-episode warmup split, executed interaction budgets, checkpoint paths and update counters. It also rechecks the paired warmup gate, confirms that each physical reset belongs to one case ID, and rejects held-out evaluation started before both online arms finished. Logging must match the declared plan; completed online-logged stages must have a W&B run record. Exit codes distinguish passed (0), failed (2), incomplete (3), and a directory without an acceptance plan (4). An existing report is never overwritten. A missing result remains incomplete rather than becoming a zero success rate, and inconsistent paired results suppress the comparison instead of producing a score.

``stage_costs`` includes live online progress and recorded elapsed time; ``completed_training_cost_by_arm`` adds reference collection, encoder/BC warmup, and that arm's online training after those stages finish. Shared preparation is included once for each arm's from-scratch cost, so summing the arms would double-count it. The elapsed timestamps include model loading and checkpoint I/O but exclude imports, W&B finalization, queue waits, evaluation, upstream VLA pretraining and manual debugging. They are not measurements of GPU-active time or human effort. Existing manifests identify head checkpoints by path and update counters, not a hash of checkpoint bytes; the audit checks recorded consistency, not cryptographic weight provenance. Passing it establishes a one-task engineering comparison, not an RL improvement or full-suite reproduction.

For completed v3 stages, ``stage_costs`` also contains ``episode_outcomes``: successes, full-horizon failures and the final episode cut short by the online budget. A budget truncation is inferred only when the final unsuccessful episode is shorter than the horizon and the recorded tick totals exactly reach the declared online budget. An unexplained early unsuccessful episode fails the audit. The original success flags and rates remain unchanged; do not convert the truncated tail to a success or report a rate with it removed as independent evaluation performance.

On the development host, ``rlt_pi05_v3_gpu0_1009`` and ``rlt_pi05_v3_gpu1_1009`` manage the original v3 queues under ``research/rlt_pi05_20261009``: ``acceptance-v3-task0-seed42`` on GPU 0 and ``acceptance-v3-task1-seed42`` on GPU 1. Read their live status rather than inferring progress from a tmux session. They wait for existing compute processes to exit, not merely for utilization to reach zero. The four-suite matrix has separate queues and artifacts. Earlier v1/v2 acceptance and CPU/pilot outputs are preserved as compatibility-failure evidence, not accepted baseline scores. The old RLT_a/planner queues were paused between jobs to give this baseline priority; their completed artifacts were not removed. Failure manifests and skipped jobs remain available rather than being relabelled as completed training.

When no GPU is available, ``cpu-reference`` can check the actual simulator loop with CPU inference and an installed OSMesa software renderer. Set ``LD_LIBRARY_PATH`` if OSMesa is unpacked outside the system paths. This mode never reserves a CUDA device; the required ``--gpu`` argument is ignored. It is slow and uses CPU BF16 autocast, so its results do not replace matched GPU evaluations:

.. code-block:: bash

   bash run_rlt_pi05.sh cpu-reference --source "$RLT_PI05_SOURCE" \
     --assets "$RLT_PI05_ASSETS" --output /path/to/new-cpu-smoke \
     --gpu 2 --episodes 1 --state-start 49 --seed 45000

Prepare a RoboDojo Evaluation
--------------------------------

Once the LIBERO pilot passes, migrate the model's observation and action contract before adding more training. `RoboDojo <https://github.com/RoboDojo-Benchmark/RoboDojo>`_ currently provides an eval-only simulator/task release, while XPolicyLab owns policy servers. The inspected revisions are RoboDojo ``cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c`` and XPolicyLab ``53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2``.

The initial target is the ARX X5 joint-control interface: head and two wrist RGB cameras, 14 state coordinates, and absolute arm-joint radians plus gripper openings in [0, 1]. Gripper state in the inspected implementation represents commanded openings, not independent measured joint response. A seven-dimensional LIBERO end-effector checkpoint cannot be padded or relabelled into this interface.

The evaluation adapter now provides XPolicyLab's ``Model`` API and native single/batch deployment loops in ``toolkits/rlt/robodojo_policy.py``. The WS server decodes images before calling the adapter. Reset at an episode boundary, update observations with explicit environment IDs, then request actions for active IDs. The adapter filters privileged simulator fields, rejects stale observations and incompatible actions, and keeps chunks attached to their IDs when environments finish mid-chunk. The backend must be stateless; per-environment memory is not implemented here.

Start with ``evaluations/robodojo/arx_x5.yaml``. It selects ``stack_bowls``, native layout seed 0, 25 episodes and one environment. Use a dedicated RoboDojo checkout with its pinned XPolicyLab submodule. Fill in absolute runtime and checkpoint paths, and select ``openpi_reference`` for a native OpenPI ARX X5 checkpoint. This loader uses the exact checkpoint directory and its ``assets/<normalization_id>/norm_stats.json``, never an automatically selected latest checkpoint. A compatible RLT backend can be supplied through ``backend: factory`` and a trusted local ``backend_factory: module:function``; the function receives the policy configuration and returns an object whose ``infer(observation)["actions"]`` is a denormalized [H, 14] array. An ARX X5 RLT weight loader and compatible trained weights are still required for that route. The existing LIBERO RLT head is not supported.

The official dataset provides a native ARX X5 Pi0.5 reference at ``ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999``. It is not an RLT actor/critic checkpoint. The asset downloader pins ``RoboDojo-Benchmark/RoboDojo`` to revision ``35efbc7dedfdbeeb6e95fb749bd885d73d483e41`` and selects the first 25 numerically ordered ``stack_bowls`` layouts, X5 robot, bowl and camera-stand assets, scene materials, and reference parameters and normalization. It excludes training data and optimizer state. The inspected bundle contains 372 files (13.275 GB). Inventory before downloading, using a new manifest path:

.. code-block:: bash

   python -m toolkits.rlt.robodojo_assets inventory \
     --manifest /path/to/assets-manifest.json --max-gib 20
   python -m toolkits.rlt.robodojo_assets download \
     --manifest /path/to/assets-manifest.json --root /path/to/bundle \
     --workers 3 --max-gib 20

These commands need Requests but do not import CUDA. Downloads resume owned partial files, verify upstream SHA-256 or Git blob hashes, and publish files only after verification. ``robodojo-download-report.json`` records progress and errors; only ``complete: true`` means all listed files passed. An exclusive lock rejects concurrent writers to the same bundle. An interrupted process may leave the lock: confirm its recorded PID is no longer downloading this bundle before removing that single lock file. Do not treat a partially populated checkpoint directory as usable. The manifest describes the selected files, not a proof that every USD dependency resolves at runtime.

Link the completed bundle's ``Assets`` directory into a dedicated RoboDojo checkout only if its ``Assets`` path does not already exist, and set ``checkpoint_path`` to the exact checkpoint leaf inside the bundle. Set ``artifact_download_report`` to the bundle's progress report so preparation refuses incomplete or failed downloads. This is a completion gate, not a fresh hash audit. Asset files can reside on NAS, but Python environments and Git checkouts require a filesystem that supports their symlinks and permissions. RAM-backed staging is temporary: archive a completed environment or retain a reproducible installation recipe before restarting the host.

Use separate Python 3.11 environments for Isaac Sim 5.1.0 and the native OpenPI server. The inspected simulator pins WebSockets 12, whereas the server imports the newer asyncio server API. Pin IsaacLab to ``afca7b09d60d8beb9c1cb28b43066499940b969b`` and CuRobo to ``d17b54ce32cba095c0b000c4c58777075d11de0e``; do not advance submodules with ``--remote``. This reference uses JAX/Orbax parameters, so a PyTorch-only LIBERO installation is not sufficient. Package metadata can be inspected without importing the simulator:

.. code-block:: bash

   python -m toolkits.rlt.robodojo_eval runtime \
     --config /path/to/robodojo-eval.yaml --output /path/to/runtime-inventory.json

``packages_present`` only checks required package metadata and the Python/Isaac version gates. It does not certify dependency compatibility or GPU execution. The report retains ``runtime_verified: false``. Complete model loading, USD resolution, rendering, driver checks and one native reference episode before reporting an environment as verified.

Check actual policy imports in its isolated interpreter before reserving a GPU:

.. code-block:: bash

   python -m toolkits.rlt.robodojo_eval probe-reference \
     --config /path/to/robodojo-eval.yaml --output /path/to/reference-import.json

This bounded subprocess hides CUDA and selects the JAX CPU backend. It imports the native loader, verifies the ARX X5 Pi0.5 configuration and normalization, and checks that the execution horizon fits the model chunk. ``reference_imports_passed`` means imports and configuration passed, not that parameters were loaded or actions generated. The native model's padded width is 32, while its robot output is 14-dimensional. Model loading repeats the configuration checks so a LIBERO configuration cannot be substituted behind an ARX X5 label. Custom factories require their own preflight.

The inspected native loader also imports Pytest, Chex, Optax and PyArrow. An isolated import check exposed missing packages and PyArrow 26's incompatibility with this OpenPI environment's NumPy 1.26.4. The native lockfile versions Pytest 8.3.5, Chex 0.1.89, Optax 0.2.4 and PyArrow 24.0.0 passed that check. Record these dependencies in the policy environment's installation recipe; package presence alone would miss the NumPy ABI failure.

When installing with uv from an RLinf worktree, use ``uv --no-config pip install`` and explicit native version pins; project-level overrides can otherwise replace the requested PyTorch version. Keep required upstream dependency overrides in a separate recorded file. The inspected IsaacLab Starlette requirement conflicts with Isaac Sim 5.1's FastAPI dependency; RoboDojo's installer pins Starlette back to 0.45.3. Matching this recipe does not prove that every optional IsaacLab feature works. Do not repair these conflicts by upgrading the active LIBERO environment.

Preparation is CPU-only: it checks source revisions, inventories and hashes the fixed native layouts, checks asset directories and executable paths, and records commands without executing them. Install NumPy and PyYAML in the preparation environment; no Isaac, OpenPI or CUDA import is needed. After editing a copy of the YAML, run:

.. code-block:: bash

   python -m toolkits.rlt.robodojo_eval prepare \
     --config /path/to/robodojo-eval.yaml --output /path/to/new-plan.json

The command never overwrites an existing report. Exit 0 means ``prepared``; exit 3 means missing prerequisites. Even a prepared manifest has ``runtime_verified: false``: file checks do not prove that Isaac Sim or the model can run. Preparation and model loading reject normalization files that do not contain finite 14-dimensional state/action statistics with valid ranges. The supplied template intentionally leaves the backend unconfigured. Native per-task episode caps are enforced; asking for 50 when this task permits 25 is an error, not a silently shortened run. Layout IDs are positions in numerically sorted filenames, not the numeric filename suffixes.

When runtime, assets and compatible weights are available and GPU evaluation is explicitly scheduled, install the plugin in that dedicated checkout:

.. code-block:: bash

   python -m toolkits.rlt.robodojo_eval install-policy \
     --config /path/to/robodojo-eval.yaml

This only creates ``XPolicyLab/policy/RLT_Pi05`` and refuses an existing directory. It does not start a server, simulator or training. The plan's ``commands`` contains the working directory, environment and separate server/client argument lists. Apply the recorded environment to both processes, start the server first, then the native client. The server binds only to loopback. GPU masking maps the selected physical GPU to client device 0. Use a new run ID rather than silently resuming an earlier result. Prepare a separate checkout/plugin for a different frozen model configuration; the audit rejects plugin changes.

The recorded environment clears ``LD_LIBRARY_PATH`` for these child processes and disables JAX memory preallocation so the policy and simulator can share their assigned GPU. An inherited CUDA 12.1 path caused native JAX compilation to segfault on the inspected host; clearing it allowed GPU arithmetic to pass without changing system libraries. CUDA device 0 is the masked device, whereas Isaac's Vulkan renderer uses the physical index. The client therefore explicitly selects ``gpu`` for rendering, disables multi-GPU rendering, and keeps physics on CUDA device 0. Check the simulator's GPU table before treating placement as verified.

Choose an unused loopback port and verify that the newly started server owns it before connecting the client. Isaac may request acceptance of the NVIDIA Omniverse license at first launch; the operator must review and accept it before unattended execution. A successful synthetic policy inference establishes weight loading and output shape only. It is not a rendered episode or a task-success measurement.

After a real evaluation, audit the explicitly recorded result path:

.. code-block:: bash

   python -m toolkits.rlt.robodojo_eval audit \
     --plan /path/to/new-plan.json --result /path/from/plan/_result.json

The audit rechecks source, adapter, plugin, layout ordering and normalization consistency. It requires exactly the planned layouts and episode count, and separates full success rate from native partial score. Missing episodes remain incomplete; duplicate or substituted layouts, including replacements after physics failures, fail the paired comparison. Exit codes are passed (0), failed (2) and incomplete (3). Scores are withheld unless the comparison passes. This checks recorded consistency, not checkpoint byte provenance, runtime versions or whether an arbitrary custom backend used hidden assistance.

Runtime provisioning and artifact downloads are separate from simulator acceptance. An ARX X5 RLT checkpoint is not supplied by this integration. CPU tests and an optional native WS round-trip establish interface behavior, not simulator success rates. Run the contract tests without GPUs:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES= python -m pytest -q \
     tests/unit_tests/test_robodojo_boundary.py \
     tests/unit_tests/test_robodojo_policy.py \
     tests/unit_tests/test_robodojo_eval.py \
     tests/unit_tests/test_robodojo_assets.py

Set ``RP_XPOLICYLAB_SOURCE`` to the pinned XPolicyLab checkout and install its WS dependencies to include the optional loopback transport test. Next validate one native reference episode before a full fixed-layout evaluation. No RoboDojo training loop is added here; human/planner assistance and memory remain outside this baseline.

Set ``RP_ROBODOJO_POLICY_PYTHON`` to the isolated native OpenPI interpreter to include CPU import/configuration tests. These tests use synthetic normalization files and never restore checkpoint parameters. They check the native ARX configuration and reject mismatched LIBERO configurations, normalization IDs and execution horizons.
