Verify a Published RLT Variant on LIBERO
========================================

This guide separates published-checkpoint evaluation, training integration and paper reproduction on an independent LIBERO research branch. Compare the frozen VLA and released learner on identical initial states before retraining. The existing ManiSkill baseline, planner and VR branches remain separate.

Why Start With LIBERO
------------------------------------------------------------

A fast, auditable baseline needs code, weights, action semantics and an evaluation protocol together. As of 2026-10-06, this branch selects AlphaBrain's public QwenOFT + ``RLT_a`` release for its first check. Changing simulators alone does not resolve learning failures.

The related works use different meanings of an RLT baseline:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Work
     - Actual comparison
     - Reproduction implication
   * - `eRLT <https://arxiv.org/html/2610.00913v1>`_
     - RLT* in LIBERO / RoboTwin shares a DSRL latent-noise RL framework rather than directly reproducing RLinf action-space training.
     - Useful representation controls, but not an acceptance target for this ManiSkill configuration. A separate official simulation-code entry remains unverified.
   * - `RouteRLT <https://arxiv.org/html/2609.26467v1>`_
     - Frozen SmolVLA, stage specialists and routing on three LIBERO Object tasks.
     - Oracle-routed RLT reports 91.11±0.96%; RouteRLT reports 92.22±2.55%. These are paper results, not local measurements or evidence that VR is required.
   * - `AlphaBrain <https://github.com/AlphaBrainGroup/AlphaBrain>`_
     - Full-token RLT and action-query RLT_a implementations. The verified public weights use QwenOFT / RLT_a.
     - Public models and evaluation code support a direct release check; its score is not a reproduction score for RLinf pi0.5 RLT.

RLT_a compresses action-query features to 256 dimensions. Its released actor directly outputs 8×7 actions, rather than structurally guaranteeing a zero residual over the reference. The RLinf ManiSkill backbone, phase gate, rewards and controller differ. Diagnose ManiSkill regressions through matched states, actual actor-control counts, normalization and BC acceptance, not by assuming missing VR is the cause.

Prepare Independent Source and Models
------------------------------------------------------------

The local directory is ``UPT_libero_dev`` on branch ``research/rlt-libero-reproduction``. The adapter pins external source and two Hugging Face revisions. It does not run upstream multi-GPU shell defaults or user-wide process-cleanup commands.

Place the external checkout on a local filesystem supporting ordinary permissions, not the current CIFS mount:

.. code-block:: bash

   export ALPHABRAIN_SOURCE="$HOME/rlinf_rlt/third_party/AlphaBrain"
   git clone https://github.com/AlphaBrainGroup/AlphaBrain.git "$ALPHABRAIN_SOURCE"
   git -C "$ALPHABRAIN_SOURCE" checkout --detach 604924beb77b04b0da49326dfae6ea423a27d28a

Skip cloning an existing directory and inspect its revision without overwriting local edits. Runtime dependencies include Torch, LIBERO and its assets, MuJoCo/EGL, Transformers, Accelerate, FlashAttention, qwen-vl-utils and upstream QwenOFT dependencies. CPU tests or ``check`` do not certify the runtime. Do not upgrade a shared environment underneath a running baseline.

On this host, extra packages live in ``$HOME/rlinf_rlt/third_party/alphabrain-deps`` and the EGL loader in ``$HOME/rlinf_rlt/third_party/alphabrain-egl``, without modifying the shared venv.

On the prepared supermicro host, ``bash run_rlt_libero.sh check`` uses the sibling
``UPT_dev/.venv``, isolated dependency directories, and downloaded NAS assets.
It installs nothing and starts no training. Pass ``evaluate`` or ``train`` with
the arguments below and a new output directory to execute an experiment.
On another host, set ``RLINF_VENV``, ``RLT_EXTERNAL_ROOT``,
``RLT_ALPHABRAIN_SOURCE``, and ``RLT_LIBERO_ASSETS`` as needed. The equivalent
environment setup without this wrapper is:

.. code-block:: bash

   export PYTHONPATH="$HOME/rlinf_rlt/third_party/alphabrain-deps${PYTHONPATH:+:$PYTHONPATH}"
   export LD_LIBRARY_PATH="$HOME/rlinf_rlt/third_party/alphabrain-egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/rlinf_rlt/third_party/alphabrain-egl/nvidia.json"
   export TMPDIR=/dev/shm
   export PYTHONDONTWRITEBYTECODE=1

Other machines should use their own system EGL rather than copy nonexistent local paths. The current shared Python environment passed a LIBERO Goal task-0 reset and ten rendered ticks on 2026-10-06. This does not certify model evaluation.

Store models on a filesystem with sufficient capacity. Downloads use immutable revisions, not latest weights, and do not need the ManiSkill dataset:

.. code-block:: bash

   export LIBERO_RLT_STORAGE=/path/to/large-disk/libero-rlt/assets
   python -m toolkits.rlt.libero_reproduction download \
     --source "$ALPHABRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE"
   python -m toolkits.rlt.libero_reproduction check \
     --source "$ALPHABRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE"

The VLA release is ``AlphaBrainGroup/qwenoft-5traj-libero-goal`` at ``91ea0817f556bf50d64008187519de19458f8188``. The learner is ``AlphaBrainGroup/alphabrain-rlt-5traj-alltasks-libero-goal`` at ``b5c0080e46f4b900c049e76bd9f3df68befac12a``, using ``rl_offpolicy_iter_00400``. ``check`` verifies source and required files without loading models or allocating GPU memory.

Evaluate Matched Pairs First
------------------------------------------------------------

Start with two initial states of one task on an idle GPU to test loading, interaction and video capture. Use a new output directory:

.. code-block:: bash

   python -m toolkits.rlt.libero_reproduction evaluate \
     --source "$ALPHABRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE" \
     --gpu 2 --tasks 0 --states 0 1 --video \
     --output /path/to/new/libero-release-smoke

The entry exposes only the selected GPU and refuses it above 512 MiB usage. CUDA selects the physical UUID; EGL resolves that same UUID to its own rendering index instead of assuming both enumerations agree. Environment workers need CPU physics and EGL only, so their CUDA devices are hidden to avoid robosuite mixing the two index spaces. Missing or ambiguous UUID mappings fail closed.

Both arms run serially with one frozen VLA, the author's normalization, gripper handling, 8-tick chunks, 10 settling ticks and task horizon. The reference arm preserves VLA proposal values, widening to NumPy-compatible float32; the learner arm loads released encoder/actor weights. Neither uses planner or human assistance. The author's video recorder excludes the initial settling period. An adapter also redirects environment text logs to stderr so they cannot corrupt the binary stdout protocol; it does not alter task dynamics or rewards.

``reference.json`` and ``rlt_a.json`` retain task/state identities. ``summary.json`` counts reference-only and learner-only successes as well as aggregate rates. ``manifest.json`` becomes complete only after the whole run succeeds. Task IDs are 0..9 and state IDs are 0..49; duplicate or out-of-range IDs are rejected rather than wrapped.

Two episodes validate integration only. Then expand with ``--tasks 0 1 2 3 4 5 6 7 8 9`` and an explicit ``--states`` list. The full 10×50 protocol re-evaluates published benchmark states; do not claim they were unseen during public-model training or generalize a selected task's score to the suite.

The second local smoke completed on 2026-10-06: LIBERO Goal task 0, initial states 0 and 1, seed 42. Both reference and public RLT_a succeeded in 2/2 episodes; four MP4 files were saved. Results are under NAS research/libero_reproduction_20261006/release_smoke_v2, indexed by experiments/libero_rlt/validation_20261006.json. This validates model/action/environment integration, not an RL gain or the author's full 10×50 score. The first attempt was stopped after EGL numbering selected a different physical GPU; UUID matching corrected the mapping. The interrupted run is not counted as passing.

Run a Controlled Training Pilot
------------------------------------------------------------

The completed 10-task, two-state comparison on 2026-10-06 measured reference 13/20 and public RLT_a 11/20: 11 shared successes, two reference-only successes and seven shared failures. It does not demonstrate an RL gain. Training below diagnoses the integration and learning path, not an established performance advantage.

Evaluation retains the original upstream checkout. Training needs a separate, audited runtime patch: UUID-to-EGL mapping for persistent workers, float32 conversion before NumPy, propagation of worker failures, and per-iteration metrics. Create the training worktree once and apply the included patch from the UPT_libero_dev directory:

.. code-block:: bash

   export RLT_ALPHABRAIN_TRAIN_SOURCE="$HOME/rlinf_rlt/third_party/AlphaBrain_rlt_runtime"
   git -C "$ALPHABRAIN_SOURCE" worktree add --detach \
     "$RLT_ALPHABRAIN_TRAIN_SOURCE" 604924beb77b04b0da49326dfae6ea423a27d28a
   git -C "$RLT_ALPHABRAIN_TRAIN_SOURCE" am \
     "$PWD/experiments/libero_rlt/patches/0001-fix-libero-propagate-rollout-faults-and-respect-expl.patch" \
     "$PWD/experiments/libero_rlt/patches/0002-fix-libero-bound-rollout-lifetime-and-publish-small.patch"

Skip these commands if the prepared training worktree already contains both patches. If it contains patch 1 only, apply patch 2 alone. The adapter requires its clean Git tree to equal ``7abd269c822569642f94f79948e4b0869e6d25d5``; commit timestamps may differ after ``git am``. It records the actual commit. The wrapper selects this tree only for ``train``; set ``RLT_ALPHABRAIN_TRAIN_SOURCE`` to override its location.

The training entry uses the same frozen released encoder but initializes a fresh actor and critic. It neither resumes the public step-400 optimizer nor trains full-token RLT. Start with 20 outer iterations:

.. code-block:: bash

   python -m toolkits.rlt.libero_reproduction train \
     --source "$RLT_ALPHABRAIN_TRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE" \
     --gpu 2 --tasks 0 --iterations 20 \
     --output /path/to/new/libero-train-pilot

The budget is one GPU, two environments, four episodes per iteration, five VLA warmup iterations, batch 128, at most 128 TD updates per iteration and replay capacity 20,000. W&B and concurrent evaluation are disabled; checkpointing occurs at the final iteration. These small budgets are not the author's complete recipe or a convergence guarantee. On return, the adapter rejects missing iterations, zero-step collection, nonfinite scalar metrics, missing actor updates after warmup, or absent final weights. ``metrics.json`` is progress evidence; only a complete manifest certifies the requested run finished.

After training, use the same paired evaluator with ``--learner-dir /path/to/checkpoints/rl_offpolicy_iter_00020``. Exit code zero is not evidence of improved autonomous success. Arguments, source revisions and experiment metadata accompany outputs. Training-level reproduction additionally requires larger samples, matched budgets and repeated training seeds.

The 128-update limit is a cap, not the number of gradient steps actually taken. Updates are ``new_transitions × utd_ratio / batch_size`` rounded down, bounded below by one and above by the cap. Upstream restarted the actor-delay counter each outer iteration: a one-update iteration with delay two made no actor update despite logging an ``actor_loss`` placeholder. Patch 2 uses a global delay counter and records actual optimizer steps. The adapter reconstructs counts only for old patch-1 histories and labels their provenance. Do not equate 100 outer iterations with 12,800 critic updates or interpret a loss key alone as an actor update.

The first 20-iteration run completed with patch 1 and saved its weights, but exposed two further limitations. The default 500-update publication interval exceeds this pilot's total updates, leaving rollout weights stale after warmup. Shutdown also closed workers while an unnecessary 21st batch was still collecting. Patch 2 collects exactly one batch per requested consumer iteration and checks thread completion before closing sockets. The adapter sets ``RLT_LIBERO_SYNC_UPDATES=1`` and publication holds the same lock as a complete collection pass. This is an explicit correction to the small-pilot runtime, not an untouched upstream run. The queued 100-iteration run uses patch 2; differences from the original 20-iteration run cannot be attributed solely to training duration. Its GPU acceptance remains pending until that run's manifest and counters are available.

Run a Bounded Overnight Queue
-----------------------------

The reviewed host-specific plan ``experiments/libero_rlt/overnight_20261006.json`` serializes eight jobs: a 20-iteration training acceptance test and paired evaluation, two 60-step planner arms and their report, a 100-pair public-release evaluation, then a 100-iteration fresh training run and its evaluation. Long training depends on the short run completing, not on a promised gain. It starts from the same initialization rather than pretending to resume optimizer state.

The queue supports only physical GPU 2. GPUs 0/1 remain reserved for the existing baseline, even if they appear idle while saving. It requires two idle checks, enforces per-job timeouts and a total 12-hour wall limit including waiting, skips dependent jobs after failures, and signals only process groups it creates. It cannot reserve the device against noncooperating external programs. Every launcher rechecks before starting; no job is guaranteed to start if GPU 2 stays busy. Edit copied output paths to new directories before reusing this dated plan.

.. code-block:: bash

   python -m toolkits.rlt.experiment_queue \
     --plan experiments/libero_rlt/overnight_20261006.json \
     --output /path/to/new/queue --check
   python -u -m toolkits.rlt.experiment_queue \
     --plan experiments/libero_rlt/overnight_20261006.json \
     --output /path/to/new/queue

The first command allocates no GPU. The second writes ``plan.json``, atomic ``status.json`` and per-job logs. A passing command needs both exit status zero and a completed experiment manifest when one is declared. Checkpointing is final-only; video is enabled only for the final task-0 comparison. Waiting, timeout and skipped jobs are not reported as completed experiments.

Reproduction Boundaries
-----------------------

This branch does not register an external model type in RLinf or convert ManiSkill weights into LIBERO weights. It provides an auditable public-model evaluation adapter and a bounded training adapter. Local scores come only from successfully completed local result files; upstream benchmark claims are not local results. Full-token RLT needs a separately matched Stage 1 encoder, not RLT_a weights.

Official RLinf main was ``c70606f08cdca259b8dec03d4430926b5b8fac9d`` when checked on 2026-10-06. Open `PR #1623 <https://github.com/RLinf/RLinf/pull/1623>`_ concerns retained replay checkpoints; `PR #1527 <https://github.com/RLinf/RLinf/pull/1527>`_ concerns RLT phase routing. Their relevance is recorded here; neither was automatically merged into the local baseline.
