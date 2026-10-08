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

Audit the Same Failed States Without Training
------------------------------------------------------------

When a released checkpoint falls short of its published score, compare evaluation
entries before retraining. ``reference`` executes the frozen SFT VLA proposal;
``rlt_a`` executes the released actor conditioned on that proposal, compressed
VLA features and proprioception. They share the VLA but not the final action
generator. The actor directly predicts actions, not a guaranteed residual sum.

Select the released RLT_a failures from a completed paired evaluation, then run
the upstream Python main and this branch's wrapper on the same task/state pairs:

.. code-block:: bash

   bash run_rlt_libero.sh audit \
     --previous /path/to/completed/libero-release100 \
     --tasks 5 6 9 --gpu 2 \
     --output /path/to/new/libero-entry-audit --check
   bash run_rlt_libero.sh audit \
     --previous /path/to/completed/libero-release100 \
     --tasks 5 6 9 --gpu 2 \
     --output /path/to/new/libero-entry-audit

The first command validates failure selection without allocating a GPU. The
second refuses a busy GPU 2, hashes the four weight files before and after the
run, and executes both entries sequentially. The wrapper also repeats the
reference arm. All episodes save videos and JSONL traces of input-image hashes,
proprioception, normalized reference/actor chunks, executed environment actions,
rewards and termination flags. Initial camera images are saved as PNG files.
``summary.json`` reports the first differing record for each pair, not just a
success-rate difference. A complete manifest requires both entries to finish
and the checkpoint hashes to remain unchanged.

The upstream Python main keeps its model loading and inference logic. Explicit
failure-ID filtering, passive trace hooks, one evaluation thread and disabling
machine-specific ``.env`` loading are recorded adaptations. Both entries share
the unchanged upstream rollout helper and the existing IPC/EGL compatibility
wrapper. Thus matching traces exclude an entry-specific difference on these
cases; they do not validate the simulator against the author's internal setup
or reproduce the published 92%. These are failure-selected development episodes,
not an unbiased benchmark or a dataset for additional training.

Check the Shared Simulator and Release Files
------------------------------------------------------------

Matching evaluation entries can still share an incompatible simulator. Before
changing model weights, check installed requirements, the complete released
configuration and tokenizer, and the benchmark assets. The CPU-only inventory
compares SHA-256 for Hub LFS files and Git blob hashes for ordinary files:

.. code-block:: bash

   bash run_rlt_libero.sh environment-audit inventory \
     --libero-source /path/to/official/LIBERO-git \
     --output /path/to/new/environment-inventory

``--libero-source`` is optional and refers to a local upstream Git repository;
the inventory compares its HEAD tree against the active LIBERO paths without
checking out files or changing the installation. ``summary.json`` contains file
hash comparisons. ``manifest.json`` records package versions and unsatisfied
active LIBERO requirements. Completion means the inspection finished, not that
every requirement or file matched. Internet access is needed for pinned Hub
metadata; no weights are downloaded.

To distinguish model differences from simulator differences, replay an existing
audit's executed actions without loading a VLA:

.. code-block:: bash

   bash run_rlt_libero.sh environment-audit replay \
     --trace /path/to/audit/official/traces/rlt_a/task_9_state_7/trace.jsonl \
     --snapshot-ticks 212 213 290 \
     --output /path/to/new/action-replay

Replay refuses a busy GPU 2, selects its EGL device by UUID, and preserves the
recorded LIBERO Goal initial state, seed and action sequence, including settling
actions. ``physics.jsonl`` records qpos, qvel, control, warm-start accelerations,
contacts, time and both camera hashes. ``model.json`` records collision masks
and geometry names; requested snapshots save camera pixels. This is an
open-loop diagnostic, not an autonomous success-rate measurement. Run separate
processes to compare simulator versions or repeated renders. Keep dependency
overlays isolated from running training environments.

On 2026-10-07, all 20 downloaded release files matched their pinned Hub revision.
The installed LIBERO tree matched upstream ``8f1084e`` for 585 assets, 135 BDDL
files and 250 initial-state files. Its three source differences concern path
configuration, downloading and explicit ``torch.load(weights_only=False)``;
the environment wrapper and task dynamics files matched. However,
``rlinf-libero==0.1.3`` requires ``mujoco>=3.0,<3.4``, while the shared environment
had 3.8.1. A separate 3.3.7 overlay changed RLT_a success from 3/9 to 3/9 and
reference success from 4/9 to 3/9 on tasks 5, 6, 9 and states 0, 1, 2.
The dependency violation is real; this pilot does not establish it as the cause
of the published-score gap.

Two process-isolated replays of task 9/state 7 under the original environment
matched all 331 recorded physics states and all primary-camera images. Their
wrist-camera hashes differed at 15 steps, locating one source of trace
non-repeatability in rendering/observation rather than differing physics states.
The exact pixel-level cause and its contribution to aggregate success remain
unverified.

The original LIBERO requirements also pin robosuite 1.4.0, whereas the shared
environment has 1.4.1, whose Panda XML adds a ``link7_collision`` geometry.
Changing only robosuite while keeping MuJoCo 3.8.1 preserved reference success
at 4/9 and RLT_a success at 3/9 on the same nine initial states. Open-loop replay
of task 9/state 7 also preserved the full recorded physics state, with no
contacts involving the added geometry. Neither single-version change has
explained the low success rate. Effects on other initial states, or interactions
from changing both versions together, remain untested.

Compare per-task outcomes as well as aggregate sample counts. If the author's
50 episodes are the same states 0–49 under the same protocol, our six task-9
failures among states 0–9 limit that task to 44/50 even if every remaining state
succeeds, below the reported 46/50. More episodes alone cannot reconcile those
particular outcomes. The release does not supply a full internal environment
lockfile or per-initial-state traces, so its environment remains unverified.

Test Image Preprocessing Separately
----------------------------------------

The public source has another shared discrepancy: training ``_pack_sample``
resizes each view to 224 by 224, as does the ordinary VLA benchmark client,
whereas the RL evaluator passes 256-by-256 images directly. The release config
omits ``datasets.vla_data.image_size``, so the model's optional resize does not
run. A CPU check of the released processor produces 81 visual tokens per view
through the original path, versus 64 from 224-pixel input. This establishes an
input-distribution difference between public code paths, not the complete
internal training configuration of the released weights or its causal effect
on success.

The optional diagnostic below applies the ordinary VLA benchmark client's
OpenCV area resize. Weights, environment, initial states and action processing
stay fixed. Default evaluation behavior remains unchanged:

.. code-block:: bash

   bash run_rlt_libero.sh evaluate --gpu 2 \
     --tasks 5 6 9 --states 0 1 2 --input-image-size 224 --video \
     --output /path/to/new/libero-images224-pilot

The manifest explicitly records ``input_image_size`` and
``preprocessing_diagnostic``. Even if this single-variable experiment improves
success, it is not by itself a strict reproduction of the published 92%.
The diagnostic is not exposed through the training entry.

The nine-state 224-input diagnostic completed at 20:06 CST on 2026-10-07.
Reference success changed from 4/9 to 6/9 and RLT_a from 3/9 to 5/9 with frozen
weights. These inspected development cases motivate further preprocessing
checks; they neither explain the entire score gap nor establish an RLT_a
advantage over reference.

The continuation plan is ``experiments/libero_rlt/overnight_audit_20261007.json``:
the original 100 task/state pairs, followed by a 224-input diagnostic over all
50 states of each of the ten tasks, without selecting tasks by pilot score.
The old six-hour queue was stopped while waiting, preserving its completed
artifacts. The new twelve-hour budget includes waiting, uses GPU 2 only and
requires two consecutive idle observations. Its tmux session is
``rlt_libero_overnight_1007``; live state is in
``research/libero_environment_audit_20261007/overnight_queue/status.json`` on NAS.
Occupied GPUs are never preempted. A deadline records unstarted jobs rather than
claiming completion.

After the 100-case run finishes, compare original and resized outcomes on CPU:

.. code-block:: bash

   bash run_rlt_libero.sh environment-audit compare-preprocessing \
     --baseline /path/to/completed/libero-release100 \
     --candidate /path/to/completed/images224-matched100 \
     --output /path/to/new/preprocessing-comparison

The report separates the nine inspected pilot cases from the remaining 91 and
lists both newly successful and newly failed states. It rejects incomplete or
assisted evaluations, updated learner weights, unmatched initial states and
different dependency versions. The remaining cases are still public benchmark
states, not unseen test data. These summaries establish no statistical
significance; even the 500-case diagnostic must disclose altered preprocessing.
Completed version pilots, inventory and replays remain indexed by
``experiments/libero_rlt/environment_audit_results_20261007.json``. Use fresh
output paths before reusing a dated plan.

Accept the Learning Loop Before Scaling
------------------------------------------------------------

Use the separate acceptance campaign to establish a new task-0 reference and
test whether a fresh actor preserves it before adding the critic objective.
The 84.2% reference result from the 2026-10-07 full-suite diagnostic remains an
old-protocol result. Do not compare its absolute score to this task-0 experiment.

Protocol ``libero-rlt-a-acceptance-v1`` fixes MuJoCo 3.3.7, robosuite 1.4.1,
NumPy 1.26.4, Transformers 4.53.2 and tokenizers 0.21.4. Both training and
evaluation resize both 256-pixel views to 224 pixels using OpenCV area resize,
then use the same frozen VLA/encoder and official action conversion. The local
acceptance venv overrides MuJoCo and Transformers while reading the existing GPU
stack from the shared venv; it does not modify that shared installation. This
is dependency isolation for this route, not a clean installation for the unused
OpenPI, LeRobot or dm-control packages. During setup, the shared Transformers
metadata reported 4.57.6 while its actual module reported 4.53.2; the acceptance
venv installs a coherent 4.53.2 copy and records both module and package versions.

On the prepared supermicro host, start the finite campaign in tmux:

.. code-block:: bash

   cd "$HOME/rlinf_rlt/UPT_libero_dev"
   bash run_rlt_libero_acceptance.sh \
     --gpus 0 1 --control-budget 32000 --bc-updates 5000 \
     --max-hours 12 --wandb online --output /path/to/new/acceptance

The command waits for two consecutive idle checks on each assigned GPU, with a
12-hour total deadline including waiting. It never stops another user's task.
``status.json`` distinguishes waiting, running, execution failure and a stopped
BC gate. W&B receives metrics when reachable; JSONL remains the primary local
record if its bounded connection attempt fails. Store outputs on a large result
volume. Only small heads, optimizer states and replay are checkpointed, every
16,000 simulator ticks and at completion; the frozen VLA is not duplicated.

The campaign first collects reference trajectories on published task-0 states
0–29. States 0–23 train BC; 24–29 measure held-out following error; 30–49 are the
development rollout gate. All are previously inspected public states. A separate
bank of 50 random resets, with seeds 20000–20049, stores XML and physical states
and checks repeat restoration and duplicate states. Only the simulator sees this
state bank: the actor still receives two images and proprioception. Validation
outcomes are read only after development decisions and are never inserted into
replay. These are new initial configurations of the same task, not new tasks or
proof that the VLA never saw similar configurations during pretraining.

BC uses 5,000 actor updates, reference dropout 0.5 and the released head's fixed
standard deviation 0.1. Its development gate requires held-out MSE below both
0.02 and half its initial value, at most 5% gripper-threshold disagreements,
reference success of at least 50%, and no more than two lost successes relative
to reference on 20 paired episodes. This is an engineering stop condition, not
a statistical noninferiority claim. A failure stops online RL instead of spending
the remaining GPU budget on an actor that has not passed basic acceptance.

After acceptance, GPU 0 runs BC-only and GPU 1 runs Q+BC from the identical
warmup checkpoint, optimizer states and replay. Both train a diagnostic critic
with the same schedule; only Q+BC sends its Q term into the actor gradient. Each
arm gets up to 32,000 simulator ticks including settling; at most ten unused
ticks can remain rather than starting an episode with no controllable step.
Actual actor/critic counts and the behavior version for every executed chunk
are saved. The synchronous learner is an explicit RLT_a training adapter: it
discounts by executed ticks, bootstraps time limits, excludes unfinished
nonterminal chunk prefixes from TD, and publishes every actor update. It is not
the unchanged AlphaBrain training script or full-token RLT.

Seed 42 is the development pilot. Seeds 43 and 44 run only if Q+BC exceeds
BC-only and reaches reference on the development gate; each new seed must also
pass its own BC gate. Results on the 50 generated validation states are reported
separately, with matched successes and losses. Conditional replication and a
single-task result must remain explicit when interpreting the outcome. Memory,
planner and human assistance are disabled throughout this experiment.

Reproduction Boundaries
-----------------------

This branch does not register an external model type in RLinf or convert ManiSkill weights into LIBERO weights. It provides an auditable public-model evaluation adapter and a bounded training adapter. Local scores come only from successfully completed local result files; upstream benchmark claims are not local results. Full-token RLT needs a separately matched Stage 1 encoder, not RLT_a weights.

Official RLinf main remained ``c70606f08cdca259b8dec03d4430926b5b8fac9d`` when rechecked on 2026-10-07. Open `PR #1623 <https://github.com/RLinf/RLinf/pull/1623>`_ concerns retained replay checkpoints; `PR #1527 <https://github.com/RLinf/RLinf/pull/1527>`_ concerns RLT phase routing. Their relevance is recorded here; neither was automatically merged into the local baseline. The AlphaBrain source and both public model revisions were also unchanged.
