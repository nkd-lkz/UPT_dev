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
