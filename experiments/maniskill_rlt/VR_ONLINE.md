# Single-Environment VR Online Acceptance

Connect the working Windows simulator and PICO controls to an online learner on physical GPU 2. Start the server, establish an SSH tunnel, then collect real intervention samples. This is a single-environment, single-step engineering smoke, not the 64-environment training configuration or a converged policy.

## Define the Training Scope

The controller operates one Windows simulation. After each executed step, the client journals the raw transition and uploads it asynchronously. The server extracts both endpoint features with frozen Stage 1 weights, updates a small actor/twin critic, and publishes complete actor snapshots for subsequent predictions.

```text
Windows: observation → server prediction → P selects policy action
                  ↘ grip takeover → local IK → executed action
                                                   ↓
                            step, display, journal true next state
                                                   ↓ bounded background upload
GPU 2: frozen Stage 1 → feature replay + human replay
                               ↓ TD + Q/BC updates
                        actor / twin critic → publish → next prediction
```

The first version uses `horizon=1`; the frozen VLA still produces a ten-step reference, whose first action becomes the reference BC target. Unexecuted chunk tails therefore cannot enter training targets. This head is incompatible with the formal ten-step baseline head. Automatic phase routing and the automatic OpenPI expert are not part of this entrypoint: the user presses `P` for policy control and holds grip for human control.

`online_smoke.yaml` starts updates after eight records, uses batch eight, caps optimization at 128 updates, publishes every four updates, and checkpoints every sixteen accepted records. When human records exist, half the batch comes from human replay and half from all-action replay, so the realized human fraction can exceed one half. Human BC targets are executed intervention actions; other BC targets are frozen VLA references. Critics use raw sparse task reward; pressing grip never creates a success reward. Both termination and truncation block bootstrap in this explicit configuration.

Uploads and local control are asynchronous; server inference and learning are serialized. This is not a promise of ten remote predictions or updates per second. Full queues, tracking faults, expired replies and network errors pause control. Increasing the queue cannot fix sustained throughput deficits. Collection and inference can continue after the optimization cap.

## Start the Isolated Server

For longer operator acceptance, use this entrypoint. It defaults to Stage 1 step 2000 and `online_pilot.yaml`: 64 transitions before learning, batch 32, at most 5000 updates, publication every eight updates and checkpointing every 50 accepted transitions. Deployment requires at least 128 published updates and publication-minibatch imitation MSE at most 0.01; otherwise reference actions continue. This is a training diagnostic, not held-out evaluation or a safety certificate.

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
bash run_rlt_vr_hil_pilot.sh check
tmux new-session -s rlt_vr_hil 'bash run_rlt_vr_hil_pilot.sh run; exec bash'
```

Enter a secret of at least 32 characters in tmux and use the same secret on Windows. The server holds the shared project GPU-2 lease and refuses an occupied GPU. `check` validates paths/settings without allocating CUDA. GPU 0/1 baseline jobs remain unchanged. This remains one Windows environment with `horizon=1`, incompatible with the production ten-step Stage 2 head and not integrated into 64-environment training. The shorter 128-update smoke command follows.

CUDA is selected by physical GPU 2 UUID and verified before model allocation. The launcher rejects GPU 2 when over 1 GiB is occupied. Real simulator smoke also pins Vulkan by PCI address. It neither joins Stage 1's Ray cluster nor runs `ray stop`. In a fresh tmux session, run:

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
export RLT_STAGE1_ACTOR=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill/runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016/checkpoints/global_step_2000/actor
read -rsp 'Enter a random connection secret of at least 32 characters: ' RLT_VR_TOKEN
export RLT_VR_TOKEN
bash run_rlt_vr_online_gpu2.sh run
```

Use the same newly chosen secret on Windows; never commit it or send it in chat. Wait for `Online learner ready`. NAS `runs/vr_online/TIMESTAMP_PID/` contains `config.json`, `metrics.jsonl` and `learner.pt`. This smoke logs JSONL metrics rather than creating a W&B run.

## Connect Windows and Intervene

Keep PICO Business Streaming and SteamVR tracking active. In one Windows PowerShell, open the tunnel, replacing `SERVER_ADDRESS` with the address used for SSH:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 8775:127.0.0.1:8775 luokz@SERVER_ADDRESS
```

Leave that window open. In a second PowerShell, update the VR branch, enter the same secret and launch the client:

```powershell
Set-Location "C:\Users\lkz\Desktop\code\UPT_vr_dev"
git pull --ff-only origin feature/rlt-pico-vr-intervention
$env:RLT_VR_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Connection secret' -AsSecureString)).Password
$vrRecord = "C:\Users\lkz\Desktop\rlt-records\online-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
conda run --no-capture-output --name rlt-vr python -X faulthandler -u -m toolkits.rlt_vr.client --online --port 8775 --render-backend cpu --record "$vrRecord" --max-episode-steps 1000 --log-interval 1
```

Check images and tracking while paused. Press `P` to request policy actions, then hold grip and move slowly to take control. A fresh trigger press toggles the gripper. Releasing grip pauses; `P` explicitly returns to policy control. `R` resets, Space pauses and `Q` exits. The 1000-step manual acceptance horizon must not be compared directly with the production baseline evaluation configuration.

The terminal's `Online learner` line shows `ack`, pending uploads and server metrics. Acceptance requires increases in `human_accepted`, `update_step` and `policy_version`, not just visible robot following. Retain local `.npz` records: server replay stores features, while local records contain the original images.

## Interrupt and Resume

For the longer pilot, restore with the same pilot configuration and inspect accepted execution metrics:

```bash
bash run_rlt_vr_hil_pilot.sh run --resume /absolute/path/previous-run/learner.pt
/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m toolkits.rlt_vr.summarize_online /absolute/path/run
```

The report counts takeover steps, contiguous takeover segments, completed episodes and actual learner updates. Human flags describe the client-declared action route; scripted clients can set them too. `service_ms` measures server feature/update processing, not end-to-end control latency. `update_budget_exhausted=true` means optimization stopped; collection and inference continue.

Stop the client before pressing Ctrl-C on the server. Interrupting an in-flight update can fault the learner; a fault never overwrites the previous good checkpoint. Normal shutdown saves networks, target, both optimizers, replay, published version and RNG. Use the following only for a short run originally started with `online_smoke.yaml`; pilot checkpoints require `run_rlt_vr_hil_pilot.sh` above, with the same configuration:

```bash
bash run_rlt_vr_online_gpu2.sh run --resume /absolute/path/previous-run/learner.pt
```

Restart Windows with a fresh recording directory and episode. Active simulation state and unacknowledged queues are not restored; do not resend old-session packets. Configuration and frozen Stage 1/normalization identity must match. Samples after the last checkpoint may require recovery from local records; automatic journal re-ingestion is not implemented.

## Evidence and Remaining Acceptance

Server-side real simulation verified the same RPC, frozen Stage 1 extraction, scripted intervention samples, TD/BC updates, actor publication, deduplication and optimizer/replay restoration. See [VR_VERIFICATION.md](VR_VERIFICATION.md). Actual Windows/PICO-to-learner acceptance requires an operator and remains pending. Scripted interventions are not human success-rate evidence.

After this smoke, extend partial ten-step chunks with masks and correct discounts, integrate formal RLinf workers/replay and phase routing, then consider dual-GPU 64-environment training. New entrypoints remain isolated from baseline and are not merged.
