# Single-Environment VR Online Acceptance

Verify the loop between Windows simulation, PICO corrections and the GPU 2 learner. Update both endpoints, check local control and ingestion, then confirm that a published actor actually executes and remains interruptible. This is an independent `horizon=1` engineering pilot, **not the production 64-environment, ten-step Stage 2 baseline**.

## Update Both Endpoints and Start Fresh

This revision uses `online_protocol=2`, with different receipt and quality-label semantics. Do not mix old and new endpoints. Updating source does not update a running process: quit the Windows client, press Ctrl-C in the original VR server terminal, then start the new server. Leave GPU 0/1 baseline and unrelated GPU 2 jobs running.

Checkpoints use schema 2. Schema 1 treated every takeover as BC data and cannot be resumed directly. Start first acceptance without `--resume`, retaining old logs and weights. Check settings, then launch a fresh tmux session:

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
bash run_rlt_vr_hil_pilot.sh check
tmux new-session -s rlt_vr_hil_v2 'bash run_rlt_vr_hil_pilot.sh run; exec bash'
```

`check` allocates no GPU. Enter a secret of at least 32 characters and use the same value on Windows; never commit or share it in chat. The launcher selects physical GPU 2 by UUID, holds the project lease and refuses occupancy above 1 GiB. It neither runs `ray stop` nor modifies baseline jobs. Wait for `Online learner ready`.

The pilot defaults to Stage 1 step 2000, 64 records before learning, batch 32, at most 5000 updates, publication every eight updates and checkpoints every 50 records. Actor execution requires at least 128 updates and eligible BC MSE at most 0.01 on the latest publication minibatch; otherwise reference continues. This is neither held-out evaluation nor a safety certificate. The shorter `run_rlt_vr_online_gpu2.sh run` uses eight records before learning, batch eight, at most 128 updates, publication every four updates and checkpoints every sixteen records. Resume with the original configuration.

## Verify Local Windows Control

Once the server is ready, keep Business Streaming and SteamVR tracking active. Open a tunnel in one PowerShell, replacing `SERVER_ADDRESS` with your SSH address:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8775:127.0.0.1:8775 luokz@SERVER_ADDRESS
```

Keep the tunnel open. In another PowerShell, synchronize the VR branch and create a fresh recording. Both endpoints must run the same revision:

```powershell
Set-Location "C:\Users\lkz\Desktop\code\UPT_vr_dev"
git pull --ff-only origin feature/rlt-pico-vr-intervention
$env:RLT_VR_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Same connection secret as the server' -AsSecureString)).Password
$vrRecord = "C:\Users\lkz\Desktop\rlt-records\acceptance-v2-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
conda run --no-capture-output --name rlt-vr python -X faulthandler -u -m toolkits.rlt_vr.client --online --port 8775 --render-backend cpu --record "$vrRecord" --max-episode-steps 1000 --log-interval 1
```

The client starts paused. Hold grip and move slowly; each fresh takeover anchors the controller to the current TCP. A fresh trigger press toggles the gripper target; releasing grip does not open it. Input and display remain local, without directly waiting for remote inference.

Control remains 10 Hz. Defaults are a 0.12 s target-smoothing constant, 0.12 m/s translation and 0.8 rad/s rotation limits. Joint increments are bounded to 0.025 rad/step, with adjacent-command limits and a 0.02 rad joint-limit margin. Relative targets remain bounded to 15 cm / 30°. These constrain commands, not measured velocity, collision avoidance or real-robot safety. `--tcp-speed` and `--angular-speed` change target rates; increasing them is not a remedy for contact stalls.

## Separate Takeover from Approved Corrections

After checking following, press `P` to request model actions and take over mid-execution. Releasing grip requests review of the last human fragment: `Y` approves it for BC; `N` excludes it. Re-grip or press `P` after review. `R` resets, Space pauses and `Q` quits. Pending fragments at reset or exit become `unreviewed`, never implicitly approved.

| Label | Replay and training use |
|---|---|
| `human + approved` | Ordinary and correction replay; executed action is the BC target |
| `human + rejected/unreviewed` | Retained for critic learning and analysis; not a direct BC target |
| `reference/actor + policy` | Ordinary replay; BC uses the frozen VLA's first reference action |

Raw `.npz` files are immutable; reviews are separate `review_*.json` files. Approval is an operator judgment, not automatic success detection. Reviews cover whole fragments and cannot revoke uploaded approval; use short, promptly reviewed segments. Half the batch comes from correction replay and half from all-action replay, so the realized human fraction can exceed one half. Rewards remain raw sparse rewards, without a bonus for grip or approval; both termination and truncation block bootstrap in this configuration.

## Read Pause Causes and Processing Progress

Upload starts only after review and preserves execution order. Later transitions cannot bypass unreviewed fragments. The display retains pause reasons and recovery hints; restored tracking or queue capacity never automatically resumes motion.

| Cause | Recovery |
|---|---|
| `tracking_invalid` / `ui_stall` | Check tracking or local stalls; release and re-grip |
| `ik_failed` / `no_progress` | Inspect target, contact and posture; re-anchor with a smaller target |
| Relative motion limit | Release, reposition and re-grip; this limit does not end the episode |
| `review_required` | Release grip, then press `Y` or `N` |
| `local_outbox_full` | Review pending fragments and wait for receipts; release before resuming |
| `upload_error` / `server_storage_full` / `local_storage_full` | Stop, retain records and inspect connection, server faults or quotas |
| `episode_ended` | Only `R` starts another episode; grip cannot clear it |

`no_progress` requires 20 consecutive control steps with position error above 2.5 cm and actual movement below 0.2 mm/step. It does not identify collision, joint limits or dynamics lag. Logs separate IK, simulator-step and journal-write time, and report target error, joint limiting and actual TCP motion. Raw samples include `teleop_json`; sparse control events go to `events.jsonl`.

The server validates and synchronously commits requests to `inbox/` before acknowledging receipt. One background model owner alternates prediction and feature/learning work. Network admission does not wait for learning, but a model operation cannot be preempted. This does not promise 10 Hz remote inference or optimization.

| Counter | Meaning |
|---|---|
| `received` / `received_sequence` | Durably received sequence, zero-based; not update count |
| `processed` / `sequence` | Client sequence processed into feature replay |
| `local_pending` / `outbox` | Queued and in-flight records, including pending review |
| `server_pending` / `pending_learning` | Durable receipts not yet fully processed |
| `approved_accepted`, `update_step`, `policy_version` | Processed approved corrections, optimizer steps, published version |

Defaults are 512 local outbox filenames, an 8 GiB raw-record quota and 128 MiB free reserve. Server defaults are 128 pending records, a 4 GiB journal quota and 1 GiB free reserve. Exhaustion applies backpressure and eventually pauses simulation; records are not dropped and queues are not unbounded. Continuous grip can fill 512 unreviewed slots in about 51 seconds, so operate in reviewed segments.

Local writes still perform synchronous `fsync`; slow storage can affect control. This decouples network and learning, not all local work, and is not hard-real-time control. Checkpoints and feature replay require space beyond the raw journal quota. Records are never automatically deleted.

## Retain Evidence When Resuming

For normal shutdown, press `Q`, wait for upload shutdown, then Ctrl-C on the server. Windows `receipt.json` identifies the last acknowledged sequence and server run. The server retains `inbox/`, processed `metrics.jsonl`, `learner.pt` and `config.json`, without creating a W&B run. Keep both sides after faults. Do not relabel old records as a new session and resend everything: that can duplicate training data.

Resume a schema-2 checkpoint with matching protocol, configuration and frozen-feature identity:

```bash
bash run_rlt_vr_hil_pilot.sh run --resume /absolute/path/previous-run/learner.pt
/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m toolkits.rlt_vr.summarize_online /absolute/path/run
```

The server replays durable receipts after its last good checkpoint before admitting a new Windows session. Checkpoints reference all ancestor `inbox/` directories: do not delete/move them or copy `learner.pt` alone. Active simulation and Windows records without acknowledgement are not automatically recovered. Retain those files and `receipt.json` for reconciliation; do not assume they were learned. Per-run logs remain separate while cumulative model counters can span runs.

The report separates human and approved-correction counts and actual reference/actor steps. Increasing `policy_version` is not evidence of actor execution: a real transition must have `policy_source=actor`. `service_ms` measures feature/learning work, not end-to-end latency. `update_budget_exhausted=true` stops optimization, not ingestion or prediction.

## Accept the Complete Loop

Use short engineering recordings before formal collection: reference execution → human correction → release and approve with `Y` → increasing `approved_accepted`, `update_step` and `policy_version` → actor passes its gate and actually executes after `P` → human takes over again. Keep reference active if the actor is not ready; do not weaken the gate to claim acceptance.

CPU tests cover control state, the real client loop, actual loopback sockets, ordered upload, delayed learning, quotas, labels, faults/recovery and synthetic-feature reference/actor/human switching. They do not establish Windows/PICO smoothness or task-success improvement. Historical GPU smoke is not acceptance of this asynchronous revision. See [verification evidence](VR_VERIFICATION.md) for current limitations and failed checks. Automatic phase routing, OpenPI expert, formal ten-step partial chunks and 64-environment worker/replay integration are not part of this entrypoint.
