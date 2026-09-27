# Local Simulation and VR Takeover With Remote Inference

This guide runs one ManiSkill environment, its camera display, and PICO intervention on a Windows laptop while a Linux server generates RLT actions. Verify simulation first, controller input second, and remote inference last. Online training integration follows hardware acceptance, not just a successful socket connection.

The branch is `feature/rlt-pico-vr-intervention`, based on baseline `ff566637`, with its own worktree at `/home/luokz/rlinf_rlt/UPT_vr_dev`. It is not merged into baseline and contains no FLARE algorithm changes. Existing Stage 1 jobs and Stage 2 entrypoints are unchanged.

## Where Each Component Runs

The local PC owns human control, so taking over does not wait for server inference. Networking still affects autonomous action availability, but it does not stream the simulator display.

```text
PICO → PICO Business Streaming / SteamVR → right-controller pose/buttons
                                                      ↓
Windows: CPU physics ← local IK / intervention gate ← operator
          ↓ two RGB images + 9 joint positions       ↑ local camera window
          └──────────────── SSH tunnel ──────────────────┐
Server: frozen Stage1 feature model → optional Stage2 actor
          └──────────── 10×8 action chunk ───────────→ Windows steps
```

Actions retain `pd_joint_delta_pos`: seven normalized joint deltas and one gripper command. Controller 6DoF motion passes through a coordinate transform and IK; it is not substituted directly into the eight-dimensional policy action. Model observations retain the baseline `3rd_view_camera` and `wide_hand_camera`, at 384×384 RGB, transmitted losslessly with PNG.

The display is a local OpenCV two-camera window, not a head-tracked stereoscopic VR application. Start by watching the laptop screen. A headset desktop view may consume SteamVR input and needs separate testing.

## Check Windows Simulation

The RTX 4060 8 GB does not load VLA weights and is a reasonable candidate for single-environment rendering. Actual throughput depends on the CPU, graphics driver, and concurrent SteamVR workload. The official matrix supports CPU simulation and rendering on Windows, not GPU simulation. This client fixes `num_envs=1` and `sim_backend=physx_cpu`. It defaults to `render_backend=cpu` on Windows to avoid SAPIEN 3.0.1's CUDA image-interoperability path, while Linux keeps `gpu`. Do not substitute WSL for native Windows graphics. [ManiSkill system support](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)

SAPIEN 3.0.1 publishes a `cp311-win_amd64` wheel. Availability is not proof of compatibility on this laptop. [SAPIEN release files](https://pypi.org/project/sapien/3.0.1/#files)

Use a dedicated PowerShell environment. The PyTorch command uses CUDA 12.6 rather than the server's 12.8 build, following the [official version instructions](https://pytorch.org/get-started/previous-versions/#v271). Do not run the full RLinf server installer on the laptop.

```powershell
git clone --branch feature/rlt-pico-vr-intervention https://github.com/nkd-lkz/UPT_dev.git UPT_vr_dev
cd UPT_vr_dev
py -3.11 -m venv .venv-vr
$PY = ".\.venv-vr\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
& $PY -m pip install -r toolkits/rlt_vr/requirements-client.txt
& $PY -m pip check
& $PY -m toolkits.rlt_vr.preflight --probe sim --steps 20
```

This checks the actual peg-insertion task, both cameras, and IK before executing 20 arm-hold steps. It neither opens a window nor contacts the server. A successful run reports `PASS` and step timings. The client solves Panda IK with ManiSkill's pinned `pytorch_kinematics` dependency. Keep Pinocchio out of this Windows environment because its LLVM OpenMP runtime cannot safely share this process with PyTorch's Intel OpenMP runtime. If a wheel or Vulkan fails, retain the full error. Changing simulator versions requires a new regression check, not an implicit claim of baseline parity.

## Check PICO Input and Manual Control

Once simulation works, connect PICO Business Streaming and SteamVR and verify that the right controller is visible. Try USB first to remove wireless variability. PICO documents a Windows PC application with a SteamVR dependency. [PICO Business Streaming](https://business.picoxr.com/us/software/streaming-assistant)

```powershell
& $PY -m toolkits.rlt_vr.preflight --probe vr --steps 60
& $PY -m toolkits.rlt_vr.client --manual-only --record C:\rlt-records\manual-001 --max-episode-steps 1000 --log-interval 1
```

The probe reports tracking validity, the button bitmask, grip/trigger state, analog `trigger_value`, and position. Holding the side grip must set `grip=True`, and pressing the index trigger must set `trigger=True` or raise `trigger_value`. Tracking should remain valid while moving. The client accepts `--clutch-button` and `--trigger-button` to override defaults 2 and 33; it also discovers a trigger axis advertised by SteamVR and treats values at or above `--trigger-threshold 0.6` as pressed. Never bypass the validity check to make the robot move.

| Input | Local behavior |
|---|---|
| Hold right grip | Anchor the current controller/TCP poses and take over |
| Move/rotate controller | Local IK; translation scale 0.5, at most 15 cm and 30 degrees from each clutch anchor |
| Fresh trigger press during takeover | Toggle the latched gripper target; releasing trigger keeps it |
| Release grip | Pause, retain gripper target, discard remaining policy actions |
| `P` | Request fresh policy inference; disabled in manual-only mode |
| Space, invalid tracking, UI stall, or IK failure | Pause and require grip release before another takeover |
| `R` | New paused episode; discard pre-reset replies |
| `Q`, Esc, or closing the window | Exit and release local resources |

Check all translation and rotation directions in free space before attempting grasping. OpenVR right/up/back maps to robot forward/left/up; `--yaw-degrees` adjusts standing orientation. IK is not collision avoidance and does not guarantee stable contact. These limits are for simulation debugging, not real-robot safety.

The terminal prints one telemetry record per second with control ownership, tracking, raw buttons, analog trigger, requested/applied relative motion, the largest arm command, gripper command, simulation-step time, and display-loop time. The same button and timing summary appears on the camera window. Reaching a relative-motion bound intentionally holds the target and reports `Motion limit reached`; release grip, move the controller back to a comfortable pose, and hold grip again to create a new anchor. For a larger simulation workspace, use an explicit bound up to 30 cm, for example `--max-displacement 0.25`; keep the default until every direction is calibrated.

The registered training task ends after 100 control steps, which is about ten seconds of executed motion at 10 Hz. Manual calibration uses `--max-episode-steps 1000` to provide about 100 seconds without changing control frequency or model inputs. The client reports `terminated` and `truncated` separately: task success causes termination, while reaching this wrapper limit causes truncation. Press `R` after either event. Keep the baseline value 100 when comparing task-level evaluation results.

The display watchdog starts after simulator and SteamVR initialization and pauses control only when one loop exceeds `--stall-timeout 2.0`. After a watchdog, tracking, or IK fault, release grip once to clear the safety latch. Raising the timeout up to ten seconds helps diagnose a slow Windows renderer, but it also delays fault detection and is not a real-robot setting.

## Start Server Inference

Only occupy the inference GPU after local takeover works. The command below reuses the server environment and starts frozen inference, not Ray, training, W&B, or a simulator. Confirm GPU 2 is still free with `nvidia-smi`.

Set the same random secret of at least 32 characters on both machines, using hidden terminal input and your password manager. Never share it in chat, commit it, or store it in launch scripts.

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
source /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
read -rs -p 'RLT VR token: ' RLT_VR_TOKEN
export RLT_VR_TOKEN
export RLT_STORAGE=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill
python -m toolkits.rlt_vr.server \
  --stage1 "$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750/actor" \
  --dataset "$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
```

Wait for `Inference ready`. Default inference returns Stage 1 `ref_chunk`. Supply `--actor /absolute/path/actor/model_state_dict/full_weights.pt` to test a Stage 2 head. That mode uses the actor for every chunk; it does not reproduce baseline automatic phase routing or model-expert takeover. Human actions still originate locally. Load only trusted checkpoints.

## Connect Windows and Interrupt a Policy

The server listens only on `127.0.0.1:8765`. Campus networking does not guarantee host-to-host access. Establish SSH reachability first, then tunnel inference rather than exposing images or credentials over the LAN.

Keep this command running in a separate PowerShell terminal, replacing the address with the server's actual SSH address:

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:8765:127.0.0.1:8765 luokz@SERVER_IP
```

In the client terminal, enter the matching secret and check small-message networking, full inference, and finally interactive takeover:

```powershell
$env:RLT_VR_TOKEN = [System.Net.NetworkCredential]::new("", (Read-Host "RLT VR token" -AsSecureString)).Password
& $PY -m toolkits.rlt_vr.preflight --probe network --steps 20
& $PY -m toolkits.rlt_vr.preflight --probe inference --steps 5
& $PY -m toolkits.rlt_vr.client --record C:\rlt-records\intervention-001
```

The window starts paused. Press `P` to run model actions, then hold grip to flush the remaining chunk and take over at the next available local control step. Use `--no-vr` for keyboard-only networking/display checks if necessary; that is not VR acceptance.

Physics retains baseline 10 Hz control, receiving at most ten actions per request. Input polling is more frequent, but camera frames update with simulation steps; this is not a 60 FPS promise. Physics pauses while waiting for inference, while the UI and controller continue polling. Human control can interrupt that wait. Responses older than `--reply-ttl` (default ten seconds), from previous episodes, or from previous control ownership are discarded. This is a pausable collection tool, not a wall-clock real-time controller.

Network p95 measures only small health messages. The inference probe also includes PNG encoding, transmission, model inference, and a simulator step. Server-only 161 ms inference is not a laptop latency measurement. Record to local SSD: uncompressed before/after RGB is about 1.8 MB per step. Check disk capacity before long sessions, then archive to NAS afterward.

## Carry Intervention Data Into Stage 2

The implemented loop ends with recording the transition that actually occurred. `metadata.json` identifies the environment contract; each `.npz` stores before/after images and qpos, executed action, raw reward, terminated/truncated, episode, model_id, and `human_intervention`. Pressing grip without a successful step does not produce a human-action label.

These files are not direct baseline replay inputs. The client uses individual steps, raw sparse rewards, and CPU physics, whereas baseline has chunk replay, RLT features, phase routing, and its own reward/termination handling. Online integration needs a separate tested change:

1. Split chunks at takeover boundaries and carry valid lengths/masks for partial chunks without fabricating actions.
2. Compute start/end features with the same frozen model and retain executed actions, per-step human masks, and policy versions.
3. Match reward accumulation, discounts, success/termination handling, time-limit bootstrap, and final observations before resets.
4. Deliver executed transitions to the actual learner training path. Test any human-sample sampling or imitation loss as a separate algorithmic change.
5. Complete Windows/PICO acceptance before considering a baseline merge. This inference-only server does not claim to train online from human samples.

## Ubuntu Fallback and Acceptance Scope

The Ubuntu RTX 3060 12 GB machine can reuse CPU single-environment simulation and the SSH inference protocol with the same simulator versions and control contract. PICO Business Streaming's PC software is Windows-based, however. Headset connectivity will not automatically migrate to Ubuntu: verify a compatible Linux VR runtime or add a separate Windows-to-Ubuntu input bridge, which this branch does not implement.

See [VR_VERIFICATION.md](VR_VERIFICATION.md) for evidence. Before merging, verify Windows reset/step and images, all six motion axes, gripper retention/toggling, mid-chunk takeover, fresh inference after release, cable/tracking/network loss, stale-response rejection, terminal-step blocking, and agreement between recorded and executed actions. The entire workflow is simulation-only.

Read [toolkits/rlt_vr](../../toolkits/rlt_vr) in this order: `simulation.py` and `client.py` for execution, `control.py` for control ownership, then `protocol.py`, `server.py`, and `vr.py`. The standalone transport uses standard-library logging to keep Ray out of the Windows dependency chain; server models retain RLinf logging.
