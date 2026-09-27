# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run local simulation/display/VR; request policy chunks over an SSH tunnel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .control import MappedTarget, OperatorControl, map_relative_target
from .online_transport import TransitionUploader
from .protocol import CAMERAS, CONTRACT, encode_image, request
from .simulation import DEFAULT_RENDER_BACKEND, LocalSimulation
from .vr import SteamVRController

logger = logging.getLogger(__name__)


class TransitionRecorder:
    """Write executed one-step transitions, not proposed actions, to a new run.

    Files are readable with allow_pickle=False. These are collection artifacts,
    not direct inputs for the baseline chunk-level online replay buffer.
    """

    def __init__(self, directory: Path, metadata: dict) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.index = 0
        with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump({"contract": CONTRACT, **metadata}, stream, indent=2)

    def append(
        self,
        observation: dict,
        action: np.ndarray,
        next_observation: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        source: str,
        episode: int,
        model_id: str,
    ) -> None:
        """Persist a transition only after the corresponding env.step succeeds."""
        if source not in {"human", "policy"}:
            raise ValueError("Only executed human or policy actions may be recorded")
        values = {
            **observation,
            **{f"next_{k}": v for k, v in next_observation.items()},
        }
        with (self.directory / f"step_{self.index:08d}.npz").open("xb") as stream:
            np.savez(
                stream,
                **values,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                human_intervention=source == "human",
                source=source,
                episode=episode,
                model_id=model_id,
            )
        self.index += 1


def run(args: argparse.Namespace) -> None:
    """Keep network inference off the local input/display thread."""
    import cv2

    token = os.environ.get("RLT_VR_TOKEN", "")
    model_id = "manual-only"
    online = getattr(args, "online", False)
    uploader = None
    session = uuid.uuid4().hex
    sequence = 0
    behavior_version = -1
    if not args.manual_only:
        health = request(
            "127.0.0.1", args.port, token, {"op": "health", "request_id": 0}, 5
        )
        if health["contract"] != CONTRACT:
            raise ValueError("Server environment contract does not match this client")
        model_id = health["model_id"]
        if online:
            if not health.get("online") or health.get("horizon") != 1:
                raise ValueError("--online requires the single-step online server")
            request(
                "127.0.0.1",
                args.port,
                token,
                {"op": "begin", "request_id": 0, "session": session},
                10,
            )
    recorder = None
    if args.record is not None:
        recorder = TransitionRecorder(
            args.record,
            {
                "model_id": model_id,
                "seed": args.seed,
                "yaw_degrees": args.yaw_degrees,
                "max_episode_steps": args.max_episode_steps,
                "online": online,
                "session": session,
            },
        )
    env = None
    vr = None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlt-inference")
    pending: Future | None = None
    operator = OperatorControl()
    gate = operator.gate
    generation, submitted = -1, 0.0
    request_id, episode, episode_steps = 0, 0, 0
    anchor_vr = anchor_tcp = None
    gripper_command = 1.0
    trigger_was_down = False
    last_step = last_loop = last_log = time.monotonic()
    last_mapping: MappedTarget | None = None
    last_action: np.ndarray | None = None
    last_step_ms = 0.0
    was_limited = False
    was_valid = True
    status = "PAUSED: P=policy, grip=human, Space=pause, R=reset, Q=quit"
    try:
        if online:
            uploader = TransitionUploader(args.port, token, session)
        env = LocalSimulation(
            args.render_backend,
            args.seed,
            args.max_episode_steps,
        )
        if not args.no_vr:
            vr = SteamVRController(
                args.clutch_button,
                args.trigger_button,
                args.trigger_threshold,
            )
        observation = env.observation()
        # Environment and SteamVR initialization may take seconds on Windows.
        # Start the watchdog only after both are ready, not before construction.
        last_step = last_loop = last_log = time.monotonic()
        logger.info(
            "Control ready: scale=%.2f max_translation=%.3fm "
            "max_rotation=%.1fdeg episode_steps=%d stall_timeout=%.1fs",
            args.translation_scale,
            args.max_displacement,
            args.max_rotation_degrees,
            args.max_episode_steps,
            args.stall_timeout,
        )
        while True:
            now = time.monotonic()
            # A long input/render stall requires deliberate re-arming.
            loop_elapsed = now - last_loop
            stalled = loop_elapsed > args.stall_timeout
            last_loop = now
            reading = vr.read() if vr else None
            valid = reading is None or reading.valid
            panel = np.concatenate([observation[k] for k in CAMERAS], axis=1)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.putText(
                panel, f"{gate.mode} | {status}", (5, 20), 0, 0.4, (0, 255, 255), 1
            )
            if reading is not None:
                diagnostic = (
                    f"buttons={reading.buttons:#x} grip={int(reading.clutch)} "
                    f"trigger={reading.trigger_value:.2f} step={last_step_ms:.0f}ms"
                )
                cv2.putText(panel, diagnostic, (5, 38), 0, 0.4, (0, 255, 255), 1)
            cv2.imshow("RLT local simulation: main / wrist", panel)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (ord("q"), 27)
                or cv2.getWindowProperty(
                    "RLT local simulation: main / wrist", cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                break
            if key == ord("r"):
                operator.reset()
                episode += 1
                episode_steps = 0
                observation = env.reset(args.seed + episode)
                anchor_vr = anchor_tcp = None
                last_mapping = None
                last_action = None
                was_limited = False
                status = "Reset; paused"
                last_loop = time.monotonic()
                continue
            previous_mode = gate.mode
            release_was_required = operator.require_release
            command = "pause" if key == ord(" ") else ""
            if key == ord("p") and not args.manual_only:
                command = "policy"
            operator.update(
                valid=valid,
                clutch=bool(reading and reading.clutch),
                command=command,
                stalled=stalled,
            )
            if uploader is not None and not uploader.ready:
                operator.pause()
                status = (
                    uploader.error
                    or "Upload queue full; paused until drained; release grip"
                )
            if not valid:
                status = "Tracking invalid: check SteamVR; release grip before retry"
                if was_valid:
                    logger.warning("Tracking became invalid; motion paused")
            elif stalled:
                status = f"Safety pause after {loop_elapsed:.1f}s stall; release grip"
                logger.warning(
                    "Display/input loop stalled for %.3fs; motion paused",
                    loop_elapsed,
                )
            elif release_was_required and not operator.require_release:
                status = "Ready: hold grip to establish a fresh motion anchor"
                logger.info("Safety latch cleared after grip release")
            was_valid = valid
            if previous_mode != gate.mode:
                if valid and not stalled:
                    status = f"{gate.mode}: grip=human, P=policy, Space=pause"
                logger.info("Control authority: %s -> %s", previous_mode, gate.mode)
                if gate.mode == "human":
                    anchor_vr, anchor_tcp = reading.pose.copy(), env.tcp_matrix()
                    # Preserve the grasp on takeover. Each fresh trigger press
                    # toggles the latched target; releasing grip never opens it.
                    finger = float(np.mean(observation["state"][7:9]))
                    gripper_command = float(
                        np.clip(2 * (finger + 0.01) / 0.05 - 1, -1, 1)
                    )
                    trigger_was_down = reading.close_gripper
                    was_limited = False
                    logger.info(
                        "Human anchor: controller_xyz=%s tcp_xyz=%s gripper=%+.1f",
                        np.round(anchor_vr[:3, 3], 3),
                        np.round(anchor_tcp[:3, 3], 3),
                        gripper_command,
                    )
                else:
                    anchor_vr = anchor_tcp = None
                    last_mapping = None
                    was_limited = False
            if gate.mode == "human" and reading is not None:
                if reading.close_gripper and not trigger_was_down:
                    gripper_command = -1.0 if gripper_command > 0 else 1.0
                    logger.info(
                        "Gripper toggled to %+.1f (buttons=%#x analog=%.3f)",
                        gripper_command,
                        reading.buttons,
                        reading.trigger_value,
                    )
                    status = f"human: gripper command {gripper_command:+.1f}"
                trigger_was_down = reading.close_gripper

            if pending is not None and pending.done():
                try:
                    response = pending.result()
                    if time.monotonic() - submitted <= args.reply_ttl:
                        accepted = gate.accept(generation, response["actions"])
                        if accepted:
                            model_id = response["model_id"]
                            behavior_version = int(
                                response.get("metrics", {}).get("policy_version", -1)
                            )
                            status = (
                                f"RPC {(time.monotonic() - submitted) * 1000:.0f} ms"
                            )
                    elif generation == gate.generation:
                        operator.pause()
                        status = "Reply expired; P retries with a fresh observation"
                except Exception as error:
                    if generation == gate.generation:
                        operator.pause()
                        status = "Network/inference failure; P retries"
                    logger.warning("Inference failed: %s", type(error).__name__)
                pending = None

            if gate.needs_prediction and pending is None:
                request_id += 1
                generation, submitted = gate.generation, time.monotonic()
                payload = {
                    "op": "predict",
                    "request_id": request_id,
                    "contract": CONTRACT,
                    **({"session": session} if online else {}),
                    "state": observation["state"].tolist(),
                    **{k: encode_image(observation[k]) for k in CAMERAS},
                }
                pending = pool.submit(
                    request, "127.0.0.1", args.port, token, payload, args.reply_ttl
                )

            if now - last_step >= 0.1:
                action = None
                if gate.mode == "human" and reading is not None:
                    last_mapping = map_relative_target(
                        anchor_vr,
                        reading.pose,
                        anchor_tcp,
                        scale=args.translation_scale,
                        yaw_degrees=args.yaw_degrees,
                        max_displacement=args.max_displacement,
                        max_rotation=np.deg2rad(args.max_rotation_degrees),
                    )
                    if last_mapping.limited:
                        status = "Motion limit reached: release grip, recenter, re-grip"
                        if not was_limited:
                            logger.warning(
                                "Relative motion limited: translation %.3f/%.3fm, "
                                "rotation %.1f/%.1fdeg; release and re-grip",
                                last_mapping.requested_translation,
                                last_mapping.applied_translation,
                                np.rad2deg(last_mapping.requested_rotation),
                                np.rad2deg(last_mapping.applied_rotation),
                            )
                    elif was_limited:
                        status = "human: inside configured motion limits"
                    was_limited = last_mapping.limited
                    action = env.human_action(last_mapping.pose, gripper_command)
                    if action is None:
                        operator.pause()
                        status = "IK failed; release grip and try a smaller motion"
                        logger.warning("IK failed; motion paused")
                elif gate.mode == "policy":
                    action = gate.next_action()
                if action is not None:
                    step_started = time.monotonic()
                    next_obs, reward, terminated, truncated = env.step(action)
                    episode_steps += 1
                    last_step_ms = (time.monotonic() - step_started) * 1000
                    last_action = action.copy()
                    if recorder is not None:
                        recorder.append(
                            observation,
                            action,
                            next_obs,
                            reward,
                            terminated,
                            truncated,
                            gate.mode,
                            episode,
                            model_id,
                        )
                    if uploader is not None:
                        uploader.submit(
                            sequence,
                            observation,
                            next_obs,
                            action=action.tolist(),
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            human=gate.mode == "human",
                            episode=episode,
                            policy_version=-1
                            if gate.mode == "human"
                            else behavior_version,
                        )
                        sequence += 1
                    observation = next_obs
                    if terminated or truncated:
                        operator.finish()
                        reasons = []
                        if terminated:
                            reasons.append("task terminal")
                        if truncated:
                            reasons.append(
                                f"time limit at {episode_steps}/{args.max_episode_steps}"
                            )
                        reason = " + ".join(reasons)
                        status = f"Episode ended: {reason}; R resets"
                        logger.info(
                            "Episode %d ended: terminated=%s truncated=%s "
                            "steps=%d/%d reward=%.3f",
                            episode,
                            terminated,
                            truncated,
                            episode_steps,
                            args.max_episode_steps,
                            reward,
                        )
                last_step = now
            if now - last_log >= args.log_interval:
                if uploader is not None:
                    logger.info(
                        "Online learner: ack=%d pending=%d metrics=%s",
                        uploader.accepted_sequence,
                        uploader.queue.qsize(),
                        uploader.metrics,
                    )
                if reading is not None:
                    mapping_text = "n/a"
                    if last_mapping is not None:
                        mapping_text = (
                            f"translation={last_mapping.requested_translation:.3f}/"
                            f"{last_mapping.applied_translation:.3f}m "
                            f"rotation={np.rad2deg(last_mapping.requested_rotation):.1f}/"
                            f"{np.rad2deg(last_mapping.applied_rotation):.1f}deg "
                            f"limited={last_mapping.limited}"
                        )
                    arm_max = (
                        float(np.max(np.abs(last_action[:7])))
                        if last_action is not None
                        else 0.0
                    )
                    logger.info(
                        "Telemetry mode=%s valid=%s buttons=%#x grip=%s "
                        "trigger=%s(%.3f) %s arm_max=%.3f gripper=%+.1f "
                        "step=%.1fms loop=%.1fms",
                        gate.mode,
                        valid,
                        reading.buttons,
                        reading.clutch,
                        reading.close_gripper,
                        reading.trigger_value,
                        mapping_text,
                        arm_max,
                        gripper_command,
                        last_step_ms,
                        loop_elapsed * 1000,
                    )
                last_log = now
            time.sleep(0.005)
    finally:
        gate.reset()
        if vr is not None:
            vr.close()
        if env is not None:
            env.close()
        cv2.destroyAllWindows()
        pool.shutdown(wait=True, cancel_futures=True)
        if uploader is not None:
            uploader.close()


def main() -> None:
    """Parse local-only controls; autonomous motion always requires P."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--render-backend", default=DEFAULT_RENDER_BACKEND)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--reply-ttl", type=float, default=10)
    parser.add_argument("--yaw-degrees", type=float, default=0)
    parser.add_argument("--clutch-button", type=int, default=2)
    parser.add_argument("--trigger-button", type=int, default=33)
    parser.add_argument("--trigger-threshold", type=float, default=0.6)
    parser.add_argument("--translation-scale", type=float, default=0.5)
    parser.add_argument("--max-displacement", type=float, default=0.15)
    parser.add_argument("--max-rotation-degrees", type=float, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--stall-timeout", type=float, default=2.0)
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Send executed steps to the h=1 online learner; requires --record",
    )
    parser.add_argument(
        "--no-vr", action="store_true", help="Explicit keyboard-only policy test"
    )
    args = parser.parse_args()
    if args.online and (args.manual_only or args.record is None):
        parser.error(
            "--online requires --record and cannot be combined with --manual-only"
        )
    if not 0 < args.reply_ttl <= 30:
        parser.error("--reply-ttl must be in (0, 30] seconds")
    if not 0 < args.trigger_threshold <= 1:
        parser.error("--trigger-threshold must be in (0, 1]")
    if not 0 < args.translation_scale <= 1:
        parser.error("--translation-scale must be in (0, 1]")
    if not 0 < args.max_displacement <= 0.3:
        parser.error("--max-displacement must be in (0, 0.3] meters")
    if not 0 < args.max_rotation_degrees <= 90:
        parser.error("--max-rotation-degrees must be in (0, 90]")
    if not 1 <= args.max_episode_steps <= 10000:
        parser.error("--max-episode-steps must be in [1, 10000]")
    if not 0.5 <= args.stall_timeout <= 10:
        parser.error("--stall-timeout must be in [0.5, 10] seconds")
    if not 0.2 <= args.log_interval <= 10:
        parser.error("--log-interval must be in [0.2, 10] seconds")
    logging.basicConfig(level=logging.INFO)
    run(args)


if __name__ == "__main__":
    main()
