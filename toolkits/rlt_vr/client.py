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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .control import OperatorControl, relative_target
from .protocol import CAMERAS, CONTRACT, encode_image, request
from .simulation import LocalSimulation
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
    if not args.manual_only:
        health = request(
            "127.0.0.1", args.port, token, {"op": "health", "request_id": 0}, 5
        )
        if health["contract"] != CONTRACT:
            raise ValueError("Server environment contract does not match this client")
        model_id = health["model_id"]
    recorder = None
    if args.record is not None:
        recorder = TransitionRecorder(
            args.record,
            {"model_id": model_id, "seed": args.seed, "yaw_degrees": args.yaw_degrees},
        )
    env = None
    vr = None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlt-inference")
    pending: Future | None = None
    operator = OperatorControl()
    gate = operator.gate
    generation, submitted = -1, 0.0
    request_id, episode = 0, 0
    anchor_vr = anchor_tcp = None
    gripper_command = 1.0
    trigger_was_down = False
    last_step = last_loop = time.monotonic()
    status = "PAUSED: P=policy, grip=human, Space=pause, R=reset, Q=quit"
    try:
        env = LocalSimulation(args.render_backend, args.seed)
        if not args.no_vr:
            vr = SteamVRController(args.clutch_button, args.trigger_button)
        observation = env.observation()
        while True:
            now = time.monotonic()
            # A long input/render stall requires deliberate re-arming.
            stalled = now - last_loop > 0.5
            last_loop = now
            reading = vr.read() if vr else None
            valid = reading is None or reading.valid
            panel = np.concatenate([observation[k] for k in CAMERAS], axis=1)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.putText(
                panel, f"{gate.mode} | {status}", (5, 20), 0, 0.4, (0, 255, 255), 1
            )
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
                observation = env.reset(args.seed + episode)
                anchor_vr = anchor_tcp = None
                status = "Reset; paused"
                last_loop = time.monotonic()
                continue
            previous_mode = gate.mode
            command = "pause" if key == ord(" ") else ""
            if key == ord("p") and not args.manual_only:
                command = "policy"
            operator.update(
                valid=valid,
                clutch=bool(reading and reading.clutch),
                command=command,
                stalled=stalled,
            )
            if not valid:
                status = "Tracking invalid: check SteamVR; release grip before retry"
            elif stalled:
                status = "UI stalled: paused; release grip before retry"
            if previous_mode != gate.mode:
                if valid and not stalled:
                    status = f"{gate.mode}: grip=human, P=policy, Space=pause"
                if gate.mode == "human":
                    anchor_vr, anchor_tcp = reading.pose.copy(), env.tcp_matrix()
                    # Preserve the grasp on takeover. Each fresh trigger press
                    # toggles the latched target; releasing grip never opens it.
                    finger = float(np.mean(observation["state"][7:9]))
                    gripper_command = float(
                        np.clip(2 * (finger + 0.01) / 0.05 - 1, -1, 1)
                    )
                    trigger_was_down = reading.close_gripper
                else:
                    anchor_vr = anchor_tcp = None
            if gate.mode == "human" and reading is not None:
                if reading.close_gripper and not trigger_was_down:
                    gripper_command = -1.0 if gripper_command > 0 else 1.0
                trigger_was_down = reading.close_gripper

            if pending is not None and pending.done():
                try:
                    response = pending.result()
                    if time.monotonic() - submitted <= args.reply_ttl:
                        accepted = gate.accept(generation, response["actions"])
                        if accepted:
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
                    "state": observation["state"].tolist(),
                    **{k: encode_image(observation[k]) for k in CAMERAS},
                }
                pending = pool.submit(
                    request, "127.0.0.1", args.port, token, payload, args.reply_ttl
                )

            if now - last_step >= 0.1:
                action = None
                if gate.mode == "human" and reading is not None:
                    target = relative_target(
                        anchor_vr,
                        reading.pose,
                        anchor_tcp,
                        yaw_degrees=args.yaw_degrees,
                    )
                    action = env.human_action(target, gripper_command)
                    if action is None:
                        operator.pause()
                        status = "IK failed; release grip and try a smaller motion"
                elif gate.mode == "policy":
                    action = gate.next_action()
                if action is not None:
                    next_obs, reward, terminated, truncated = env.step(action)
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
                    observation = next_obs
                    if terminated or truncated:
                        operator.finish()
                        status = (
                            f"Episode ended (reward={reward}); R starts next episode"
                        )
                last_step = now
            time.sleep(0.005)
    finally:
        gate.reset()
        if vr is not None:
            vr.close()
        if env is not None:
            env.close()
        cv2.destroyAllWindows()
        pool.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    """Parse local-only controls; autonomous motion always requires P."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--reply-ttl", type=float, default=10)
    parser.add_argument("--yaw-degrees", type=float, default=0)
    parser.add_argument("--clutch-button", type=int, default=2)
    parser.add_argument("--trigger-button", type=int, default=33)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument(
        "--no-vr", action="store_true", help="Explicit keyboard-only policy test"
    )
    args = parser.parse_args()
    if not 0 < args.reply_ttl <= 30:
        parser.error("--reply-ttl must be in (0, 30] seconds")
    logging.basicConfig(level=logging.INFO)
    run(args)


if __name__ == "__main__":
    main()
