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

"""Check local simulation, tracking, or network independently before takeover."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import platform
import time

import numpy as np

from .protocol import CAMERAS, CONTRACT, encode_image, request, validate_actions

logger = logging.getLogger(__name__)


def main() -> None:
    """Run bounded checks without starting a training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", choices=["sim", "vr", "network", "inference"], required=True
    )
    from .simulation import DEFAULT_RENDER_BACKEND

    parser.add_argument("--render-backend", default=DEFAULT_RENDER_BACKEND)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("--steps must be in [1, 100]")
    logging.basicConfig(level=logging.INFO)
    logger.info("OS=%s Python=%s", platform.platform(), platform.python_version())
    if args.probe == "vr":
        from .vr import SteamVRController

        vr = SteamVRController()
        try:
            for _ in range(args.steps):
                reading = vr.read()
                logger.info(
                    "valid=%s buttons=%s grip=%s trigger=%s trigger_value=%.3f xyz=%s",
                    reading.valid,
                    hex(reading.buttons),
                    reading.clutch,
                    reading.close_gripper,
                    reading.trigger_value,
                    reading.pose[:3, 3].round(3),
                )
                time.sleep(0.5)
        finally:
            vr.close()
        return
    token = os.environ.get("RLT_VR_TOKEN", "")
    if args.probe == "network":
        timings = []
        for index in range(args.steps):
            start = time.monotonic()
            response = request(
                "127.0.0.1", args.port, token, {"op": "health", "request_id": index}, 5
            )
            if response["contract"] != CONTRACT:
                raise ValueError("Client/server contract mismatch")
            timings.append((time.monotonic() - start) * 1000)
        logger.info(
            "Health RTT ms p50=%.1f p95=%.1f (not full inference)",
            *np.percentile(timings, [50, 95]),
        )
        return
    from .simulation import LocalSimulation

    for package in ("mani_skill", "sapien", "torch", "numpy"):
        logger.info("%s=%s", package, importlib.metadata.version(package))
    env = LocalSimulation(args.render_backend)
    try:
        observation = env.observation()
        logger.info(
            "Observation shapes: %s", {k: v.shape for k, v in observation.items()}
        )
        action = env.human_action(env.tcp_matrix(), 1)
        if action is None or np.max(np.abs(action[:7])) > 0.01:
            raise RuntimeError("IK round trip failed at current TCP pose")
        timings = []
        for index in range(args.steps):
            start = time.monotonic()
            if args.probe == "inference":
                response = request(
                    "127.0.0.1",
                    args.port,
                    token,
                    {
                        "op": "predict",
                        "request_id": index,
                        "contract": CONTRACT,
                        "state": observation["state"].tolist(),
                        **{k: encode_image(observation[k]) for k in CAMERAS},
                    },
                    30,
                )
                chunk = validate_actions(response["actions"])
                # One action/request for validation; this is not baseline evaluation.
                action = chunk[0]
                logger.info(
                    "Model=%s inference_ms=%.1f",
                    response["model_id"],
                    response["inference_ms"],
                )
            observation, reward, terminated, truncated = env.step(action)
            timings.append((time.monotonic() - start) * 1000)
            if terminated or truncated:
                break
        logger.info(
            "PASS steps=%d step/RPC ms p50=%.1f p95=%.1f reward=%s",
            len(timings),
            *np.percentile(timings, [50, 95]),
            reward,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
