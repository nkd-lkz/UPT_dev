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

"""Run frozen OpenPI/RLT inference without Ray or a server-side simulator."""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import socketserver
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .protocol import (
    CONTRACT,
    decode_observation,
    receive,
    send,
    validate_actions,
)

logger = logging.getLogger(__name__)


class InferenceServer(socketserver.TCPServer):
    """Serialize inference requests with a single model owner and no job queue."""

    allow_reuse_address = True
    request_queue_size = 1

    def __init__(
        self,
        address: tuple[str, int],
        token: str,
        predict: Callable[[dict], np.ndarray],
        model_id: str,
        dispatch: Callable[[dict], dict] | None = None,
    ) -> None:
        if len(token) < 32:
            raise ValueError("Set RLT_VR_TOKEN to a secret of at least 32 characters")
        self.token = token
        self.predict = predict
        self.model_id = model_id
        self.dispatch = dispatch
        super().__init__(address, _Handler)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(10)
        request_id = None
        try:
            message = receive(self.request)
            supplied = message.get("token", "")
            if not isinstance(supplied, str) or not hmac.compare_digest(
                supplied.encode(), self.server.token.encode()
            ):
                raise ValueError("Authentication failed")
            request_id = message.get("request_id")
            if type(request_id) is not int or request_id < 0:
                raise ValueError("Invalid request id")
            if self.server.dispatch is not None:
                result = self.server.dispatch(message)
                send(self.request, {**result, "ok": True, "request_id": request_id})
                return
            if message.get("op") == "health":
                send(
                    self.request,
                    {
                        "ok": True,
                        "request_id": request_id,
                        "contract": CONTRACT,
                        "model_id": self.server.model_id,
                    },
                )
                return
            if message.get("op") != "predict":
                raise ValueError("Unknown operation")
            observation = decode_observation(message)
            start = time.monotonic()
            actions = validate_actions(self.server.predict(observation))
            send(
                self.request,
                {
                    "ok": True,
                    "request_id": request_id,
                    "actions": actions.tolist(),
                    "inference_ms": (time.monotonic() - start) * 1000,
                    "model_id": self.server.model_id,
                },
            )
        except Exception as error:
            # Do not log request contents: they contain credentials and images.
            logger.warning("RPC rejected: %s: %s", type(error).__name__, error)
            try:
                send(
                    self.request,
                    {
                        "ok": False,
                        "request_id": request_id,
                        "error": "Request rejected; inspect server configuration/logs",
                    },
                )
            except (OSError, ValueError):
                pass


class ConcurrentInferenceServer(socketserver.ThreadingMixIn, InferenceServer):
    """Bound frontend connections while one dispatch worker owns GPU state."""

    daemon_threads = True
    request_queue_size = 8

    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class RLTInference:
    """Load a trusted Stage1 export and an optional trusted Stage2 head.

    Base mode executes ref_chunk. Supplying an actor executes that head on every
    chunk, not the baseline environment's automatic phase-switching route.
    """

    def __init__(self, config: Path, stage1: Path, dataset: Path, actor: Path | None):
        import torch
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.openpi import get_model
        from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights

        if resolve_full_weights(stage1) is None:
            raise FileNotFoundError(
                "Stage1 must contain an RLinf full_weights.pt export"
            )
        if not (dataset / "norm_stats.json").is_file():
            raise FileNotFoundError("Dataset norm_stats.json is required")
        cfg = OmegaConf.merge(
            OmegaConf.load(config),
            {
                "model_path": str(stage1),
                "openpi_data": {
                    "repo_id": str(dataset),
                    "norm_stats_path": str(dataset / "norm_stats.json"),
                },
            },
        )
        self.feature = get_model(cfg).eval().to("cuda")
        self.actor = None
        if actor is not None:
            from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

            self.actor = RLTMLPPolicy(
                z_dim=2048,
                proprio_dim=9,
                action_dim=8,
                num_action_chunks=10,
                ref_num_action_chunks=10,
                add_q_head=True,
                q_head_type="default",
                fixed_std=0.002,
            ).eval()
            self.actor.load_state_dict(
                torch.load(actor, map_location="cpu", weights_only=True), strict=True
            )
            self.actor.to("cuda")

    def extract(self, observation: dict) -> dict:
        """Extract frozen RLT features for an unnormalized camera observation."""
        import torch

        env_obs = {
            "main_images": torch.from_numpy(observation["main_image"][None]),
            "wrist_images": torch.from_numpy(observation["wrist_image"][None]),
            "states": torch.from_numpy(observation["state"][None]),
            "task_descriptions": ["insert the peg in the hole"],
            "extra_view_images": None,
        }
        with torch.inference_mode():
            features = self.feature.extract_rlt_obs(env_obs)
        return {key: value.detach().float().cpu() for key, value in features.items()}

    def __call__(self, observation: dict) -> np.ndarray:
        """Return environment-normalized actions, not model-space actions."""
        import torch

        features = self.extract(observation)
        with torch.inference_mode():
            if self.actor is None:
                actions = features["ref_chunk"]
            else:
                actions, _ = self.actor.predict_action_batch(features, mode="eval")
        return actions[0].float().cpu().numpy()


def main() -> None:
    """Launch an explicit inference-only process on the selected visible GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--actor", type=Path, help="Stage2 full_weights.pt; omit for base"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("model.yaml")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get("RLT_VR_TOKEN", "")
    if len(token) < 32:
        parser.error("Set RLT_VR_TOKEN to a secret of at least 32 characters")
    if len(
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    ) != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly one available GPU")
    model = RLTInference(args.config, args.stage1, args.dataset, args.actor)
    model_id = f"stage1={args.stage1};actor={args.actor or 'base'}"
    with InferenceServer(("127.0.0.1", args.port), token, model, model_id) as server:
        logger.info(
            "Inference ready on loopback port %s; use SSH forwarding", args.port
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
