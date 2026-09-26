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

"""Bounded JSON RPC for one local simulator and one inference server.

Run over an SSH tunnel. The token authenticates requests but does not encrypt
images. No remote Python objects or checkpoint paths are deserialized.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import struct
from typing import Any

import numpy as np
from PIL import Image

VERSION = 1
MAX_MESSAGE = 4 * 1024 * 1024
CAMERAS = ("main_image", "wrist_image")
ENV_ID = "PegInsertionSideWideClearanceObserverWideWrist-v1"
CONTRACT = {
    "version": VERSION,
    "env_id": ENV_ID,
    "control_mode": "pd_joint_delta_pos",
    "control_hz": 10,
    "state_dim": 9,
    "action_dim": 8,
    "image_size": 384,
}


def _read_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise ConnectionError("Peer closed an incomplete message")
        result.extend(part)
    return bytes(result)


def receive(sock: socket.socket) -> dict[str, Any]:
    """Read a length-prefixed object; reject oversized frames before allocation."""
    size = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Message size exceeds protocol limit")
    result = json.loads(_read_exact(sock, size))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def send(sock: socket.socket, message: dict[str, Any]) -> None:
    """Send an object without allowing non-finite JSON numbers."""
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE:
        raise ValueError("Message size exceeds protocol limit")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def encode_image(image: np.ndarray) -> str:
    """Encode lossless model input without JPEG distribution shift."""
    if image.shape != (384, 384, 3) or image.dtype != np.uint8:
        raise ValueError("Expected a 384x384 RGB uint8 image")
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG", compress_level=1)
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_image(value: str) -> np.ndarray:
    """Validate image dimensions before decompression."""
    data = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (384, 384) or image.mode != "RGB":
            raise ValueError("Expected a 384x384 RGB PNG")
        return np.array(image)


def validate_actions(value: Any) -> np.ndarray:
    """Validate one normalized Panda action chunk; never silently clip policy."""
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= len(actions) <= 10:
        raise ValueError("Expected action chunk with shape [1..10, 8]")
    if not np.isfinite(actions).all() or (np.abs(actions) > 1.00001).any():
        raise ValueError("Actions must be finite and normalized to [-1, 1]")
    return actions


def decode_observation(message: dict[str, Any]) -> dict[str, Any]:
    """Validate environment semantics and reconstruct a single observation."""
    if message.get("contract") != CONTRACT:
        raise ValueError("Client/server environment contract mismatch")
    state = np.asarray(message["state"], dtype=np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("Expected nine finite Panda joint positions")
    return {"state": state, **{k: decode_image(message[k]) for k in CAMERAS}}


def request(host: str, port: int, token: str, payload: dict, timeout: float) -> dict:
    """Execute one RPC with a bounded connection lifetime."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send(sock, {**payload, "token": token})
        response = receive(sock)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Inference request failed"))
    if response.get("request_id") != payload.get("request_id"):
        raise ValueError("Mismatched inference response")
    return response
