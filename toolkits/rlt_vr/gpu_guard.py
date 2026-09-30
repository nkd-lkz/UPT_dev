# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Select physical GPU 2 before importing CUDA users."""

import os
import subprocess
from contextlib import contextmanager


@contextmanager
def gpu2_lease():
    """Hold the shared project GPU-2 lock for the complete server lifetime."""
    import fcntl

    with open("/tmp/rlt-atomic-gpu2.lock", "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another RLT job owns the GPU-2 lease") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def isolate_gpu2(*, require_idle: bool = True) -> dict[str, str]:
    """Pin CUDA by UUID and Vulkan by PCI address; reject an occupied GPU."""
    row = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "2",
                "--query-gpu=uuid,pci.bus_id,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    uuid, bus, memory = (value.strip() for value in row)
    if require_idle and int(memory) > 1024:
        raise RuntimeError(f"Physical GPU 2 is busy ({memory} MiB); refusing launch")
    domain, bus_id, slot = bus.lower().split(":")
    render = f"pci:{int(domain, 16):04x}:{bus_id}:{slot}"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return {"uuid": uuid, "render_backend": render}


def verify_cuda(uuid: str) -> None:
    """Verify the one visible CUDA device before allocating model weights."""
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA device")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError(f"CUDA UUID mismatch: {actual} != {uuid}")
