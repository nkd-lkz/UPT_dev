# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Keep LIBERO text diagnostics out of AlphaBrain's binary stdout protocol."""

import os
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path


class ProtocolOutput:
    """Send text to stderr while retaining the original binary response pipe."""

    def __init__(self, binary, text):
        self.buffer = binary
        self.text = text

    def write(self, value: str) -> int:
        """Route library print calls to diagnostics."""
        return self.text.write(value)

    def flush(self) -> None:
        """Flush both independently owned output streams."""
        self.text.flush()
        self.buffer.flush()

    def __getattr__(self, name):
        return getattr(self.text, name)


def run_worker(worker: Path, *, socket_protocol: bool = False) -> None:
    """Run an unchanged worker after separating binary and text stdout."""
    if not worker.is_file():
        raise FileNotFoundError(worker)
    # CPU MuJoCo uses EGL only. Hide CUDA so robosuite does not compare an EGL
    # index to the parent's distinct CUDA UUID/index namespace.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["RLT_LIBERO_EGL_DEVICE_ID"]
    log_dir = os.environ.get("RLT_LIBERO_WORKER_LOG_DIR")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    diagnostics = (
        (Path(log_dir) / f"worker-{os.getpid()}.log").open("x", buffering=1)
        if log_dir
        else nullcontext(sys.stderr)
    )
    with (
        os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as protocol,
        diagnostics as log,
    ):
        # The upstream parent does not drain stderr during normal operation.
        # Write production diagnostics to a per-worker file, not a bounded pipe.
        if log_dir:
            os.dup2(log.fileno(), sys.stderr.fileno())
        # C extensions can also print to fd 1; preserve their messages on stderr.
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = (
            sys.stderr if socket_protocol else ProtocolOutput(protocol, sys.stderr)
        )
        try:
            runpy.run_path(str(worker), run_name="__main__")
        finally:
            sys.stdout.flush()
            sys.stdout = sys.stderr


if __name__ == "__main__":
    source = Path(os.environ["RLT_ALPHABRAIN_SOURCE"])
    fast = len(sys.argv) == 2
    run_worker(
        source
        / "AlphaBrain/training/reinforcement_learning/envs"
        / ("libero_env_worker_fast.py" if fast else "libero_env_worker.py"),
        socket_protocol=fast,
    )
