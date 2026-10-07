# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wait for an unused GPU without stopping or modifying existing processes.

This does not reserve the device. The portable launcher must acquire its own
lease and recheck memory immediately before allocating CUDA resources.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def gpu_available(memory_mib: str, processes: str) -> bool:
    """Require one memory reading below 1 GiB and an empty compute-process list."""
    memory = memory_mib.strip()
    if not memory.isdecimal():
        raise ValueError("Expected one numeric GPU memory reading")
    return int(memory) <= 1024 and not processes.strip()


def main() -> None:
    """Wait until a device is free or the absolute Unix deadline is reached."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu < 0 or not time.time() < args.deadline <= time.time() + 24 * 3600:
        raise ValueError("Deadline must be within the next 24 hours")
    while time.time() < args.deadline:
        status = {
            "state": "waiting_for_gpu",
            "gpu": args.gpu,
            "checked_at": time.time(),
            "deadline": args.deadline,
        }
        try:

            def query(option):
                return subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(args.gpu),
                        option,
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=15,
                    stderr=subprocess.PIPE,
                )

            memory = query("--query-gpu=memory.used")
            processes = query("--query-compute-apps=pid")
            available = gpu_available(memory, processes)
            status.update(
                memory_mib=int(memory.strip()),
                compute_processes=len(processes.splitlines()),
            )
            if available:
                status["state"] = "available_not_reserved"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            available = False
            status["error"] = str(error)
        temp = args.status.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(args.status)
        if available:
            return
        time.sleep(min(60, max(0, args.deadline - time.time())))
    args.status.write_text(
        json.dumps(
            {"state": "wait_expired", "gpu": args.gpu, "deadline": args.deadline}
        )
        + "\n"
    )
    raise SystemExit(124)


if __name__ == "__main__":
    main()
