# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run a finite experiment plan serially after the selected GPU becomes idle."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import signal
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger(__name__)
LEASE_DIRECTORY = Path("/dev/shm")


def validate_plan(plan: dict) -> None:
    """Require explicit argv, dependencies on earlier jobs, and finite budgets."""
    if type(plan.get("gpu")) is not int or plan["gpu"] < 0:
        raise ValueError("Require an explicit nonnegative physical GPU index")
    if not math.isfinite(plan.get("hours", 0)) or not 0 < plan.get("hours", 0) <= 12:
        raise ValueError("Require a wall-clock budget in (0, 12] hours")
    known = set()
    for job in plan["jobs"]:
        name = job["name"]
        if not name or not all(c.isalnum() or c in "_-" for c in name) or name in known:
            raise ValueError("Require unique filesystem-safe job names")
        if not set(job.get("depends_on", [])).issubset(known):
            raise ValueError("Dependencies must refer to earlier jobs")
        if not job["argv"] or not all(isinstance(arg, str) for arg in job["argv"]):
            raise ValueError("Require an argv string list")
        if not 0 < job["timeout_seconds"] <= 43200:
            raise ValueError("Require a finite per-job timeout")
        if not Path(job["cwd"]).is_dir():
            raise ValueError(f"Working directory does not exist: {job['cwd']}")
        known.add(name)


def gpu_idle(gpu: int) -> tuple[bool, str]:
    """Require low allocated memory AND no compute processes; fail closed on errors."""
    try:
        query = ["nvidia-smi", "-i", str(gpu)]
        memory = subprocess.check_output(
            query + ["--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        pids = subprocess.check_output(
            query + ["--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        return int(
            memory
        ) <= 512 and not pids, f"memory={memory} MiB pids={pids or 'none'}"
    except (ValueError, subprocess.SubprocessError, OSError) as error:
        return False, f"GPU query failed: {type(error).__name__}"


def completed_manifest(job: dict) -> bool:
    """Check semantic completion in addition to the command's exit status."""
    manifest = job.get("manifest")
    if manifest is None:
        return True
    try:
        data = json.loads(Path(manifest).read_text())
        return data.get("complete") is True and data.get("exit_code", 0) == 0
    except (OSError, ValueError):
        return False


def stop_group(process: subprocess.Popen) -> None:
    """Signal only the fresh process group owned by this queue job."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        pass
    # Workers may outlive their launcher. This group was created with setsid and
    # never contains a pre-existing GPU task or another queue job.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_plan(plan: dict, output: Path) -> dict:
    """Wait for two idle observations, run each job, and retain failure evidence."""
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    deadline = time.monotonic() + plan["hours"] * 3600
    state = {
        "state": "running",
        "started_at": time.time(),
        "gpu_index": plan["gpu"],
        "jobs": {job["name"]: {"state": "pending"} for job in plan["jobs"]},
    }

    def save():
        state["updated_at"] = time.time()
        temp = output / "status.tmp.json"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(output / "status.json")

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Queue received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    lock = (LEASE_DIRECTORY / f"rlt-research-gpu{plan['gpu']}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        raise RuntimeError(
            f"An experiment queue already owns the GPU {plan['gpu']} lease"
        ) from error
    process = None
    try:
        save()
        for job in plan["jobs"]:
            row = state["jobs"][job["name"]]
            if any(
                state["jobs"][dep]["state"] != "passed"
                for dep in job.get("depends_on", [])
            ):
                row["state"] = "skipped_dependency"
                save()
                continue
            idle_checks = 0
            while job.get("requires_gpu", True) and time.monotonic() < deadline:
                idle, detail = gpu_idle(plan["gpu"])
                idle_checks = idle_checks + 1 if idle else 0
                row.update(state="waiting_gpu", gpu=detail, idle_checks=idle_checks)
                save()
                LOG.info("%s: %s idle=%s/2", job["name"], detail, idle_checks)
                if idle_checks >= 2:
                    break
                time.sleep(min(30, max(0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                row["state"] = "not_started_deadline"
                state["state"] = "deadline"
                break
            row.update(state="running", started_at=time.time())
            save()
            LOG.info("Starting %s", job["name"])
            env = {
                **os.environ,
                "TMPDIR": "/dev/shm",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TORCH_EXTENSIONS_DIR": f"/dev/shm/rlt-night-cache/{job['name']}/torch",
                "TRITON_CACHE_DIR": f"/dev/shm/rlt-night-cache/{job['name']}/triton",
                "MPLCONFIGDIR": f"/dev/shm/rlt-night-cache/{job['name']}/matplotlib",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                **job.get("env", {}),
            }
            with (output / f"{job['name']}.log").open("x") as log:
                process = subprocess.Popen(
                    job["argv"],
                    cwd=job["cwd"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                row["pid"] = process.pid
                save()
                try:
                    code = process.wait(
                        timeout=max(
                            0.01,
                            min(job["timeout_seconds"], deadline - time.monotonic()),
                        )
                    )
                    row.update(
                        exit_code=code,
                        state="passed"
                        if code == 0 and completed_manifest(job)
                        else "failed",
                    )
                except subprocess.TimeoutExpired:
                    row.update(state="timeout", exit_code=None)
                finally:
                    stop_group(process)
                    process = None
            row["finished_at"] = time.time()
            LOG.info("Finished %s: %s", job["name"], row["state"])
            save()
        if state["state"] == "running":
            state["state"] = "finished"
    except KeyboardInterrupt:
        state["state"] = "interrupted"
        raise
    except Exception as error:
        state.update(state="error", error=str(error))
        raise
    finally:
        if process is not None:
            stop_group(process)
        for row in state["jobs"].values():
            if row["state"] == "pending":
                row["state"] = f"not_started_{state['state']}"
            elif row["state"] in {"running", "waiting_gpu"}:
                row["state"] = state["state"]
        save()
        lock.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return state


def main() -> None:
    """Run a reviewed plan on one explicitly authorized physical GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.check:
        print(
            f"Validated {len(plan['jobs'])} jobs on GPU {plan['gpu']}, wall limit {plan['hours']} hours"
        )
        return
    run_plan(plan, args.output)


if __name__ == "__main__":
    main()
