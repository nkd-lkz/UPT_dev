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
    if type(plan.get("stop_on_failure", False)) is not bool:
        raise ValueError("stop_on_failure must be a boolean")
    revision = plan.get("revision")
    if revision is not None and (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("revision must be a full lowercase Git commit")
    if type(plan.get("require_clean_worktree", False)) is not bool:
        raise ValueError("require_clean_worktree must be a boolean")
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


def repository_state(cwd: str) -> tuple[str, bool]:
    """Return the checked-out revision and whether tracked or untracked files differ."""
    revision = subprocess.check_output(
        ["git", "-C", cwd, "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", cwd, "status", "--porcelain"], text=True, timeout=10
        ).strip()
    )
    return revision, dirty


def revision_error(plan: dict, job: dict) -> str | None:
    """Reject a queued job if its reviewed source contract has changed."""
    expected = plan.get("revision")
    if expected is None:
        return None
    try:
        revision, dirty = repository_state(job["cwd"])
    except (subprocess.SubprocessError, OSError) as error:
        return f"source audit failed: {type(error).__name__}"
    if revision != expected:
        return f"revision changed: expected {expected}, found {revision}"
    if plan.get("require_clean_worktree", False) and dirty:
        return "worktree changed after plan review"
    return None


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


def manifest_error(job: dict) -> str | None:
    """Read an explicit failure even when a child is stuck shutting down."""
    if not job.get("manifest"):
        return None
    try:
        data = json.loads(Path(job["manifest"]).read_text())
        if data.get("complete") is False and data.get("error"):
            return str(data["error"])
    except (OSError, ValueError):
        pass
    return None


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
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state["state"] = "running"
                break
            except BlockingIOError:
                if not plan.get("wait_for_lease", False):
                    raise RuntimeError(
                        f"An experiment queue already owns the GPU {plan['gpu']} lease"
                    ) from None
                state["state"] = "waiting_lease"
                save()
                if time.monotonic() >= deadline:
                    state["state"] = "deadline"
                    for row in state["jobs"].values():
                        row["state"] = "not_started_deadline"
                    save()
                    lock.close()
                    signal.signal(signal.SIGTERM, previous_sigterm)
                    return state
                time.sleep(min(30, max(0, deadline - time.monotonic())))
    except BaseException:
        lock.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        raise
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
            source_error = revision_error(plan, job)
            if source_error is not None:
                row.update(state="revision_mismatch", error=source_error)
                state.update(state="halted_revision_mismatch", failed_job=job["name"])
                save()
                break
            idle_checks = 0
            while job.get("requires_gpu", True) and time.monotonic() < deadline:
                source_error = revision_error(plan, job)
                if source_error is not None:
                    break
                idle, detail = gpu_idle(plan["gpu"])
                idle_checks = idle_checks + 1 if idle else 0
                row.update(state="waiting_gpu", gpu=detail, idle_checks=idle_checks)
                save()
                LOG.info("%s: %s idle=%s/2", job["name"], detail, idle_checks)
                if idle_checks >= 2:
                    break
                time.sleep(min(30, max(0, deadline - time.monotonic())))
            if source_error is not None:
                row.update(state="revision_mismatch", error=source_error)
                state.update(state="halted_revision_mismatch", failed_job=job["name"])
                save()
                break
            if time.monotonic() >= deadline:
                row["state"] = "not_started_deadline"
                state["state"] = "deadline"
                break
            source_error = revision_error(plan, job)
            if source_error is not None:
                row.update(state="revision_mismatch", error=source_error)
                state.update(state="halted_revision_mismatch", failed_job=job["name"])
                save()
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
                    job_deadline = min(
                        deadline, time.monotonic() + job["timeout_seconds"]
                    )
                    while True:
                        remaining = job_deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(job["argv"], 0)
                        try:
                            code = process.wait(timeout=min(2, remaining))
                        except subprocess.TimeoutExpired:
                            error = manifest_error(job)
                            if error is not None:
                                row.update(state="failed", exit_code=None, error=error)
                                break
                            continue
                        row.update(
                            exit_code=code,
                            state="passed"
                            if code == 0 and completed_manifest(job)
                            else "failed",
                        )
                        break
                except subprocess.TimeoutExpired:
                    row.update(state="timeout", exit_code=None)
                finally:
                    stop_group(process)
                    process = None
            row["finished_at"] = time.time()
            LOG.info("Finished %s: %s", job["name"], row["state"])
            save()
            if plan.get("stop_on_failure", False) and row["state"] != "passed":
                state.update(state="halted_failure", failed_job=job["name"])
                break
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
