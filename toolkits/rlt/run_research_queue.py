# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resume bounded research jobs after GPU availability and evidence validation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from toolkits.rlt.wait_for_gpu import gpu_available

TERMINAL = {
    "completed",
    "failed",
    "dependency_failed",
    "wait_expired",
    "budget_expired",
}


def write_json(path: Path, value: dict) -> None:
    """Atomically publish one owned status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path: Path) -> dict:
    """Read an optional atomic status file."""
    return json.loads(path.read_text()) if path.exists() else {}


def job_states(campaign: Path, jobs: list[dict]) -> dict[str, dict]:
    """Read prerequisite campaigns without taking ownership of their jobs."""
    return {
        job["id"]: read_json(
            Path(job.get("evidence_campaign", campaign)) / "jobs" / f"{job['id']}.json"
        )
        for job in jobs
    }


def gpu_status(gpu: int) -> dict:
    """Fail closed on unreadable memory, active compute processes or query errors."""

    def query(option):
        return subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), option, "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=15,
        )

    try:
        memory = query("--query-gpu=memory.used")
        processes = query("--query-compute-apps=pid")
        return {
            "available": gpu_available(memory, processes),
            "memory_mib": int(memory.strip()),
            "compute_process_count": len(processes.splitlines()),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}


def validate_evaluation(root: Path, *, seed: int, reference: bool) -> dict:
    """Reject incomplete, duplicated, changing-weight or wrongly routed episodes."""
    paths = list(root.glob("stage2_*/episode-records.json")) + list(
        root.glob("stage2_*/stage2_portable/episode-records.json")
    )
    if len(paths) != 1:
        raise ValueError("Expected exactly one completed evaluation run")
    path = paths[0]
    code = root / path.relative_to(root).parts[0] / "exit_code.txt"
    if not code.exists() or code.read_text().strip() != "0":
        raise ValueError("Evaluation launcher did not finish successfully")
    audit = read_json(path.parent / "route-audit.json")
    episodes = json.loads(path.read_text())
    if (
        not audit.get("weights_unchanged")
        or len(audit.get("initial_observation_sha256", "")) != 64
    ):
        raise ValueError("Missing frozen-weight or initial-state audit")
    if len(episodes) != 16 or {e["lane"] for e in episodes} != set(range(16)):
        raise ValueError("Expected sixteen distinct completed lanes")
    if any(e["seed"] != seed or e["success_once"] not in (0, 1) for e in episodes):
        raise ValueError("Unexpected evaluation seed or outcome")
    if reference and audit.get("actor_slots") != 0:
        raise ValueError("Reference comparator executed actor commands")
    return {
        "seed": seed,
        "successes": int(sum(e["success_once"] for e in episodes)),
        "episodes": episodes,
        "audit": audit,
    }


def validate_job(job: dict) -> dict:
    """Check the declared scientific artifact before marking a job complete."""
    root = Path(job["output"])
    if job["kind"] == "evaluation":
        return validate_evaluation(
            root, seed=job["seed"], reference=job["arm"] == "reference"
        )
    result = read_json(root / "results.json")
    if not result.get("completed"):
        raise ValueError("Diagnostic has no complete report")
    if job["kind"] == "matched" and (
        result.get("pairs") != job.get("pairs", 56)
        or not result.get("intervention_detected")
    ):
        raise ValueError(
            "Matched-state probe incomplete or drive intervention ineffective"
        )
    if job["kind"] == "shift" and len(result.get("rows", [])) != job.get("streams", 12):
        raise ValueError("Expected six paired A-B-A/stationary streams")
    if job["kind"] == "phase" and (
        len(result.get("rows", [])) != job["streams"]
        or not result.get("all_contact_stages_valid")
    ):
        raise ValueError(
            "Incomplete contact-stage factorial or invalid contact preparation"
        )
    if job["kind"] == "control":
        rows = result.get("rows", [])
        keys = {(r["seed"], r["stage"], r["dynamics"], r["method"]) for r in rows}
        if (
            len(rows) != job["streams"]
            or len(keys) != len(rows)
            or not result.get("all_initial_states_matched")
            or any(r["control_ticks"] != 780 for r in rows)
            or result.get("control_ticks") != 780 * len(rows)
            or result.get("invalid_contact_streams")
            != sum(not r["contact_valid"] for r in rows)
        ):
            raise ValueError("Incomplete or unmatched closed-loop tracking evidence")
    return {"scope": result["scope"], "results_path": str(root / "results.json")}


def baseline_summary(jobs: list[dict], states: dict[str, dict]) -> dict:
    """Require all four arms and paired initial states; do not infer significance."""
    baseline = [j for j in jobs if j["kind"] == "evaluation"]
    if len(baseline) != 16 or any(
        states.get(j["id"], {}).get("state") != "completed" for j in baseline
    ):
        raise ValueError("All sixteen baseline runs must be validated first")
    arms = {}
    fingerprints = {}
    for job in baseline:
        result = validate_job(job)
        seed = job["seed"]
        fingerprint = result["audit"]["initial_observation_sha256"]
        if seed in fingerprints and fingerprints[seed] != fingerprint:
            raise ValueError("Paired initial observations differ across arms")
        fingerprints[seed] = fingerprint
        arms.setdefault(job["arm"], []).append(result)
    if set(arms) != {"bc_only", "q_bc", "reference", "old_zero"}:
        raise ValueError("Missing baseline arm")
    summary = {}
    for name, runs in arms.items():
        if sorted(r["seed"] for r in runs) != [4101, 4102, 4103, 4104]:
            raise ValueError("Duplicate or missing evaluation seed")
        if name != "reference" and not sum(r["audit"]["actor_slots"] for r in runs):
            raise ValueError("No learned actor was executed")
        summary[name] = {
            "successes": sum(r["successes"] for r in runs),
            "episodes": 64,
            "gate_entered": int(
                sum(e["entered_actor_phase_once"] for r in runs for e in r["episodes"])
            ),
            "runs": runs,
        }
    return {
        "arms": summary,
        "conclusion": "Descriptive single-training-seed evaluation; old_zero has a different training budget",
        "next_action": "Run bounded response diagnostics; do not automatically launch RL training",
    }


def stop_child(process: subprocess.Popen) -> None:
    """Terminate only this runner's newly created process group."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def execute_job(
    job: dict, campaign: Path, *, timeout: float, heartbeat
) -> tuple[int, float]:
    """Execute an argv, keeping a durable log and a bounded child lifetime."""
    for item in job.get("inputs", []):
        if (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("Pinned input changed")
    env = os.environ.copy()
    for key in (
        "RLT_SMOKE_RESUME_DIR",
        "RLT_SMOKE_RAY_PORT",
        "CUDA_VISIBLE_DEVICES",
        "RAY_ADDRESS",
    ):
        env.pop(key, None)
    env.update(job.get("env", {}))
    started = time.time()
    with (campaign / "logs" / f"{job['id']}.log").open("a") as log:
        process = subprocess.Popen(
            job["command"],
            cwd=job["cwd"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                elapsed = time.time() - started
                heartbeat(process.pid, elapsed)
                if elapsed >= timeout:
                    stop_child(process)
                    return 124, time.time() - started
                time.sleep(min(10, max(0.05, timeout - elapsed)))
            return process.returncode, time.time() - started
        finally:
            stop_child(process)


def run_queue(campaign: Path, gpu: int) -> None:
    """Resume completed jobs; wait for dependencies and GPUs up to a fixed deadline."""
    manifest = read_json(campaign / "manifest.json")
    jobs = manifest["jobs"]
    if gpu not in (0, 1) or not jobs:
        raise ValueError("Expected GPU 0/1 and a nonempty manifest")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job identifiers")
    owned_jobs = [
        job for job in jobs if job["gpu"] == gpu and "evidence_campaign" not in job
    ]
    (campaign / "logs").mkdir(exist_ok=True)
    queue_path = campaign / f"gpu{gpu}.json"
    lease = (campaign / f"gpu{gpu}.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = read_json(queue_path)
    used = float(previous.get("run_seconds", 0))
    deadline = manifest["wait_deadline"]
    limit = manifest["run_seconds_per_gpu"]
    queue = {
        "gpu": gpu,
        "pid": os.getpid(),
        "run_seconds": used,
        "wait_deadline": deadline,
    }

    def update(state, **kwargs):
        queue.update(state=state, checked_at=time.time(), **kwargs)
        write_json(queue_path, queue)

    try:
        for job in owned_jobs:
            path = campaign / "jobs" / f"{job['id']}.json"
            prior = read_json(path)
            if prior.get("state") == "completed":
                validate_job(job)
                continue
            if prior.get("state") in TERMINAL or prior.get("state") == "running":
                raise RuntimeError(
                    f"Review prior job {job['id']} before retrying: {prior.get('state')}"
                )
            update("waiting", job=job["id"])
            while True:
                states = job_states(campaign, jobs)
                dependencies = [
                    states[d].get("state", "pending")
                    for d in job.get("dependencies", [])
                ]
                if any(d in TERMINAL - {"completed"} for d in dependencies):
                    write_json(path, {"state": "dependency_failed", "job": job["id"]})
                    break
                if time.time() >= deadline or used >= limit:
                    state = (
                        "wait_expired" if time.time() >= deadline else "budget_expired"
                    )
                    write_json(path, {"state": state, "job": job["id"]})
                    break
                if all(d == "completed" for d in dependencies):
                    if job.get("require_paired_baseline"):
                        write_json(
                            campaign / "baseline-summary.json",
                            baseline_summary(jobs, states),
                        )
                    availability = gpu_status(gpu)
                    update("waiting_for_gpu", availability=availability)
                    if availability["available"]:
                        break
                else:
                    update("waiting_for_dependencies", dependencies=dependencies)
                time.sleep(min(60, max(0, deadline - time.time())))
            if read_json(path).get("state") in TERMINAL:
                continue
            write_json(
                path, {"state": "running", "job": job["id"], "started_at": time.time()}
            )

            def heartbeat(pid, elapsed):
                update(
                    "running",
                    child_pid=pid,
                    job_elapsed=elapsed,
                    run_seconds=used + elapsed,
                )

            code, elapsed = execute_job(
                job,
                campaign,
                timeout=min(job["timeout_seconds"], limit - used),
                heartbeat=heartbeat,
            )
            used += elapsed
            queue["run_seconds"] = used
            if code != 0:
                write_json(
                    path,
                    {
                        "state": "failed",
                        "job": job["id"],
                        "exit_code": code,
                        "elapsed_seconds": elapsed,
                    },
                )
                raise RuntimeError(f"Job {job['id']} failed with exit code {code}")
            evidence = validate_job(job)
            write_json(
                path,
                {
                    "state": "completed",
                    "job": job["id"],
                    "exit_code": code,
                    "elapsed_seconds": elapsed,
                    "evidence": evidence,
                },
            )
        update("finished", run_seconds=used)
    except BaseException as error:
        if "path" in locals() and read_json(path).get("state") == "running":
            write_json(
                path, {"state": "failed", "job": job["id"], "error": repr(error)}
            )
        # Mark pending owned jobs so another GPU cannot wait forever for them.
        for job in owned_jobs:
            path = campaign / "jobs" / f"{job['id']}.json"
            if read_json(path).get("state") not in TERMINAL:
                write_json(path, {"state": "dependency_failed", "error": repr(error)})
        update("failed", error=repr(error))
        raise
    finally:
        lease.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    run_queue(args.campaign, args.gpu)


if __name__ == "__main__":
    main()
