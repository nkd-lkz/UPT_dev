# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wait for idle GPU 2, validate a smoke run, then run a bounded learning pilot."""

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rlinf.utils.logging import get_logger

LOGGER = get_logger()
ROOT = Path(__file__).resolve().parents[2]


def gpu_idle(memory: str, compute_pids: str) -> bool:
    """Accept a single readable GPU with at most 1 GiB and no compute processes."""
    return memory.strip().isdigit() and int(memory) <= 1024 and not compute_pids.strip()


def query_gpu() -> tuple[str, str]:
    """Read GPU 2 through NVML's CLI without creating a CUDA context."""
    outputs = []
    for query in ("--query-gpu=memory.used", "--query-compute-apps=pid"):
        outputs.append(
            subprocess.check_output(
                ["nvidia-smi", "-i", "2", query, "--format=csv,noheader,nounits"],
                text=True,
                timeout=15,
            ).strip()
        )
    return tuple(outputs)


def summarize_scalars(series: dict[str, list[tuple[int, float]]]) -> dict:
    """Require finite learning evidence; summarize evaluation separately."""
    if not series:
        raise ValueError("No scalar events were written.")
    invalid = [
        tag
        for tag, events in series.items()
        if any(not math.isfinite(v) for _, v in events)
    ]
    if invalid:
        raise ValueError(f"Non-finite metrics: {invalid}")
    totals = {}
    for component in ("actor", "critic"):
        suffix = f"rlt/{component}_updates_run"
        matches = [events for tag, events in series.items() if tag.endswith(suffix)]
        if len(matches) != 1 or sum(v for _, v in matches[0]) < 1:
            raise ValueError(f"Missing positive {component} learner updates.")
        totals[f"{component}_updates"] = sum(v for _, v in matches[0])
    atomic = [tag for tag in series if "atomic/reference_probability" in tag]
    if not atomic:
        raise ValueError("No atomic selector metrics; refusing to advance the queue.")
    return {
        **totals,
        "metrics": {
            tag: {
                "first": events[0][1],
                "last": events[-1][1],
                "min": min(v for _, v in events),
                "max": max(v for _, v in events),
                "count": len(events),
            }
            for tag, events in series.items()
            if events
        },
        "evaluations": [
            {"step": step, "success_once": value}
            for step, value in series.get("eval/success_once", [])
        ],
        "interpretation": "Integration evidence only; success must be compared on matched seeds.",
    }


def summarize_run(run_dir: Path, steps: int) -> dict:
    """Validate completed TensorBoard metrics and the final CPU-loadable weights."""
    import torch

    # Avoid loading TensorFlow, but restore module state for other RLinf utilities.
    blocked = "tensorflow" not in sys.modules
    if blocked:
        sys.modules["tensorflow"] = None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        events = EventAccumulator(
            str(run_dir / "tensorboard"), size_guidance={"scalars": 0}
        )
        events.Reload()
        series = {
            tag: [(e.step, e.value) for e in events.Scalars(tag)]
            for tag in events.Tags()["scalars"]
        }
    finally:
        if blocked:
            sys.modules.pop("tensorflow", None)
    report = summarize_scalars(series)
    checkpoint_suffix = Path(
        f"checkpoints/global_step_{steps}/actor/model_state_dict/full_weights.pt"
    )
    checkpoint_candidates = [run_dir / checkpoint_suffix]
    checkpoint_candidates.extend(run_dir.glob(f"*/{checkpoint_suffix}"))
    checkpoints = sorted(
        {path.resolve() for path in checkpoint_candidates if path.is_file()}
    )
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected exactly one step-{steps} checkpoint below {run_dir}, "
            f"found {len(checkpoints)}: {checkpoints}"
        )
    checkpoint = checkpoints[0]
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not any("selector.weight" in key for key in weights):
        raise ValueError("Final weights do not contain the atomic selector.")
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Non-finite final weights.")
    report["checkpoint"] = str(checkpoint)
    report["videos"] = [str(path) for path in sorted(run_dir.rglob("*.mp4"))]
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def wait_for_gpu(deadline: float, poll: int) -> None:
    """Require two idle observations; the launcher rechecks under its own lock."""
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            memory, pids = query_gpu()
            consecutive = consecutive + 1 if gpu_idle(memory, pids) else 0
            LOGGER.info(
                "GPU 2: used=%s MiB, compute_pids=%s, idle_checks=%s/2",
                memory,
                pids.replace("\n", ",") or "none",
                consecutive,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            consecutive = 0
            LOGGER.warning("GPU 2 status unavailable: %s", exc)
        if consecutive >= 2:
            return
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    raise TimeoutError("GPU 2 did not become available within the waiting budget.")


def run_phase(root: Path, profile: str, steps: int, timeout: int) -> dict:
    """Run one isolated child job and stop on failure, timeout or invalid metrics."""
    env = {
        **os.environ,
        "RLT_ATOMIC_PROFILE": profile,
        "RLT_ATOMIC_STEPS": str(steps),
        "RLT_ATOMIC_OUTPUT_DIR": str(root / profile),
    }
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=90s",
        str(timeout),
        "bash",
        str(ROOT / "run_rlt_atomic_gpu2.sh"),
        "--run",
    ]
    with (root / f"{profile}-launcher.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        LOGGER.info(
            "Started %s, supervisor PID=%s, log=%s", profile, process.pid, log.name
        )
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code != 0:
        raise RuntimeError(
            f"{profile} exited {code}; inspect {log.name}. No next job launched."
        )
    report = summarize_run(root / profile, steps)
    LOGGER.info(
        "%s passed: %s actor updates, %s",
        profile,
        report["actor_updates"],
        report["evaluations"],
    )
    return report


def main() -> None:
    """Execute only on explicit --run; default behavior checks both configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-phase", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--validated-smoke", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=21600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 30 <= args.wait_seconds <= 86400 or not 5 <= args.poll_seconds <= 60:
        parser.error("Wait must be 30..86400 seconds; poll must be 5..60 seconds.")
    if args.run and args.output is None:
        parser.error("--run requires a new --output directory.")
    if args.start_phase == "pilot" and args.validated_smoke is None:
        parser.error("--start-phase pilot requires --validated-smoke.")
    # The orchestrator's report reading is CPU-only. The child launcher sets GPU 2.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for profile in ("smoke", "pilot"):
        env = {**os.environ, "RLT_ATOMIC_PROFILE": profile}
        env.pop("RLT_ATOMIC_STEPS", None)
        subprocess.run(
            ["bash", str(ROOT / "run_rlt_atomic_gpu2.sh"), "--check"],
            env=env,
            check=True,
        )
    if not args.run:
        return
    with open(f"/tmp/rlt-atomic-queue-gpu2-{os.getuid()}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        status = {"git_revision": revision, "status": "waiting", "completed": {}}
        started = time.monotonic()
        waiting_used = 0.0

        def save_status() -> None:
            status["elapsed_seconds"] = time.monotonic() - started
            temporary = root / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(root / "status.json")

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
        try:
            phases = [("smoke", 2, 3600), ("pilot", 20, 14400)]
            if args.start_phase == "pilot":
                status["completed"]["smoke"] = summarize_run(
                    args.validated_smoke.resolve(), 2
                )
                phases = phases[1:]
            for profile, steps, timeout in phases:
                status.update(status="waiting", next_phase=profile)
                save_status()
                waiting_started = time.monotonic()
                wait_for_gpu(
                    waiting_started + args.wait_seconds - waiting_used,
                    args.poll_seconds,
                )
                waiting_used += time.monotonic() - waiting_started
                current = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip()
                dirty = subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
                if current != revision or dirty:
                    raise RuntimeError(
                        "Research checkout changed while waiting; refusing a mixed-code run."
                    )
                status.update(status="running", next_phase=profile)
                save_status()
                status["completed"][profile] = run_phase(root, profile, steps, timeout)
            status["status"] = "completed"
        except BaseException as exc:
            status.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save_status()


if __name__ == "__main__":
    main()
