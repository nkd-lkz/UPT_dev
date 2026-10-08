# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run a finite, gated two-GPU LIBERO acceptance campaign in tmux."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from toolkits.rlt.libero_acceptance import PROTOCOL, digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def matched(reference: list[dict], learned: list[dict]) -> dict:
    """Compare paired complete episodes, including initial observation identity."""

    def keyed(rows):
        result = {}
        for row in rows:
            if (
                row["case_id"] in result
                or type(row["success"]) is not bool
                or row["budget_cut"]
            ):
                raise ValueError("Invalid, duplicate or incomplete evaluation")
            result[row["case_id"]] = row
        return result

    a, b = keyed(reference), keyed(learned)
    if not a or a.keys() != b.keys():
        raise ValueError("Evaluation case identities differ")
    for case in a:
        for key in ("initial_observation", "state_sha256"):
            if a[case][key] != b[case][key]:
                raise ValueError(f"Initial state/render mismatch: {case}, {key}")
    gains = sum(b[k]["success"] and not a[k]["success"] for k in a)
    losses = sum(a[k]["success"] and not b[k]["success"] for k in a)
    n = gains + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)
    return {
        "episodes": len(a),
        "reference_successes": sum(v["success"] for v in a.values()),
        "learner_successes": sum(v["success"] for v in b.values()),
        "gains": gains,
        "losses": losses,
        "paired_difference": (gains - losses) / len(a),
        "exact_mcnemar_two_sided": p,
    }


def warmup_gate(fit: dict, comparison: dict) -> dict:
    """Apply preregistered development thresholds, not a significance claim."""
    a, b = fit["initial_following"], fit["final_following"]
    finite = all(isinstance(v, (float, int)) and math.isfinite(v) for v in b.values())
    passed = (
        finite
        and b["mse"] < min(0.02, a["mse"] * 0.5)
        and b["gripper_disagreement"] <= 0.05
        and comparison["reference_successes"] >= comparison["episodes"] / 2
        and comparison["learner_successes"] >= comparison["reference_successes"] - 2
    )
    return {
        "passed": bool(passed),
        "following": b,
        "rollout": comparison,
        "criterion": "MSE < min(.02, half initial); gripper mismatch <=5%; reference >=50%; BC loses <=2/20 cases",
    }


def make_cases(output: Path) -> dict[str, Path]:
    """Keep training, BC holdout, development gate and fresh validation separate."""

    def published(indices):
        return [
            {
                "id": f"published_{i:02d}",
                "kind": "published",
                "task": 0,
                "state": i,
                "seed": 42,
            }
            for i in indices
        ]

    cases = {
        "collection": published(range(30)),
        "training": published(range(24)),
        "development": published(range(30, 50)),
        "validation": [
            {
                "id": f"generated_{seed}",
                "kind": "generated",
                "task": 0,
                "seed": seed,
                "snapshot": str(output / "initial_states" / f"{seed}.json"),
            }
            for seed in range(20000, 20050)
        ],
    }
    paths = {}
    for name, rows in cases.items():
        paths[name] = output / f"cases_{name}.json"
        write_json(paths[name], rows)
    return paths


class Campaign:
    """Own only launched process groups; persist waiting, failures and gates."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True)
        self.cases = make_cases(self.output)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.deadline = time.monotonic() + args.max_hours * 3600
        self.status = {
            "state": "running",
            "protocol": PROTOCOL,
            "jobs": {},
            "gates": {},
        }
        write_json(self.output / "status.json", self.status)
        write_json(
            self.output / "plan.json",
            {
                "protocol": PROTOCOL,
                "gpus": args.gpus,
                "seeds": [42, 43, 44],
                "control_budget_per_arm": args.control_budget,
                "bc_updates": args.bc_updates,
                "cases": {
                    k: {"path": str(p), "sha256": digest(p)}
                    for k, p in self.cases.items()
                },
                "expansion_rule": "All warmup gates pass; seed42 Q+BC exceeds BC and reaches reference on development cases",
                "validation_rule": "Fresh generated states; evaluated only after development decision; never used for updates",
                "scope": "Task0 adaptation; not the official 10-task benchmark or full-token RLT",
            },
        )

    def publish(self, name: str, value: dict) -> None:
        """Update a single job atomically while two workers may be active."""
        with self.lock:
            self.status["jobs"][name] = value
            write_json(self.output / "status.json", self.status)

    def run(self, name: str, mode: str, gpu: int, **options) -> Path:
        """Wait for the assigned GPU, then run a bounded child process group."""
        directory = self.output / name
        argv = [
            "bash",
            str(ROOT / "run_rlt_libero.sh"),
            "acceptance",
            mode,
            "--gpu",
            str(gpu),
            "--output",
            str(directory),
            "--wandb",
            self.args.wandb,
            "--group",
            self.output.name,
        ]
        for key, value in options.items():
            if value is False or value is None:
                continue
            argv.append("--" + key.replace("_", "-"))
            if value is not True:
                argv.append(str(value))
        record = {
            "state": "waiting_gpu",
            "gpu": gpu,
            "argv": argv,
            "output": str(directory),
        }
        self.publish(name, record)
        # This advisory lock coordinates our jobs, never kills outside processes.
        with open(
            f"/dev/shm/rlt-libero-acceptance-{os.getuid()}-gpu{gpu}.lock", "a"
        ) as lease:
            deadline = min(
                self.deadline, time.monotonic() + self.args.wait_hours * 3600
            )
            idle = 0
            while True:
                if self.stop.is_set():
                    raise RuntimeError("Campaign cancelled after sibling failure")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"GPU {gpu} did not become idle")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.stop.wait(30)
                    continue
                used = int(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            str(gpu),
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).strip()
                )
                idle = idle + 1 if used <= 512 else 0
                if idle >= 2:
                    break
                record.update(used_mib=used, idle_checks=idle)
                self.publish(name, record.copy())
                self.stop.wait(30)
            record.update(state="running", start_time=time.time())
            self.publish(name, record.copy())
            with (self.output / f"{name}.log").open("x") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record["pid"] = process.pid
                self.publish(name, record.copy())
                limit = min(
                    self.deadline, time.monotonic() + self.args.job_hours * 3600
                )
                try:
                    while process.poll() is None:
                        if self.stop.wait(5) or time.monotonic() > limit:
                            raise TimeoutError(f"Cancelled or timed out: {name}")
                    if process.returncode:
                        raise RuntimeError(
                            f"{name} exited {process.returncode}; inspect {name}.log"
                        )
                    manifest = json.loads((directory / "manifest.json").read_text())
                    if not manifest["complete"] or manifest["protocol"] != PROTOCOL:
                        raise ValueError(f"{name} did not complete this protocol")
                except BaseException as error:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    record.update(
                        state="failed", error=str(error), end_time=time.time()
                    )
                    self.publish(name, record.copy())
                    raise
            record.update(state="complete", end_time=time.time(), exit_code=0)
            self.publish(name, record)
        return directory

    def parallel(self, *calls) -> list:
        """Wait for both outcomes; cancel only our sibling on execution failure."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(call) for call in calls]
            try:
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    results[future] = future.result()
                return [results[future] for future in futures]
            except BaseException:
                self.stop.set()
                raise

    def episodes(self, name: str) -> list[dict]:
        """Read completed episode evidence for an explicit job."""
        return json.loads((self.output / name / "episodes.json").read_text())

    def execute(self) -> None:
        """Collect once, accept BC, compare seed42, then conditionally replicate."""
        g0, g1 = self.args.gpus
        self.parallel(
            lambda: self.run("collect", "collect", g0, cases=self.cases["collection"]),
            lambda: self.run(
                "state_bank",
                "generate",
                g1,
                cases=self.cases["validation"],
                video_episodes=0,
            ),
        )
        self.parallel(
            lambda: self.run(
                "fit42",
                "fit",
                g0,
                cache=self.output / "collect/cache.pt",
                seed=42,
                bc_updates=self.args.bc_updates,
            ),
            lambda: self.run(
                "reference_dev",
                "evaluate",
                g1,
                cases=self.cases["development"],
                reference=True,
            ),
        )
        self.run(
            "warm42_dev",
            "evaluate",
            g0,
            cases=self.cases["development"],
            checkpoint=self.output / "fit42/checkpoint.pt",
        )
        fit = json.loads((self.output / "fit42/summary.json").read_text())
        gate = warmup_gate(
            fit, matched(self.episodes("reference_dev"), self.episodes("warm42_dev"))
        )
        self.status["gates"]["warm42"] = gate
        write_json(self.output / "status.json", self.status)
        if not gate["passed"]:
            self.status["state"] = "stopped_bc_gate"
            write_json(
                self.output / "results.json",
                {
                    "state": "stopped_bc_gate",
                    "gate": gate,
                    "online_training_started": False,
                    "next_action": "Diagnose BC following and actor-control loss before RL",
                },
            )
            return
        passed_seeds = []
        for seed in (42, 43, 44):
            if seed != 42:
                self.run(
                    f"fit{seed}",
                    "fit",
                    g0,
                    cache=self.output / "collect/cache.pt",
                    seed=seed,
                    bc_updates=self.args.bc_updates,
                )
                self.run(
                    f"warm{seed}_dev",
                    "evaluate",
                    g0,
                    cases=self.cases["development"],
                    checkpoint=self.output / f"fit{seed}/checkpoint.pt",
                )
                fit = json.loads((self.output / f"fit{seed}/summary.json").read_text())
                gate = warmup_gate(
                    fit,
                    matched(
                        self.episodes("reference_dev"), self.episodes(f"warm{seed}_dev")
                    ),
                )
                self.status["gates"][f"warm{seed}"] = gate
                if not gate["passed"]:
                    break
            warm = self.output / f"fit{seed}/checkpoint.pt"
            self.parallel(
                *[
                    lambda arm=arm, gpu=gpu: self.run(
                        f"{arm}{seed}",
                        "train",
                        gpu,
                        cases=self.cases["training"],
                        checkpoint=warm,
                        objective=arm,
                        seed=seed,
                        control_budget=self.args.control_budget,
                    )
                    for arm, gpu in (("bc_only", g0), ("q_bc", g1))
                ]
            )
            self.parallel(
                *[
                    lambda arm=arm, gpu=gpu: self.run(
                        f"{arm}{seed}_dev",
                        "evaluate",
                        gpu,
                        cases=self.cases["development"],
                        checkpoint=self.output / f"{arm}{seed}/checkpoint.pt",
                        objective=arm,
                    )
                    for arm, gpu in (("bc_only", g0), ("q_bc", g1))
                ]
            )
            paired = matched(
                self.episodes(f"bc_only{seed}_dev"), self.episodes(f"q_bc{seed}_dev")
            )
            reaches_ref = sum(
                r["success"] for r in self.episodes(f"q_bc{seed}_dev")
            ) >= sum(r["success"] for r in self.episodes("reference_dev"))
            expand = paired["paired_difference"] > 0 and reaches_ref
            self.status["gates"][f"rl{seed}"] = {
                "expand": expand,
                "comparison": paired,
                "reaches_reference": reaches_ref,
            }
            passed_seeds.append(seed)
            write_json(self.output / "status.json", self.status)
            if seed == 42 and not expand:
                break
        # Only now look at the independently generated validation bank. It is
        # never consulted by a warmup/expansion gate and never placed in replay.
        self.run(
            "reference_validation",
            "evaluate",
            g0,
            cases=self.cases["validation"],
            reference=True,
        )
        comparisons = {}
        for seed in passed_seeds:
            self.parallel(
                *[
                    lambda arm=arm, gpu=gpu: self.run(
                        f"{arm}{seed}_validation",
                        "evaluate",
                        gpu,
                        cases=self.cases["validation"],
                        checkpoint=self.output / f"{arm}{seed}/checkpoint.pt",
                        objective=arm,
                    )
                    for arm, gpu in (("bc_only", g0), ("q_bc", g1))
                ]
            )
            comparisons[str(seed)] = {
                arm: matched(
                    self.episodes("reference_validation"),
                    self.episodes(f"{arm}{seed}_validation"),
                )
                for arm in ("bc_only", "q_bc")
            }
        write_json(
            self.output / "results.json",
            {
                "protocol": PROTOCOL,
                "validation": comparisons,
                "seeds": passed_seeds,
                "limitation": "Task0, local RLT_a acceptance recipe; conditional replication and no official benchmark claim",
            },
        )
        self.status["state"] = "complete"


def main() -> None:
    """Create a new campaign directory and preserve every failed phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--control-budget", type=int, default=32000)
    parser.add_argument("--bc-updates", type=int, default=5000)
    parser.add_argument("--wait-hours", type=float, default=10)
    parser.add_argument("--job-hours", type=float, default=4)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument(
        "--wandb", choices=["online", "offline", "disabled"], default="online"
    )
    args = parser.parse_args()
    if args.output.exists() or len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("Require a new directory and two distinct physical GPUs")
    if (
        min(
            args.wait_hours,
            args.job_hours,
            args.max_hours,
            args.control_budget,
            args.bc_updates,
        )
        <= 0
    ):
        parser.error("Budgets and time limits must be positive")
    campaign = Campaign(args)

    def stop(signum, frame):
        campaign.stop.set()
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    try:
        campaign.execute()
    except BaseException as error:
        campaign.status.update(state="failed", error=str(error))
        raise
    finally:
        write_json(campaign.output / "status.json", campaign.status)


if __name__ == "__main__":
    main()
