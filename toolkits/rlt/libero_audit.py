# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit two evaluation entries with fixed weights and recorded failed states."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from toolkits.rlt.libero_reproduction import (
    ASSETS,
    asset_paths,
    check_source,
    episode_plan,
    runtime_versions,
)

LOGGER = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """Hash a file without allocating its full contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def failed_cases(previous: Path, tasks: list[int]) -> dict[int, list[int]]:
    """Select every RLT_a failure in the requested tasks from a completed run."""
    episode_plan(tasks, [0])
    manifest = json.loads((previous / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("mode") != "evaluate":
        raise ValueError("Failure selection requires a completed evaluation")
    if manifest.get("learner_dir") or manifest.get("assets") != {
        key: list(value) for key, value in ASSETS.items()
    }:
        raise ValueError("Audit requires the same pinned public release")
    rows = json.loads((previous / "rlt_a.json").read_text())
    seen = set()
    result = {task: [] for task in tasks}
    for row in rows:
        task, state = row["task"], row["state"]
        episode_plan([task], [state])
        if (task, state) in seen or not isinstance(row["success"], bool):
            raise ValueError("Invalid or duplicate task/state outcome")
        seen.add((task, state))
        if task in result and not row["success"]:
            result[task].append(state)
    expected = set(episode_plan(manifest["tasks"], manifest["states"]))
    if seen != expected or any(not states for states in result.values()):
        raise ValueError("Missing outcomes or no failures in a requested task")
    return {task: sorted(states) for task, states in result.items()}


def compare_records(left: list[dict], right: list[dict]) -> dict:
    """Report the first exact trace difference and executed-action discrepancy."""
    import numpy as np

    first = None
    max_action_error = 0.0
    for i, (a, b) in enumerate(zip(left, right)):
        if first is None and a != b:
            first = {
                "record": i,
                "kind": a.get("kind"),
                "fields": sorted(
                    k for k in a.keys() | b.keys() if a.get(k) != b.get(k)
                ),
            }
        if a.get("kind") == b.get("kind") == "step":
            max_action_error = max(
                max_action_error,
                float(np.max(np.abs(np.asarray(a["action"]) - b["action"]))),
            )
    if len(left) != len(right) and first is None:
        first = {"record": min(len(left), len(right)), "fields": ["length"]}
    return {
        "exact_match": first is None,
        "first_difference": first,
        "records": [len(left), len(right)],
        "max_executed_action_abs_diff_common_prefix": max_action_error,
    }


def observation_record(obs: dict) -> dict:
    """Capture exact input pixels and proprioception without transforming them."""
    import numpy as np

    result = {"state": np.asarray(obs["state"]).tolist()}
    for key in ("primary_image", "wrist_image"):
        pixels = np.ascontiguousarray(obs[key])
        result[key] = {
            "shape": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        }
    return result


@contextmanager
def trace_evaluation(output: Path, cases: dict[int, list[int]]):
    """Observe existing env/actor calls; filter only the explicit episode IDs.

    The official Python main and the local evaluate function retain their own
    model construction. Both use the pinned upstream rollout function and the
    same IPC/EGL compatibility wrapper. This is not an environment-version audit.
    """
    import numpy as np
    import torch
    from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
        ActionTokenActor,
    )
    from AlphaBrain.training.reinforcement_learning.envs import libero_env
    from AlphaBrain.training.reinforcement_learning.eval import eval_helpers

    from toolkits.rlt.libero_reproduction import ReferenceActor

    active = threading.local()
    original_helper = eval_helpers._eval_deterministic_local
    original_actor = ActionTokenActor.forward
    original_reference = ReferenceActor.__call__
    summaries = []

    class ObservedEnv(libero_env.LiberoEnv):
        def reset(self, **kwargs):
            obs = super().reset(**kwargs)
            task, state = kwargs["task_id"], kwargs["initial_state_idx"]
            self.trace_dir = (
                output / "traces" / active.arm / f"task_{task}_state_{state}"
            )
            self.trace_dir.mkdir(parents=True, exist_ok=False)
            self.trace = (self.trace_dir / "trace.jsonl").open("x", buffering=1)
            self.tick = 0
            active.env = self
            self.record(
                {
                    "kind": "reset",
                    "task": task,
                    "initial_state": state,
                    "seed": kwargs["seed"],
                    "observation": observation_record(obs),
                }
            )
            obs["primary_image"].save(self.trace_dir / "initial_primary.png")
            obs["wrist_image"].save(self.trace_dir / "initial_wrist.png")
            return obs

        def record(self, row):
            self.trace.write(json.dumps(row, allow_nan=False) + "\n")

        def step(self, action):
            executed = np.asarray(action).copy()
            obs, reward, done = super().step(action)
            self.record(
                {
                    "kind": "step",
                    "tick": self.tick,
                    "action": executed.tolist(),
                    "reward": reward,
                    "done": done,
                    "observation": observation_record(obs),
                }
            )
            self.tick += 1
            return obs, reward, done

        def close(self):
            try:
                super().close()
            finally:
                if hasattr(self, "trace") and not self.trace.closed:
                    self.record({"kind": "close", "ticks": self.tick})
                    self.trace.close()

    def record_decision(token, reference, proprio, action):
        if torch.is_grad_enabled():
            raise RuntimeError("Audit requires inference without gradients")
        row = {
            "kind": "decision",
            "tick": active.env.tick,
            "reference_chunk": reference.detach().float().cpu().tolist(),
            "actor_chunk": action.detach().float().cpu().tolist(),
            "proprio": proprio.detach().float().cpu().tolist(),
        }
        if token is not None:
            row["rl_token"] = token.detach().float().cpu().tolist()
        active.env.record(row)

    def actor_forward(model, token, reference, proprio=None, deterministic=False):
        if model.training or not deterministic:
            raise RuntimeError("Actor must be deterministic and in eval mode")
        result = original_actor(
            model, token, reference, proprio, deterministic=deterministic
        )
        record_decision(token, reference, proprio, result[0])
        return result

    def reference_forward(model, token, reference, proprio, *, deterministic):
        result = original_reference(
            model, token, reference, proprio, deterministic=deterministic
        )
        record_decision(token, reference, proprio, result[0])
        return result

    def selected_helper(**kwargs):
        task = kwargs["task_id"]
        selected = [
            state for state in kwargs["episode_indices"] if state in cases.get(task, [])
        ]
        if not selected:
            return []
        active.arm = (
            "reference" if isinstance(kwargs["actor"], ReferenceActor) else "rlt_a"
        )
        kwargs["episode_indices"] = selected
        rows = original_helper(**kwargs)
        summaries.extend(
            {"arm": active.arm, "task": task, "state": state, "success": bool(success)}
            for _, state, success in rows
        )
        return rows

    with (
        patch.object(libero_env, "LiberoEnv", ObservedEnv),
        patch.object(eval_helpers, "_eval_deterministic_local", selected_helper),
        patch.object(ActionTokenActor, "forward", actor_forward),
        patch.object(ReferenceActor, "__call__", reference_forward),
    ):
        yield summaries


def run_entry(args: argparse.Namespace, cases: dict[int, list[int]]) -> None:
    """Run one unchanged entry in its own process with passive trace observers."""
    check_source(args.source)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        MUJOCO_GL="egl",
        LIBERO_PYTHON=sys.executable,
        TOKENIZERS_PARALLELISM="false",
        RLT_ALPHABRAIN_SOURCE=str(args.source),
        RLT_LIBERO_EGL_DEVICE_ID=str(args.egl_index),
        RLT_LIBERO_WORKER_LOG_DIR=str(args.output / "worker_logs"),
    )
    sys.path.insert(0, str(args.source))
    os.environ["PYTHONPATH"] = (
        str(args.source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    from AlphaBrain.training.reinforcement_learning.envs import libero_env

    libero_env._WORKER_SCRIPT = str(Path(__file__).with_name("libero_worker.py"))
    args.output.mkdir(parents=True, exist_ok=False)
    with trace_evaluation(args.output, cases) as rows:
        if args.entry == "wrapped":
            from toolkits.rlt.libero_reproduction import evaluate

            config = SimpleNamespace(
                storage=args.storage,
                learner_dir=None,
                tasks=list(cases),
                states=sorted({s for v in cases.values() for s in v}),
                seed=args.seed,
                output=args.output,
                video=True,
            )
            evaluate(
                config, cases=[(t, s) for t, states in cases.items() for s in states]
            )
        else:
            # The official Python entry is preserved. Its multi-GPU shell script
            # is intentionally not used: it includes user-wide worker cleanup.
            import runpy

            from AlphaBrain.training.reinforcement_learning import _bootstrap

            vla, rlt = asset_paths(args.storage)
            sys.argv = [
                "eval_libero.py",
                "--vla_ckpt",
                str(vla),
                "--action_token_ckpt",
                str(rlt),
                "--suite",
                "libero_goal",
                "--task_ids",
                ",".join(map(str, cases)),
                "--n_eps_per_task",
                "50",
                "--num_workers",
                "1",
                "--gpu",
                str(args.gpu),
                "--seed",
                str(args.seed),
                "--actor_hidden_dim",
                "512",
                "--video_dir",
                str(args.output / "videos"),
                "--results_json",
                str(args.output / "upstream_summary.json"),
            ]
            # Keep explicit audit environment, not machine-dependent .env defaults.
            with patch.object(_bootstrap, "_load_env_from_repo_root", lambda: None):
                runpy.run_path(
                    str(
                        args.source
                        / "AlphaBrain/training/reinforcement_learning/eval/eval_libero.py"
                    ),
                    run_name="__main__",
                )
    expected = {(task, state) for task, states in cases.items() for state in states}
    for arm in ["reference", "rlt_a"] if args.entry == "wrapped" else ["rlt_a"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if (
            len(arm_rows) != len(expected)
            or {(r["task"], r["state"]) for r in arm_rows} != expected
        ):
            raise RuntimeError(f"Incomplete or duplicate {arm} audit")
    (args.output / "outcomes.json").write_text(json.dumps(rows, indent=2) + "\n")


def summarize(output: Path, cases: dict[int, list[int]]) -> dict:
    """Require complete per-episode artifacts before comparing the two entries."""
    comparisons = []
    outcomes = {
        entry: json.loads((output / entry / "outcomes.json").read_text())
        for entry in ("official", "wrapped")
    }
    for task, states in cases.items():
        for state in states:
            traces = []
            successes = []
            for entry in ("official", "wrapped"):
                path = (
                    output
                    / entry
                    / "traces/rlt_a"
                    / f"task_{task}_state_{state}/trace.jsonl"
                )
                records = [json.loads(line) for line in path.read_text().splitlines()]
                if (
                    not records
                    or records[0]["kind"] != "reset"
                    or records[-1]["kind"] != "close"
                ):
                    raise ValueError(f"Incomplete trace: {path}")
                traces.append(records)
                matches = [
                    r
                    for r in outcomes[entry]
                    if r["arm"] == "rlt_a" and r["task"] == task and r["state"] == state
                ]
                if len(matches) != 1:
                    raise ValueError("Missing or duplicate audit outcome")
                successes.append(matches[0]["success"])
            comparisons.append(
                {
                    "task": task,
                    "state": state,
                    "success_official_wrapped": successes,
                    **compare_records(*traces),
                }
            )
    return {
        "complete": True,
        "episodes_per_entry": len(comparisons),
        "exact_trace_matches": sum(r["exact_match"] for r in comparisons),
        "outcomes": outcomes,
        "comparisons": comparisons,
        "limitation": "Failure-selected diagnostics, not a benchmark success rate. Both entries share the pinned upstream evaluator and the local simulator/IPC adapter; matching does not validate the author's 92% environment.",
    }


def main() -> None:
    """Select failures, lease an idle GPU, and run two bounded read-only entries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[5, 6, 9])
    parser.add_argument("--gpu", type=int, choices=[2], default=2)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--entry", choices=["official", "wrapped"], help=argparse.SUPPRESS
    )
    parser.add_argument("--gpu-uuid", help=argparse.SUPPRESS)
    parser.add_argument("--egl-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.source = args.source.resolve()
    check_source(args.source)
    cases = failed_cases(args.previous, args.tasks)
    previous_manifest = json.loads((args.previous / "manifest.json").read_text())
    args.seed = previous_manifest["seed"]
    if args.entry:
        run_entry(args, cases)
        return
    LOGGER.info(
        "Failure-selected cases: %s (%d episodes per entry)",
        cases,
        sum(map(len, cases.values())),
    )
    if args.check:
        return
    if args.output.exists():
        parser.error("--output must be a new directory")
    import fcntl

    from toolkits.rlt.libero_egl import egl_device_index

    with Path("/dev/shm/rlt-libero-audit-gpu2.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpu_uuid, memory = map(str.strip, info.split(","))
        if int(memory) > 512:
            parser.error(f"GPU 2 busy ({memory} MiB); no model loaded")
        egl_index = egl_device_index(gpu_uuid)
        args.output.mkdir(parents=True)
        vla, rlt = asset_paths(args.storage)
        weights = [
            vla / "model.safetensors",
            rlt / "actor.pt",
            rlt / "encoder.pt",
            rlt / "critic.pt",
        ]
        LOGGER.info("Hashing public weights before read-only evaluation")
        before = {str(path): file_hash(path) for path in weights}
        manifest = {
            "complete": False,
            "cases": cases,
            "selection": "all public RLT_a failures in requested tasks",
            "previous": str(args.previous),
            "source_revision": check_source(args.source),
            "assets": ASSETS,
            "seed": args.seed,
            "gpu_uuid": gpu_uuid,
            "egl_index": egl_index,
            "runtime_versions": runtime_versions(),
            "training": False,
            "assistance": False,
            "weight_sha256_before": before,
            "audit_sha256": file_hash(Path(__file__)),
            "compatibility_adapters": [
                "shared IPC stdout isolation",
                "UUID/EGL mapping",
                "explicit failure ID filter",
                "passive trace hooks",
                "official .env loading disabled",
                "official num_workers=1",
            ],
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for entry in ("official", "wrapped"):
            command = [
                sys.executable,
                "-u",
                "-m",
                "toolkits.rlt.libero_audit",
                "--source",
                str(args.source),
                "--storage",
                str(args.storage),
                "--previous",
                str(args.previous),
                "--output",
                str(args.output / entry),
                "--tasks",
                *map(str, args.tasks),
                "--entry",
                entry,
                "--gpu-uuid",
                gpu_uuid,
                "--egl-index",
                str(egl_index),
            ]
            LOGGER.info("Starting %s entry", entry)
            with (args.output / f"{entry}.log").open("x") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=7200,
                )
        report = summarize(args.output, cases)
        after = {str(path): file_hash(path) for path in weights}
        if before != after:
            raise RuntimeError("Checkpoint files changed during the audit")
        manifest.update(
            complete=True, weight_sha256_after=after, weights_unchanged=True
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        LOGGER.info(
            "Audit complete: %d/%d exact traces; weights unchanged",
            report["exact_trace_matches"],
            report["episodes_per_entry"],
        )


if __name__ == "__main__":
    main()
