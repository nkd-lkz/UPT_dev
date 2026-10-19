# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect frozen-reference features on declared non-evaluation RoboDojo layouts."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from toolkits.rlt.experiment_queue import gpu_idle, stop_group
from toolkits.rlt.robodojo_boundary import (
    ACTION_SEMANTICS,
    FIELDS,
    decode_actions,
    encode_observation,
)
from toolkits.rlt.robodojo_eval import (
    ROBO_DOJO_REVISION,
    XPOLICYLAB_REVISION,
    _source_check,
)
from toolkits.rlt.robodojo_layouts import digest, verify
from toolkits.rlt.robodojo_runner import (
    foreign_gpu_processes,
    owns_listener,
    renderer_index,
)
from toolkits.rlt.robodojo_transition import RoboDojoTransitions


def save(path: Path, data: dict) -> None:
    """Publish a complete JSON status without exposing a partially written file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native_task_name(config: dict) -> str:
    """Return the explicitly configured native task without a fallback."""
    task = config.get("task")
    if not isinstance(task, str) or re.fullmatch(r"[a-z][a-z0-9_]*", task) is None:
        raise ValueError("Require an explicit snake_case RoboDojo task")
    return task


def prior_outcomes(path: Path | None, manifest: Path) -> list[dict]:
    """Load completed episodes from one compatible, immutable prior attempt."""
    if path is None:
        return []
    provenance = json.loads((path / "provenance.json").read_text())
    if provenance.get("layout_sha256") != digest(manifest):
        raise ValueError("Resume attempt uses a different layout manifest")
    result = path / "collection-result.json"
    failure = path / "collection-failure.json"
    payload = json.loads((result if result.is_file() else failure).read_text())
    rows = payload.get("episodes", payload.get("outcomes"))
    if not isinstance(rows, list):
        raise ValueError("Resume attempt has no explicit episode outcomes")
    completed = []
    for row in rows:
        if (
            isinstance(row, dict)
            and type(row.get("layout_id")) is int
            and row.get("role") in {"train", "development"}
            and row.get("state") == "complete"
            and row.get("end_reason") in {"success", "failure", "time_limit"}
        ):
            completed.append(dict(row) | {"artifact_root": str(path)})
    return completed


def select_collection_layouts(
    split: dict,
    *,
    train_episodes: int,
    development_episodes: int,
    train_offset: int = 0,
    development_offset: int = 0,
) -> list[dict]:
    """Select an explicit, non-evaluation layout window for one collection run."""
    train = [row for row in split["layouts"] if row["role"] == "train"]
    development = [row for row in split["layouts"] if row["role"] == "development"]
    selected = train[train_offset : train_offset + train_episodes]
    selected += development[
        development_offset : development_offset + development_episodes
    ]
    expected = train_episodes + development_episodes
    if len(selected) != expected:
        raise ValueError(
            "Requested layout window exceeds the declared train/development split"
        )
    return selected


def select_reserved_layouts(split: dict, episodes: int) -> list[dict]:
    """Select a fixed protected layout window for read-only reference evaluation."""
    selected = [row for row in split["layouts"] if row["role"] == "reserved"]
    if len(selected) < episodes:
        raise ValueError("Requested evaluation exceeds the declared reserved split")
    return selected[:episodes]


class FeatureModel:
    """Serve reference chunks and save pre-action features, never privileged inputs."""

    def __init__(
        self,
        config: dict,
        manifest: Path,
        output: Path,
        *,
        reserved_evaluation: bool = False,
    ) -> None:
        from toolkits.rlt.robodojo_features import FrozenArxFeatures

        self.backend = FrozenArxFeatures(config["policy"])
        self.reserved_evaluation = reserved_evaluation
        self.layouts = {
            row["id"]: row
            for row in verify(manifest)["layouts"]
            if (row["role"] == "reserved") == reserved_evaluation
        }
        self.output = output
        self.reset()

    def reset(self) -> None:
        """Invalidate observations and require explicit layout identity again."""
        self.layout_id = None
        self.observation = None
        self.chunk = 0

    def begin_episode(self, identity: dict) -> None:
        """Bind this reference rollout to one declared train/development layout."""
        index = identity["layout_id"]
        if (
            index not in self.layouts
            or identity["sha256"] != self.layouts[index]["sha256"]
        ):
            raise ValueError("Undeclared reference-collection identity")
        self.reset()
        self.layout_id = index
        episode_id = identity.get("episode_id", index)
        if (
            type(episode_id) not in {int, str}
            or not str(episode_id).replace("-", "").isalnum()
        ):
            raise ValueError("Require a filesystem-safe episode identity")
        self.episode_id = str(episode_id)
        (self.output / f"episode-{self.episode_id}").mkdir(exist_ok=False)

    def update_obs(self, raw: dict) -> None:
        """Keep only the same three RGB views, state and language as the baseline."""
        if self.layout_id is None:
            raise RuntimeError("Begin a declared episode first")
        self.observation = encode_observation(raw)

    def get_action(self) -> dict:
        """Persist features before execution and return one ten-action reference chunk."""
        if self.observation is None:
            raise RuntimeError("Fresh observation required")
        observation, self.observation = self.observation, None
        prediction = self.backend.infer(
            observation,
            seed=130013 + self.layout_id * 1000 + self.chunk,
            check_parity=self.chunk == 0,
        )
        commands = decode_actions(
            prediction["reference"], semantics=ACTION_SEMANTICS, horizon=10
        )
        cache = None
        if not self.reserved_evaluation:
            cache = (
                self.output / f"episode-{self.episode_id}" / f"chunk-{self.chunk}.npz"
            )
            np.savez(
                cache,
                hidden=prediction["hidden"],
                state=observation["state"],
                reference=prediction["reference"][:10],
            )
        result = {
            "commands": commands,
            "cache_file": str(cache) if cache else None,
            "chunk": self.chunk,
            "parity_max_abs": prediction["parity_max_abs"],
        }
        if self.reserved_evaluation:
            result["features"] = {
                "hidden": prediction["hidden"],
                "state": observation["state"],
                "reference": prediction["reference"][:10],
            }
        self.chunk += 1
        return result


def create_native(config: dict, app, port: int):
    """Mirror pinned native main config assembly, leaving all source files unchanged."""
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from src.eval_client.eval_env import create_eval_env
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    task = native_task_name(config)
    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    evaluation = load_yaml(str(Path(ENV_CONFIG_PATH) / "arx_x5.yml"))
    evaluation.update(
        task_name=task,
        num_envs=1,
        device_id=0,
        eval_batch=True,
        policy_name="RLT_Pi05",
        additional_info="training_collection",
        seed=0,
        physx_monitor_enabled=False,
    )
    assembled = OmegaConf.create(
        {
            **{
                name: load_yaml(
                    str(
                        Path(ENV_CONFIG_PATH)
                        / name
                        / (evaluation["config"][name] + ".yml")
                    )
                )
                for name in ("sim", "scene", "camera", "robot")
            },
            "task_env": load_yaml(
                registry.task_config_path(
                    str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), task
                )
            ),
            "eval_cfg": evaluation,
            "deploy_cfg": {
                "policy_name": "RLT_Pi05",
                "host": "127.0.0.1",
                "port": port,
                "protocol": "ws",
                "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
                "trial_id": os.environ["ROBODOJO_RUN_ID"],
                "action_case_id": "collection",
            },
        }
    )
    OmegaConf.update(assembled, "sim.scene.num_envs", 1, force_add=True)
    assembled, _ = process_config(process_randomization(assembled), task_name=task)
    OmegaConf.update(
        assembled,
        "camera.default_frequency",
        evaluation["observation"].get("collect_freq", 0),
        force_add=True,
    )
    assembled.sim.seed = [0]
    return create_eval_env(assembled, app)


def close_native(native) -> None:
    """Close the native environment and its persistent policy connection once."""
    try:
        native.model_client.close()
    finally:
        native.close()


def client(args: argparse.Namespace, config: dict) -> None:
    """Execute native-length episodes without reusing or replacing protected layouts."""
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    from isaaclab.app import AppLauncher

    launch = argparse.Namespace(
        headless=True,
        enable_cameras=True,
        device="cuda:0",
        kit_args=f"--/renderer/activeGpu={args.renderer_index} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0",
    )
    add_zero_delay_kit_args(launch)
    app = AppLauncher(launch).app
    native = None
    outcomes = prior_outcomes(args.resume_from, args.layouts)
    try:
        split = verify(args.layouts)
        if args.reserved_evaluation_episodes:
            selected = select_reserved_layouts(split, args.reserved_evaluation_episodes)
        else:
            selected = select_collection_layouts(
                split,
                train_episodes=args.train_episodes,
                development_episodes=args.development_episodes,
                train_offset=args.train_offset,
                development_offset=args.development_offset,
            )
        completed_ids = {row["layout_id"] for row in outcomes}
        selected = [row for row in selected if row["id"] not in completed_ids]
        for entry in selected:
            # Isaac's scene registry is assembled around the first layout. A fresh
            # native env per independent layout prevents later articulation names
            # from referring to objects that were never registered in that scene.
            native = create_native(config, app, args.port)
            native.seed_manager.seed_info = {
                entry["id"]: {"scene_layout": entry["path"]}
            }
            wrapped = RoboDojoTransitions(
                native,
                layouts={entry["id"]: entry["sha256"]},
                protected_evaluation_hashes=frozenset(
                    row["sha256"] for row in split["protected"]
                ),
                mode="training" if entry["role"] == "train" else "probe",
            )
            try:
                observation, info = wrapped.reset(entry["id"])
            except Exception as exc:
                # An unstable generated layout is retained as a failure and is
                # never silently replaced, but it does not discard other IDs.
                outcomes.append(
                    {
                        "layout_id": entry["id"],
                        "role": entry["role"],
                        "state": "failed",
                        "phase": "reset",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                save(args.output / "collection-progress.json", {"completed": outcomes})
                close_native(native)
                native = None
                continue
            try:
                native.model_client.call(
                    func_name="begin_episode",
                    obs={"layout_id": entry["id"], "sha256": entry["sha256"]},
                )
                records = []
                while True:
                    native.model_client.call(
                        func_name="update_obs", obs=native.get_obs()
                    )
                    prediction = native.model_client.call(func_name="get_action")
                    prefix = []
                    reward = 0.0
                    for tick, command in enumerate(prediction["commands"]):
                        transition = wrapped.step(
                            np.concatenate([command[name] for name, _ in FIELDS]),
                            source="reference_teacher",
                        )
                        prefix.append(transition.info["executed_action"].tolist())
                        reward += 0.99**tick * transition.reward
                        if transition.terminated or transition.truncated:
                            break
                    records.append(
                        {
                            "chunk": prediction["chunk"],
                            "cache_file": prediction["cache_file"],
                            "parity_max_abs": prediction["parity_max_abs"],
                            "ticks": len(prefix),
                            "executed_physical": prefix,
                            "reward": reward,
                            "process_score": transition.info["process_score"],
                            "terminated": transition.terminated,
                            "truncated": transition.truncated,
                            "post_state": transition.observation["state"].tolist(),
                            "end_reason": transition.info["end_reason"],
                        }
                    )
                    episode = {
                        "layout": entry,
                        "reset": info,
                        "records": records,
                        "complete": transition.terminated or transition.truncated,
                    }
                    save(
                        args.output / f"episode-{entry['id']}" / "episode.json", episode
                    )
                    save(
                        args.output / "collection-progress.json",
                        {
                            "completed": outcomes,
                            "current_layout": entry["id"],
                            "control_steps": transition.info["control_step"],
                            "chunks": len(records),
                        },
                    )
                    if transition.terminated or transition.truncated:
                        outcomes.append(
                            {
                                "layout_id": entry["id"],
                                "role": entry["role"],
                                "state": "complete",
                                "ticks": transition.info["control_step"],
                                "end_reason": transition.info["end_reason"],
                                "max_process_score": max(
                                    row["process_score"] for row in records
                                ),
                                "artifact_root": str(args.output),
                            }
                        )
                        save(
                            args.output / "collection-progress.json",
                            {"completed": outcomes},
                        )
                        break
            except Exception as exc:
                # A bad layout is a reported failure, never silently substituted.
                outcomes.append(
                    {
                        "layout_id": entry["id"],
                        "role": entry["role"],
                        "state": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                save(args.output / "collection-failure.json", {"outcomes": outcomes})
                raise
            close_native(native)
            native = None
        save(
            args.output / "collection-result.json",
            {
                "complete": True,
                "episodes": outcomes,
                "manifest_sha256": digest(args.layouts),
                "training_collection": not args.reserved_evaluation_episodes,
                "evaluation_role": (
                    "reserved" if args.reserved_evaluation_episodes else None
                ),
                "online_rl": False,
                "resume_from": str(args.resume_from) if args.resume_from else None,
            },
        )
    finally:
        if native is not None:
            close_native(native)
        app.close()


def supervise(args: argparse.Namespace, config: dict) -> None:
    """Own one selected GPU and preserve every bounded collection attempt."""
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise ValueError("Explicit operator EULA acceptance required")
    root = Path(config["robodojo_root"])
    _source_check(root, ROBO_DOJO_REVISION)
    _source_check(root / "XPolicyLab", XPOLICYLAB_REVISION)
    verify(args.layouts)
    feature_gate = json.loads(args.feature_gate.read_text())
    if feature_gate.get("state") != "passed" or feature_gate.get("parity_max_abs") != 0:
        raise ValueError("Frozen feature parity gate must pass before collection")
    args.output.mkdir(parents=True, exist_ok=False)
    status = {
        "state": "starting",
        "online_rl": False,
        "train_episodes": args.train_episodes,
        "development_episodes": args.development_episodes,
        "train_offset": args.train_offset,
        "development_offset": args.development_offset,
        "reserved_evaluation_episodes": args.reserved_evaluation_episodes,
        "gpu_index": args.gpu,
        "resume_from": str(args.resume_from) if args.resume_from else None,
    }
    save(
        args.output / "provenance.json",
        {
            "config": config,
            "layout_manifest": str(args.layouts),
            "layout_sha256": digest(args.layouts),
            "feature_gate": feature_gate,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "layout_window": {
                "train_offset": args.train_offset,
                "train_episodes": args.train_episodes,
                "development_offset": args.development_offset,
                "development_episodes": args.development_episodes,
                "reserved_evaluation_episodes": args.reserved_evaluation_episodes,
            },
            "source_sha256": {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "robodojo_collect.py",
                    "robodojo_features.py",
                    "robodojo_transition.py",
                    "robodojo_layouts.py",
                )
            },
        },
    )
    children = []
    lease = Path(f"/dev/shm/rlt-research-gpu{args.gpu}.lock").open("a")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_checks = 0
        wait_deadline = time.monotonic() + args.wait_idle_hours * 3600
        while True:
            idle, detail = gpu_idle(args.gpu)
            foreign = foreign_gpu_processes([], gpu=args.gpu)
            idle_checks = idle_checks + 1 if idle and not foreign else 0
            if idle_checks >= 2:
                break
            status.update(
                state="waiting_gpu",
                gpu=detail,
                foreign_pids=foreign,
                idle_checks=idle_checks,
            )
            save(args.output / "status.json", status)
            if not args.wait_idle_hours or time.monotonic() >= wait_deadline:
                raise RuntimeError(f"GPU {args.gpu} occupied: {detail}; pids={foreign}")
            time.sleep(min(30, max(0, wait_deadline - time.monotonic())))
        uuid = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        renderer = renderer_index(args.renderer_table.read_text(), uuid)
        environment = dict(os.environ)
        environment.pop("LD_PRELOAD", None)
        environment.update(
            CUDA_VISIBLE_DEVICES=uuid,
            LD_LIBRARY_PATH="",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            NO_PROXY="127.0.0.1,localhost",
            no_proxy="127.0.0.1,localhost",
            ROBODOJO_RUN_ID=args.output.name,
            PYTHONPATH=os.pathsep.join(
                map(
                    str,
                    (Path(__file__).resolve().parents[2], root, root / "XPolicyLab"),
                )
            ),
        )
        common = [
            "-u",
            "-m",
            "toolkits.rlt.robodojo_collect",
            "--config",
            str(args.config),
            "--output",
            str(args.output),
            "--layouts",
            str(args.layouts),
            "--port",
            str(args.port),
            "--train-episodes",
            str(args.train_episodes),
            "--development-episodes",
            str(args.development_episodes),
            "--train-offset",
            str(args.train_offset),
            "--development-offset",
            str(args.development_offset),
            "--reserved-evaluation-episodes",
            str(args.reserved_evaluation_episodes),
            "--gpu",
            str(args.gpu),
        ]
        if args.task:
            common.extend(("--task", args.task))
        if args.resume_from:
            common.extend(("--resume-from", str(args.resume_from)))
        with (
            (args.output / "server.log").open("x") as server_log,
            (args.output / "client.log").open("x") as client_log,
        ):
            server = subprocess.Popen(
                [config["policy_python"], *common, "--server"],
                cwd=root,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(server)
            status.update(state="loading_reference", server_pid=server.pid)
            save(args.output / "status.json", status)
            deadline = time.monotonic() + 600
            while not owns_listener(server.pid, args.port):
                server_code = server.poll()
                timed_out = time.monotonic() > deadline
                foreign = foreign_gpu_processes(children, gpu=args.gpu)
                if server_code is not None or timed_out or foreign:
                    raise RuntimeError(
                        "Reference startup failed: "
                        f"server_returncode={server_code}, "
                        f"timed_out={timed_out}, foreign_pids={foreign}"
                    )
                if (args.output / "STOP").exists():
                    raise KeyboardInterrupt("Operator STOP")
                time.sleep(2)
            environment.update(
                LD_LIBRARY_PATH=str(args.sim_library_path),
                VK_ICD_FILENAMES=str(args.vulkan_icd),
                VK_DRIVER_FILES=str(args.vulkan_icd),
            )
            process = subprocess.Popen(
                [
                    config["sim_python"],
                    *common,
                    "--client",
                    "--renderer-index",
                    str(renderer),
                ],
                cwd=root,
                env=environment,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(process)
            status.update(state="collecting", client_pid=process.pid)
            save(args.output / "status.json", status)
            deadline = time.monotonic() + args.timeout_seconds
            while process.poll() is None:
                server_code = server.poll()
                timed_out = time.monotonic() > deadline
                foreign = foreign_gpu_processes(children, gpu=args.gpu)
                if server_code is not None or timed_out or foreign:
                    raise RuntimeError(
                        "Collection stopped unexpectedly: "
                        f"server_returncode={server_code}, "
                        f"client_returncode={process.poll()}, "
                        f"timed_out={timed_out}, foreign_pids={foreign}"
                    )
                if (args.output / "STOP").exists():
                    raise KeyboardInterrupt("Operator STOP")
                time.sleep(2)
            result = json.loads((args.output / "collection-result.json").read_text())
            if process.returncode or not result["complete"]:
                raise RuntimeError("Missing successful explicit collection result")
            status.update(state="complete", result=result)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child in reversed(children):
            stop_group(child)
        lease.close()
        status["finished_at"] = time.time()
        save(args.output / "status.json", status)
        signal.signal(signal.SIGTERM, previous)


def main() -> None:
    """Select isolated feature server, Isaac client or GPU-owning supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "output", "layouts"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("feature-gate", "renderer-table", "sim-library-path", "vulkan-icd"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--renderer-index", type=int)
    parser.add_argument("--port", type=int, default=6134)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--development-episodes", type=int, default=2)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--development-offset", type=int, default=0)
    parser.add_argument("--reserved-evaluation-episodes", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--wait-idle-hours", type=float, default=0)
    parser.add_argument("--gpu", type=int, choices=range(3), default=2)
    parser.add_argument("--task")
    args = parser.parse_args()
    reserved_evaluation = bool(args.reserved_evaluation_episodes)
    if reserved_evaluation:
        if (
            not 1 <= args.reserved_evaluation_episodes <= 25
            or args.train_episodes
            or args.development_episodes
            or args.resume_from is not None
        ):
            parser.error(
                "Reserved evaluation requires 1..25 episodes, no collection "
                "episodes and no resume"
            )
    elif not 1 <= args.train_episodes <= 16 or not 1 <= args.development_episodes <= 4:
        parser.error("Require 1..16 train and 1..4 development episodes")
    if args.train_offset < 0 or args.development_offset < 0:
        parser.error("Layout offsets must be non-negative")
    if not 0 <= args.wait_idle_hours <= 72:
        parser.error("wait-idle-hours must be in [0, 72]")
    config = yaml.safe_load(args.config.read_text())
    if args.task:
        if re.fullmatch(r"[a-z][a-z0-9_]*", args.task) is None:
            parser.error("Task override must be snake_case")
        config = dict(config, task=args.task)
    if args.server:
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig

        asyncio.run(
            PolicyServer(
                FeatureModel(
                    config,
                    args.layouts,
                    args.output,
                    reserved_evaluation=reserved_evaluation,
                ),
                PolicyServerConfig(host="127.0.0.1", port=args.port),
            ).serve_forever()
        )
    elif args.client:
        client(args, config)
    else:
        if not all(
            (
                args.feature_gate,
                args.renderer_table,
                args.sim_library_path,
                args.vulkan_icd,
            )
        ):
            parser.error(
                "Supervisor requires feature gate, renderer table, graphics libraries and ICD"
            )
        supervise(args, config)


if __name__ == "__main__":
    main()
