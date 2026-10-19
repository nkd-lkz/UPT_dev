# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run a three-action native reference probe through the transition adapter."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from toolkits.rlt.experiment_queue import gpu_idle, stop_group
from toolkits.rlt.robodojo_boundary import FIELDS
from toolkits.rlt.robodojo_collect import native_task_name
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
from toolkits.rlt.robodojo_task_plan import CANDIDATES
from toolkits.rlt.robodojo_transition import RoboDojoTransitions

SUPPORTED_TASKS = frozenset(candidate.name for candidate in CANDIDATES)


def _save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_probe_config(config: dict) -> str:
    """Validate that a selected task uses the shared frozen reference backend."""
    task = native_task_name(config)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task is outside the selected compatibility set: {task}")
    if config.get("policy", {}).get("backend") != "openpi_reference":
        raise ValueError("Compatibility probes require the frozen OpenPI reference")
    return task


def native_probe(args: argparse.Namespace, config: dict) -> None:
    """Construct the native scene and exercise exactly one declared probe reset."""
    from isaaclab.app import AppLauncher

    launch = argparse.Namespace(
        headless=True,
        enable_cameras=True,
        device="cuda:0",
        kit_args=f"--/renderer/activeGpu={args.renderer_index} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0",
    )
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(launch)
    app = AppLauncher(launch).app
    env = None
    try:
        import importlib

        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from omegaconf import OmegaConf
        from src.eval_client.eval_env import create_eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        task = validate_probe_config(config)
        registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        eval_cfg = load_yaml(str(Path(ENV_CONFIG_PATH) / "arx_x5.yml"))
        eval_cfg.update(
            task_name=task,
            num_envs=1,
            device_id=0,
            eval_batch=True,
            policy_name="RLT_Pi05",
            additional_info="transition_probe",
            seed=config["seed"],
            physx_monitor_enabled=False,
        )
        env_cfg = OmegaConf.create(
            {
                **{
                    name: load_yaml(
                        str(
                            Path(ENV_CONFIG_PATH)
                            / name
                            / (eval_cfg["config"][name] + ".yml")
                        )
                    )
                    for name in ("sim", "scene", "camera", "robot")
                },
                "task_env": load_yaml(
                    registry.task_config_path(
                        str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), task
                    )
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": {
                    "policy_name": "RLT_Pi05",
                    "port": args.port,
                    "host": "127.0.0.1",
                    "protocol": "ws",
                    "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
                    "trial_id": os.environ["ROBODOJO_RUN_ID"],
                    "action_case_id": task,
                },
            }
        )
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, _ = process_config(env_cfg, task_name=task)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0]
        env = create_eval_env(env_cfg, app)
        split = verify(args.layouts) if args.layouts else None
        if split is not None:
            if split.get("task") != task:
                raise ValueError("Generated layout task differs from probe task")
            entries = [row for row in split["layouts"] if row["id"] == args.layout_id]
            if len(entries) != 1:
                raise ValueError("Generated probe layout ID is missing or ambiguous")
            entry = entries[0]
            env.seed_manager.seed_info = {0: {"scene_layout": entry["path"]}}
            protected = frozenset(row["sha256"] for row in split["protected"])
            mode = "training" if entry["role"] == "train" else "probe"
        else:
            entry = None
            protected = frozenset()
            mode = "probe"
        layout = Path(env.seed_manager.seed_info[0]["scene_layout"])
        layout_digest = hashlib.sha256(layout.read_bytes()).hexdigest()
        if entry is not None and layout_digest != entry["sha256"]:
            raise ValueError("Generated layout bytes changed before reset")
        if entry is None:
            protected = frozenset({layout_digest})
        wrapped = RoboDojoTransitions(
            env,
            layouts={0: layout_digest},
            protected_evaluation_hashes=protected,
            mode=mode,
        )
        observation, reset_info = wrapped.reset(0)
        original_limit = env.step_lim
        env.step_lim = 3  # Explicit diagnostic horizon, not benchmark performance.
        from PIL import Image

        Image.fromarray(observation["images"]["cam_high"].transpose(1, 2, 0)).save(
            args.output / "initial.png"
        )
        records = []
        for tick in range(3):
            env.model_client.call(func_name="update_obs", obs=env.get_obs())
            commands = env.model_client.call(func_name="get_action")
            action = np.concatenate([commands[0][name] for name, _ in FIELDS])
            transition = wrapped.step(action, source="reference_teacher")
            records.append(
                {
                    "tick": tick,
                    "reward": transition.reward,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                    "pre_state": observation["state"].tolist(),
                    "post_state": transition.observation["state"].tolist(),
                    "info": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in transition.info.items()
                    },
                }
            )
            _save(args.output / "transitions.json", {"transitions": records})
            observation = transition.observation
            if transition.terminated or transition.truncated:
                break
        if not records or not (records[-1]["terminated"] or records[-1]["truncated"]):
            raise RuntimeError("Diagnostic horizon did not produce an episode boundary")
        if (
            len(records) != 3
            or records[-1]["info"]["end_reason"] != "time_limit"
            or any(record["reward"] != 0 for record in records)
        ):
            raise RuntimeError(
                "Expected three unsolved actions followed by diagnostic truncation; "
                "inspect reward setup and scene before accepting this probe"
            )
        if records[-1]["info"]["control_step"] != len(records):
            raise RuntimeError("Native command acknowledgement count differs")
        Image.fromarray(observation["images"]["cam_high"].transpose(1, 2, 0)).save(
            args.output / "final.png"
        )
        _save(
            args.output / "result.json",
            {
                "passed": True,
                "task": task,
                "reset": reset_info,
                "mode": "transition_probe",
                "training": False,
                "baseline_success_rate_evidence": False,
                "gradient_updates": 0,
                "original_native_limit": original_limit,
                "probe_limit": 3,
                "acknowledged_actions": len(records),
                "end_reason": records[-1]["info"]["end_reason"],
                "policy": "native_openpi_pi05_reference_not_alphabrain_rlt",
                "generated_layout": entry,
            },
        )
    finally:
        if env is not None:
            try:
                env.model_client.close()
            finally:
                env.close()
        app.close()


def supervise(args: argparse.Namespace, config: dict) -> None:
    """Own only the selected GPU and bounded server/client process groups."""
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise ValueError("Require operator-authorized Omniverse EULA acceptance")
    root = Path(config["robodojo_root"])
    _source_check(root, ROBO_DOJO_REVISION)
    _source_check(root / "XPolicyLab", XPOLICYLAB_REVISION)
    validate_probe_config(config)
    if args.layouts:
        split = verify(args.layouts)
        if split.get("task") != config["task"]:
            raise ValueError("Generated layout task differs from configured task")
    # Refuse a different installed policy rather than rewriting the sealed plugin.
    deploy = root / "XPolicyLab/policy/RLT_Pi05/deploy.yml"
    installed = yaml.safe_load(deploy.read_text())
    if any(installed.get(key) != value for key, value in config["policy"].items()):
        raise ValueError("Installed native policy differs from probe configuration")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    _save(
        args.output / "provenance.json",
        {
            "native_revision": ROBO_DOJO_REVISION,
            "xpolicylab_revision": XPOLICYLAB_REVISION,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "robodojo_transition.py",
                    "robodojo_transition_probe.py",
                    "robodojo_boundary.py",
                )
            },
            "expected_boundary": "time_limit_after_three_actions",
            "task": config["task"],
            "layout_manifest": str(args.layouts) if args.layouts else None,
            "layout_manifest_sha256": digest(args.layouts) if args.layouts else None,
            "layout_id": args.layout_id if args.layouts else None,
            "training": False,
        },
    )
    status = {"state": "starting", "training": False, "started_at": time.time()}
    children = []

    def save() -> None:
        status["updated_at"] = time.time()
        _save(args.output / "status.json", status)

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Signal {signum}")

    lease = Path(f"/dev/shm/rlt-research-gpu{args.gpu}.lock").open("a")
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle, detail = gpu_idle(args.gpu)
        if not idle or foreign_gpu_processes([], gpu=args.gpu):
            raise RuntimeError(f"GPU {args.gpu} is occupied: {detail}")
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
            PYTHONPATH=os.pathsep.join(
                map(
                    str,
                    (
                        Path(__file__).resolve().parents[2],
                        root,
                        root / "XPolicyLab",
                    ),
                )
            ),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            NO_PROXY="127.0.0.1,localhost",
            no_proxy="127.0.0.1,localhost",
            ROBODOJO_RUN_ID=args.output.name,
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            PYTHONUNBUFFERED="1",
        )
        with (
            (args.output / "server.log").open("x") as server_log,
            (args.output / "client.log").open("x") as client_log,
        ):
            server = subprocess.Popen(
                [
                    config["policy_python"],
                    str(root / "XPolicyLab/setup_policy_server.py"),
                    "--config_path",
                    str(deploy),
                    "--protocol",
                    "ws",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                ],
                cwd=root,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(server)
            status.update(
                state="loading_reference",
                server_pid=server.pid,
                gpu_uuid=uuid,
                renderer_gpu=renderer,
            )
            save()
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
            client = subprocess.Popen(
                [
                    config["sim_python"],
                    "-u",
                    "-m",
                    "toolkits.rlt.robodojo_transition_probe",
                    "--native-client",
                    "--config",
                    str(args.config),
                    "--task",
                    config["task"],
                    "--output",
                    str(args.output),
                    "--port",
                    str(args.port),
                    "--renderer-index",
                    str(renderer),
                    *(
                        (
                            "--layouts",
                            str(args.layouts),
                            "--layout-id",
                            str(args.layout_id),
                        )
                        if args.layouts
                        else ()
                    ),
                ],
                cwd=root,
                env=environment,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(client)
            status.update(state="running_simulator", client_pid=client.pid)
            save()
            deadline = time.monotonic() + 600
            while client.poll() is None:
                server_code = server.poll()
                timed_out = time.monotonic() > deadline
                foreign = foreign_gpu_processes(children, gpu=args.gpu)
                if server_code is not None or timed_out or foreign:
                    raise RuntimeError(
                        "Simulator stopped unexpectedly: "
                        f"server_returncode={server_code}, "
                        f"timed_out={timed_out}, foreign_pids={foreign}"
                    )
                if (args.output / "STOP").exists():
                    raise KeyboardInterrupt("Operator STOP")
                time.sleep(2)
            result = json.loads((args.output / "result.json").read_text())
            if client.returncode or result.get("passed") is not True:
                raise RuntimeError("Simulator probe lacks a successful explicit result")
            status.update(state="passed", result=result)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child in reversed(children):
            stop_group(child)
        lease.close()
        status["finished_at"] = time.time()
        save()
        signal.signal(signal.SIGTERM, previous)


def main() -> None:
    """Select the isolated simulator client or its GPU-owning supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=6133)
    parser.add_argument("--gpu", type=int, choices=range(3), default=2)
    parser.add_argument("--native-client", action="store_true")
    parser.add_argument("--layouts", type=Path)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--renderer-index", type=int)
    parser.add_argument("--renderer-table", type=Path)
    parser.add_argument("--sim-library-path", type=Path)
    parser.add_argument("--vulkan-icd", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.task:
        config = dict(config, task=args.task)
    if not 1 <= args.port <= 65535 or not args.output.is_absolute():
        parser.error("Require a valid port and absolute output path")
    if args.layout_id < 0 or (args.layouts is None and args.layout_id != 0):
        parser.error("Generated layout IDs require a manifest")
    if not args.native_client and not all(
        (args.renderer_table, args.sim_library_path, args.vulkan_icd)
    ):
        parser.error(
            "Supervisor requires renderer table, simulator libraries and Vulkan ICD"
        )
    if args.native_client:
        native_probe(args, config)
    else:
        supervise(args, config)


if __name__ == "__main__":
    main()
