# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run a bounded sealed native reference evaluation on GPU 2, never train."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import TextIO

from toolkits.rlt import robodojo_eval as evaluation
from toolkits.rlt.experiment_queue import gpu_idle, stop_group
from toolkits.rlt.robodojo_resume import audit_continuation, verify_resume


def verify_result(plan: dict, receipt: Path | None = None) -> dict:
    """Require the sealed episode result; Kit may swallow Python exit errors."""
    report = (
        audit_continuation(receipt)
        if receipt
        else evaluation.audit(plan, Path(plan["expected_result"]))
    )
    if report["status"] != "passed":
        raise RuntimeError(f"Native result audit did not pass: {report}")
    return report


def renderer_index(summary: str, uuid: str) -> int:
    """Use Kit's GPU table; its order differs from standalone vulkaninfo."""
    matches = set()
    for index, body in re.findall(r"^\| (\d+)\s+\|([^\n]*\n[^\n]*)", summary, re.M):
        found = re.search(r"\|\s*([a-fA-F0-9]{8})\.\.\s*\|", body)
        if found and uuid.removeprefix("GPU-").lower().startswith(
            found.group(1).lower()
        ):
            matches.add(int(index))
    if len(matches) != 1:
        raise ValueError(f"Kit UUID mapping is missing or ambiguous: {uuid}")
    return matches.pop()


def owns_listener(pid: int, port: int) -> bool:
    """Check the listener inode belongs to this server, not another service."""
    sockets = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
            if target.startswith("socket:["):
                sockets.add(target[8:-1])
        except OSError:
            continue
    for row in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = row.split()
        if (
            int(fields[1].split(":")[1], 16) == port
            and fields[3] == "0A"
            and fields[9] in sockets
        ):
            return True
    return False


def foreign_gpu_processes(children: list[subprocess.Popen], gpu: int = 2) -> list[int]:
    """Include graphics processes and allow only our fresh process groups."""
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    snapshot = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    groups = {child.pid for child in children}
    foreign = []
    for row in snapshot.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2 or fields[0] != uuid:
            continue
        pid = int(fields[1])
        try:
            if os.getpgid(pid) not in groups:
                foreign.append(pid)
        except ProcessLookupError:
            continue
    return foreign


def validate_launch(plan_path: Path, receipt: Path | None = None) -> None:
    """Require a fresh sealed plan or an unchanged explicit continuation receipt."""
    plan = evaluation.read_json(plan_path)
    config = plan["config"]
    if (
        plan["status"] != "prepared"
        or config["gpu"] != 2
        or not 1 <= config["episodes"] <= 25
        or config["num_envs"] != 1
        or config["group"] != "reference"
        or config["policy"]["backend"] != "openpi_reference"
    ):
        raise ValueError("Runner requires one native reference environment on GPU 2")
    if receipt:
        saved = evaluation.read_json(receipt)["snapshots"]["plan"]
        if Path(saved["path"]).resolve() != plan_path.resolve():
            raise ValueError("Receipt belongs to a different plan")
        verify_resume(receipt)
    elif evaluation.prepare(config) != plan:
        raise ValueError(
            "Fresh plan changed or result exists; require explicit recovery"
        )
    plugin = Path(config["robodojo_root"]) / "XPolicyLab/policy/RLT_Pi05"
    for name, content in evaluation._plugin_files(config).items():
        if (plugin / name).read_text() != content:
            raise ValueError(f"Policy plugin differs from plan: {name}")


def try_acquire_lease(lease: TextIO) -> bool:
    """Acquire one advisory GPU lease without blocking or launching a child."""
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def check_stop(path: Path) -> None:
    """Stop waiting or running when the operator creates the explicit stop file."""
    if path.exists():
        raise KeyboardInterrupt(f"Stop requested: {path}")


def main() -> None:
    """Wait for exclusive GPU 2 ownership and execute one audited reference run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--lease-directory", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--sim-library-path", type=Path)
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--vulkaninfo", type=Path, required=True)
    parser.add_argument("--renderer-table", type=Path, required=True)
    parser.add_argument("--client-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    plan = evaluation.read_json(args.plan)
    plan_digest = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    output = args.output
    validate_launch(args.plan, args.resume_receipt)
    if not 60 <= args.client_timeout_seconds <= 10800:
        raise ValueError("Client timeout must be between 60 and 10800 seconds")
    if args.sim_library_path and not args.sim_library_path.is_dir():
        raise ValueError("Simulator library directory does not exist")
    if args.vulkan_icd and not args.vulkan_icd.is_file():
        raise ValueError("Vulkan ICD file does not exist")
    if not args.vulkaninfo.is_file() or not os.access(args.vulkaninfo, os.X_OK):
        raise ValueError("Vulkan inventory executable does not exist")
    if not args.renderer_table.is_file():
        raise ValueError("Require a verified Kit renderer table for this host")
    # Refuse an occupied endpoint (for example SSH X11 forwarding on port 6010).
    # A listening socket alone must not be mistaken for our model being ready.
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", plan["config"]["port"]))
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise ValueError("Require operator-authorized EULA acceptance")
    output.mkdir()
    status = {
        "started_at": time.time(),
        "state": "starting",
        "training": False,
        "eula_authorized_by_user": True,
        "plan": str(args.plan),
        "plan_sha256": plan_digest,
        "expected_episodes": plan["config"]["episodes"],
        "client_timeout_seconds": args.client_timeout_seconds,
    }
    children = []
    stop_file = args.stop_file or output / "STOP"
    status.update(
        resume_receipt=str(args.resume_receipt) if args.resume_receipt else None,
        stop_file=str(stop_file),
        wait_deadline=None,
    )

    def save():
        status["updated_at"] = time.time()
        pending = output / "status.tmp.json"
        pending.write_text(json.dumps(status, indent=2) + "\n")
        pending.replace(output / "status.json")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (args.lease_directory / "rlt-research-gpu2.lock").open("a") as lease:
            while not try_acquire_lease(lease):
                check_stop(stop_file)
                status.update(state="waiting_lease")
                save()
                if not args.wait_idle:
                    raise RuntimeError("Another queue owns the GPU 2 lease")
                time.sleep(5)
            idle_checks = 0
            while idle_checks < 2:
                check_stop(stop_file)
                idle, detail = gpu_idle(2)
                idle = idle and not foreign_gpu_processes([])
                idle_checks = idle_checks + 1 if idle else 0
                status.update(
                    state="waiting_gpu", gpu_status=detail, idle_checks=idle_checks
                )
                save()
                if not idle and not args.wait_idle:
                    raise RuntimeError("GPU 2 acquired by another process")
                if idle_checks < 2:
                    time.sleep(5)
            validate_launch(args.plan, args.resume_receipt)
            if hashlib.sha256(args.plan.read_bytes()).hexdigest() != plan_digest:
                raise ValueError("Plan changed while waiting for the GPU")
            check_stop(stop_file)
            commands = plan["commands"]
            environment = dict(os.environ, **commands["env"])
            environment.pop("LD_PRELOAD", None)
            environment.update(
                PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
            )
            with (
                (output / "server.log").open("x") as server_log,
                (output / "client.log").open("x") as client_log,
            ):
                try:
                    server = subprocess.Popen(
                        commands["server_argv"],
                        cwd=commands["cwd"],
                        env=environment,
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    children.append(server)
                    status.update(state="loading_reference", server_pid=server.pid)
                    save()
                    deadline = time.monotonic() + 900
                    while True:
                        check_stop(stop_file)
                        if server.poll() is not None:
                            raise RuntimeError(f"Reference exited: {server.returncode}")
                        foreign = foreign_gpu_processes(children)
                        if foreign:
                            raise RuntimeError(
                                f"GPU 2 acquired concurrently by external processes: {foreign}"
                            )
                        try:
                            if owns_listener(server.pid, plan["config"]["port"]):
                                break
                            raise ConnectionRefusedError("Server is not listening yet")
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "Reference server startup exceeded 900 s"
                                )
                            time.sleep(2)
                    sim_environment = environment.copy()
                    if args.sim_library_path:
                        sim_environment["LD_LIBRARY_PATH"] = str(args.sim_library_path)
                    if args.vulkan_icd:
                        sim_environment["VK_ICD_FILENAMES"] = str(args.vulkan_icd)
                        sim_environment["VK_DRIVER_FILES"] = str(args.vulkan_icd)
                    status["simulator_env_overrides"] = {
                        key: value
                        for key, value in sim_environment.items()
                        if environment.get(key) != value
                    }
                    summary = subprocess.check_output(
                        [str(args.vulkaninfo), "--summary"],
                        env=sim_environment,
                        text=True,
                        stderr=subprocess.PIPE,
                        timeout=30,
                    )
                    gpu_uuid = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            "2",
                            "--query-gpu=uuid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    ).strip()
                    if gpu_uuid.removeprefix("GPU-").lower() not in summary.lower():
                        raise ValueError(
                            "Requested GPU missing from Vulkan enumeration"
                        )
                    index = renderer_index(args.renderer_table.read_text(), gpu_uuid)
                    client_argv = list(commands["client_argv"])
                    position = client_argv.index("--kit_args") + 1
                    client_argv[position], replaced = re.subn(
                        r"--/renderer/activeGpu=\d+",
                        f"--/renderer/activeGpu={index}",
                        client_argv[position],
                    )
                    if replaced != 1:
                        raise ValueError(
                            "Require exactly one explicit renderer GPU setting"
                        )
                    status.update(
                        renderer_gpu=index,
                        gpu_uuid=gpu_uuid,
                        client_argv=client_argv,
                        renderer_table=str(args.renderer_table),
                    )
                    client = subprocess.Popen(
                        client_argv,
                        cwd=commands["cwd"],
                        env=sim_environment,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    children.append(client)
                    status.update(state="running_simulator", client_pid=client.pid)
                    save()
                    deadline = time.monotonic() + args.client_timeout_seconds
                    while client.poll() is None:
                        check_stop(stop_file)
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"Reference exited during evaluation: {server.returncode}"
                            )
                        foreign = foreign_gpu_processes(children)
                        if foreign:
                            raise RuntimeError(
                                f"GPU 2 acquired concurrently by external processes: {foreign}"
                            )
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Native evaluation exceeded {args.client_timeout_seconds} seconds"
                            )
                        time.sleep(5)
                    code = client.returncode
                    status.update(
                        state="client_exited",
                        client_exit_code=code,
                        expected_result=plan["expected_result"],
                    )
                    if code != 0:
                        raise RuntimeError(f"Simulator exited: {code}")
                    status.update(
                        state="completed",
                        result_audit=verify_result(plan, args.resume_receipt),
                    )
                finally:
                    # Release owned processes before closing the GPU lease.
                    for child in reversed(children):
                        stop_group(child)
    except BaseException as error:
        status.update(
            state="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        status["finished_at"] = time.time()
        save()
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
