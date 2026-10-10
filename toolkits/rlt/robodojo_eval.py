# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and audit native RoboDojo evaluations without starting a policy or GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from toolkits.rlt.robodojo_boundary import (
    ROBO_DOJO_REVISION,
    XPOLICYLAB_REVISION,
    RoboDojoBoundary,
)
from toolkits.rlt.robodojo_policy import validate_normalization

SCHEMA = "robodojo-eval-v1"
POLICY_NAME = "RLT_Pi05"
NATIVE_REVISIONS = {
    "third_party/IsaacLab": "afca7b09d60d8beb9c1cb28b43066499940b969b",
    "third_party/curobo": "d17b54ce32cba095c0b000c4c58777075d11de0e",
}
NATIVE_DEPENDENCIES = {
    "sim_python": (
        "isaacsim",
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "nvidia-curobo",
        "torch",
        "numpy",
        "websockets",
        "open3d",
    ),
    "policy_python": (
        "openpi",
        "openpi-client",
        "jax",
        "jaxlib",
        "flax",
        "orbax-checkpoint",
        "torch",
        "transformers",
        "lerobot",
        "numpy",
        "websockets",
        "pytest",
        "chex",
        "optax",
        "pyarrow",
    ),
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    """Read a JSON object, rejecting duplicate keys instead of losing evidence."""
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_check(root: Path, revision: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != revision:
        raise ValueError(f"Source revision mismatch: {root}: {head} != {revision}")
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            ".",
            ":(exclude)XPolicyLab",
        ],
        check=True,
        capture_output=True,
    )


def validate_config(config: dict) -> None:
    """Reject ambiguous task, budget, identity, and model-interface declarations."""
    for field in ("task", "run_id", "group"):
        if not isinstance(config.get(field), str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", config[field]
        ):
            raise ValueError(f"Invalid {field}")
    for field, minimum in (
        ("episodes", 1),
        ("num_envs", 1),
        ("port", 1),
        ("gpu", 0),
        ("seed", 0),
    ):
        if type(config.get(field)) is not int or config[field] < minimum:
            raise ValueError(f"Invalid {field}")
    if config["port"] > 65535 or config["num_envs"] > config["episodes"]:
        raise ValueError("Invalid port or evaluation batch size")
    for field in ("robodojo_root", "sim_python", "policy_python"):
        if not Path(config[field]).is_absolute():
            raise ValueError(f"Use an absolute path for {field}")
    policy = config["policy"]
    if (
        policy.get("policy_name"),
        policy.get("env_cfg_type"),
        policy.get("action_type"),
    ) != (POLICY_NAME, "arx_x5", "joint"):
        raise ValueError("Require RLT_Pi05 with the arx_x5 joint interface")
    RoboDojoBoundary(
        lambda obs: None,
        model_contract=policy["model_contract"],
        horizon=policy["action_horizon"],
    )
    norm_id = policy["model_contract"]["normalization_id"]
    if (
        not isinstance(norm_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", norm_id)
        or norm_id in {".", ".."}
    ):
        raise ValueError("normalization_id must be one checkpoint asset directory")
    if not Path(policy["checkpoint_path"]).is_absolute():
        raise ValueError("checkpoint_path must be an absolute exact checkpoint path")
    if policy["backend"] not in {"unconfigured", "openpi_reference", "factory"}:
        raise ValueError("Unknown backend")
    if policy["backend"] == "openpi_reference" and config["group"] != "reference":
        raise ValueError("An OpenPI reference must not be labelled an RLT result")


def _layout_inventory(config: dict) -> list[dict]:
    root, task = Path(config["robodojo_root"]), config["task"]
    task_table = yaml.safe_load((root / "task/RoboDojo/config/_task.yml").read_text())
    if not (root / f"task/RoboDojo/config/{task}.yml").is_file():
        raise ValueError(f"Unknown task config: {task}")
    options = task_table["common"] | task_table.get("tasks", {}).get(task, {})
    if options["robot_config"] != "dual_x5":
        raise ValueError("Task requires a different embodiment")
    if config["episodes"] > options.get("eval_nums", 50):
        raise ValueError(
            "Requested episodes exceed the native task cap; do not silently truncate"
        )
    directory = root / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(config["seed"])
    pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
    files = sorted(
        (p for p in directory.iterdir() if pattern.fullmatch(p.name)),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    suffixes = [int(p.stem.rsplit("_", 1)[1]) for p in files]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("Ambiguous numeric layout ordering")
    if len(files) < config["episodes"]:
        raise ValueError(
            f"Only {len(files)} native layouts for {config['episodes']} episodes"
        )
    # Native layout IDs are ordinal positions, NOT the filename's numeric suffix.
    return [
        {"layout_id": i, "path": str(path), "sha256": _digest(path)}
        for i, path in enumerate(files[: config["episodes"]])
    ]


def _commands(config: dict) -> dict:
    root = Path(config["robodojo_root"])
    project = Path(__file__).resolve().parents[2]
    deploy = root / "XPolicyLab/policy" / POLICY_NAME / "deploy.yml"
    environment = {
        "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
        "PYTHONPATH": os.pathsep.join(map(str, (project, root, root / "XPolicyLab"))),
        "ROBODOJO_RUN_ID": config["run_id"],
        "EVAL_NUM": str(config["episodes"]),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    return {
        "cwd": str(root),
        "env": environment,
        "server_argv": [
            config["policy_python"],
            str(root / "XPolicyLab/setup_policy_server.py"),
            "--config_path",
            str(deploy),
            "--protocol",
            "ws",
            "--host",
            "127.0.0.1",
            "--port",
            str(config["port"]),
        ],
        "client_argv": [
            config["sim_python"],
            str(root / "src/eval_client/main.py"),
            "--task_name",
            config["task"],
            "--env_cfg_type",
            "arx_x5",
            "--device_id",
            "0",
            "--num_envs",
            str(config["num_envs"]),
            "--policy_name",
            POLICY_NAME,
            "--seed",
            str(config["seed"]),
            "--port",
            str(config["port"]),
            "--host",
            "127.0.0.1",
            "--protocol",
            "ws",
            "--additional_info",
            config["group"],
            "--headless",
            "--enable_cameras",
        ],
    }


def _plugin_files(config: dict) -> dict[str, str]:
    deploy = dict(
        config["policy"],
        task_name=config["task"],
        bench_name="RoboDojo",
        seed=config["seed"],
        eval_batch=True,
        protocol="ws",
        host="127.0.0.1",
        port=config["port"],
    )
    return {
        "__init__.py": "",
        "model.py": "from toolkits.rlt.robodojo_policy import Model\n",
        "deploy.py": "from toolkits.rlt.robodojo_policy import eval_one_episode, eval_one_episode_batch\n",
        "deploy.yml": yaml.safe_dump(deploy, sort_keys=False),
    }


def prepare(config: dict) -> dict:
    """Inventory fixed layouts and prerequisites; never import a model or simulator.

    ``prepared`` means filesystem checks passed, not that Isaac or the policy has
    run. Commands are returned as data only. Missing assets yield blocked reports.
    """
    validate_config(config)
    root = Path(config["robodojo_root"])
    errors, layouts = [], []
    for source, revision in (
        (root, ROBO_DOJO_REVISION),
        (root / "XPolicyLab", XPOLICYLAB_REVISION),
    ):
        try:
            _source_check(source, revision)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(f"Source check: {exc}")
    try:
        layouts = _layout_inventory(config)
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"Fixed layouts: {exc}")
    for relative in ("Assets/Robots", "Assets/Object/RoboDojo"):
        path = root / relative
        if not path.is_dir() or not any(path.iterdir()):
            errors.append(f"Missing assets: {path}")
    for field in ("sim_python", "policy_python"):
        if not Path(config[field]).is_file() or not os.access(config[field], os.X_OK):
            errors.append(f"Missing executable {field}: {config[field]}")
    policy = config["policy"]
    artifact_report = None
    if config.get("artifact_download_report"):
        try:
            artifact_report = read_json(Path(config["artifact_download_report"]))
            if artifact_report.get("complete") is not True or artifact_report.get(
                "errors"
            ):
                errors.append("Artifact download is incomplete or has integrity errors")
        except (OSError, ValueError) as exc:
            errors.append(f"Artifact download report: {exc}")
    norm_file = (
        Path(policy["checkpoint_path"])
        / "assets"
        / policy["model_contract"]["normalization_id"]
        / "norm_stats.json"
    )
    normalization = {
        "path": str(norm_file),
        "sha256": _digest(norm_file) if norm_file.is_file() else None,
    }
    if normalization["sha256"] is None:
        errors.append(f"Missing checkpoint normalization: {norm_file}")
    else:
        try:
            validate_normalization(norm_file)
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"Checkpoint normalization: {exc}")
    if policy["backend"] == "unconfigured":
        errors.append("No compatible ARX X5 backend selected")
    if policy["backend"] == "factory" and not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*",
        policy.get("backend_factory") or "",
    ):
        errors.append("Require a trusted backend_factory module:function")
    if policy["backend"] == "openpi_reference":
        if not policy.get("train_config_name"):
            errors.append("Require an explicit OpenPI training config name")
        checkpoint = Path(policy["checkpoint_path"])
        if (
            not (checkpoint / "params").is_dir()
            and not (checkpoint / "model.safetensors").is_file()
        ):
            errors.append("No native OpenPI weights in the exact checkpoint directory")
    result_path = (
        root
        / "eval_result/RoboDojo"
        / config["task"]
        / POLICY_NAME
        / "arx_x5"
        / f"{config['seed']}_{config['group']}"
        / config["run_id"]
        / "_result.json"
    )
    resume_path = result_path.parent.parent / f"_resume_{config['run_id']}.json"
    if result_path.parent.exists() or resume_path.exists():
        errors.append("Run ID already exists; use a new ID, not an implicit resume")
    return {
        "schema": SCHEMA,
        "status": "blocked" if errors else "prepared",
        "errors": errors,
        "config": copy.deepcopy(config),
        "layouts": layouts,
        "normalization": normalization,
        "artifact_download": {
            "report_path": config.get("artifact_download_report"),
            "complete": artifact_report.get("complete") if artifact_report else None,
        },
        "source_revisions": {
            "robodojo": ROBO_DOJO_REVISION,
            "xpolicylab": XPOLICYLAB_REVISION,
        },
        "adapter_sha256": {
            name: _digest(Path(__file__).with_name(name))
            for name in (
                "robodojo_boundary.py",
                "robodojo_policy.py",
                "robodojo_eval.py",
            )
        },
        "runtime_verified": False,
        "assistance": "none",
        "memory": False,
        "expected_result": str(result_path),
        "commands": _commands(config),
    }


def install_policy(config: dict) -> Path:
    """Create a new untracked XPolicyLab plugin, refusing all existing directories."""
    validate_config(config)
    root = Path(config["robodojo_root"])
    _source_check(root, ROBO_DOJO_REVISION)
    _source_check(root / "XPolicyLab", XPOLICYLAB_REVISION)
    target = root / "XPolicyLab/policy" / POLICY_NAME
    # mkdir is exclusive: a prior policy or checkpoint is never overwritten.
    target.mkdir()
    for name, content in _plugin_files(config).items():
        (target / name).write_text(content)
    return target


def runtime_inventory(config: dict) -> dict:
    """Inspect isolated Python package metadata, never import or launch Isaac.

    Package presence is a provisioning gate only. Driver, rendering, CUDA kernel,
    model loading, and actual episode checks still require a scheduled GPU run.
    """
    validate_config(config)
    report = {
        "status": "packages_present",
        "runtime_verified": False,
        "interpreters": {},
        "errors": [],
    }
    report["source_revisions"] = dict(NATIVE_REVISIONS)
    for relative, revision in NATIVE_REVISIONS.items():
        try:
            _source_check(Path(config["robodojo_root"]) / relative, revision)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["errors"].append(f"{relative}: {error}")
    program = (
        "import json, sys; from importlib import metadata\n"
        "versions = {}\n"
        "for name in sys.argv[1:]:\n"
        " try: versions[name] = metadata.version(name)\n"
        " except metadata.PackageNotFoundError: versions[name] = None\n"
        "print(json.dumps({'python': list(sys.version_info[:3]), 'packages': versions}))\n"
    )
    for field, packages in NATIVE_DEPENDENCIES.items():
        try:
            result = subprocess.run(
                [config[field], "-I", "-B", "-c", program, *packages],
                env=dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"
                ),
                check=True,
                text=True,
                capture_output=True,
                timeout=30,
            )
            inventory = json.loads(result.stdout)
            report["interpreters"][field] = inventory
            for name, version in inventory["packages"].items():
                if version is None:
                    report["errors"].append(f"{field}: missing {name}")
            if inventory["python"][:2] != [3, 11]:
                report["errors"].append(f"{field}: require Python 3.11")
            if field == "sim_python" and inventory["packages"]["isaacsim"] not in {
                None,
                "5.1.0.0",
                "5.1.0",
            }:
                report["errors"].append("Require benchmark-pinned Isaac Sim 5.1.0")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["errors"].append(f"{field}: {error}")
    if report["errors"]:
        report["status"] = "blocked"
    return report


def probe_reference(config: dict) -> dict:
    """Test policy imports and native configuration in a bounded CPU subprocess.

    Unlike package inventory, this detects import-time ABI and dependency errors.
    It never creates a backend, restores weights, or starts an Isaac process.
    """
    validate_config(config)
    report = {
        "status": "blocked",
        "runtime_verified": False,
        "gpu_started": False,
        "parameters_loaded": False,
        "errors": [],
    }
    if config["policy"]["backend"] != "openpi_reference":
        report["errors"].append("CPU reference probe requires openpi_reference")
        return report
    project = str(Path(__file__).resolve().parents[2])
    program = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "from toolkits.rlt.robodojo_policy import inspect_reference; "
        "print(json.dumps(inspect_reference(json.load(sys.stdin))))"
    )
    try:
        result = subprocess.run(
            [config["policy_python"], "-I", "-B", "-c", program, project],
            input=json.dumps(config["policy"]),
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                JAX_PLATFORMS="cpu",
                PYTHONDONTWRITEBYTECODE="1",
            ),
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode:
            report["errors"].append(
                f"Policy import/config exit {result.returncode}: {result.stderr[-6000:]}"
            )
        else:
            report["reference"] = json.loads(result.stdout)
            report["status"] = "reference_imports_passed"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["errors"].append(f"Reference probe: {error}")
    return report


def audit(plan: dict, result_path: Path) -> dict:
    """Check one native result against exact planned layouts, not a latest-file glob.

    Missing results or planned layouts are incomplete. Unexpected/duplicate IDs,
    changed layouts or inconsistent aggregates fail and suppress comparable scores.
    This audits recorded consistency, not cryptographic checkpoint provenance.
    """
    if plan.get("schema") != SCHEMA or plan.get("status") != "prepared":
        raise ValueError("Audit requires a prepared evaluation manifest")
    validate_config(plan["config"])
    if result_path.resolve() != Path(plan["expected_result"]).resolve():
        raise ValueError("Result path differs from the planned run identity")
    expected = [row["layout_id"] for row in plan["layouts"]]
    if expected != list(range(plan["config"]["episodes"])):
        raise ValueError("Manifest does not contain the exact planned layout budget")
    report = {
        "status": "incomplete",
        "expected_episodes": len(expected),
        "completed_episodes": 0,
        "success_rate": None,
        "partial_score_percent": None,
        "errors": [],
    }
    for row in [*plan["layouts"], plan["normalization"]]:
        path = Path(row["path"])
        if not path.is_file() or _digest(path) != row["sha256"]:
            report["errors"].append(f"Changed or missing input: {path}")
    if not result_path.is_file():
        report["status"] = "failed" if report["errors"] else "incomplete"
        return report
    root = Path(plan["config"]["robodojo_root"])
    for source, revision in (
        (root, ROBO_DOJO_REVISION),
        (root / "XPolicyLab", XPOLICYLAB_REVISION),
    ):
        try:
            _source_check(source, revision)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report["errors"].append(f"Source check: {exc}")
    for name, digest in plan["adapter_sha256"].items():
        if Path(name).name != name or _digest(Path(__file__).with_name(name)) != digest:
            report["errors"].append(f"Changed adapter: {name}")
    plugin = root / "XPolicyLab/policy" / POLICY_NAME
    for name, content in _plugin_files(plan["config"]).items():
        if not (plugin / name).is_file() or (plugin / name).read_text() != content:
            report["errors"].append(f"Installed policy differs from the plan: {name}")
    try:
        current = _layout_inventory(plan["config"])
        if [(x["layout_id"], x["path"]) for x in current] != [
            (x["layout_id"], x["path"]) for x in plan["layouts"]
        ]:
            report["errors"].append("Native layout ordering changed after preparation")
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(f"Fixed layouts: {exc}")
    data = read_json(result_path)
    details = data.get("details")
    if not isinstance(details, dict):
        raise ValueError("Native result details must be an object")
    seen, successes, total_score = set(), 0, 0.0
    for row in details.values():
        if not isinstance(row, dict):
            report["errors"].append("Invalid episode record")
            continue
        index, success, score = (
            row.get("layout_id"),
            row.get("success"),
            row.get("score"),
        )
        if type(index) is not int or index not in expected or index in seen:
            report["errors"].append(f"Unexpected or duplicate layout_id: {index}")
            continue
        if (
            type(success) is not bool
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            report["errors"].append(f"Invalid outcome for layout_id: {index}")
            continue
        if success and score != 1:
            report["errors"].append(
                f"Successful episode has a non-unit native score: {index}"
            )
        seen.add(index)
        successes += int(success)
        total_score += score
    count = len(details)
    report["completed_episodes"] = count
    if type(data.get("eval_time")) is not int or data["eval_time"] != count:
        report["errors"].append("Native eval_time does not match details")
    for key, value in (
        ("success_rate", successes / count if count else 0),
        ("score", 100 * total_score / count if count else 0),
    ):
        declared = data.get(key)
        if (
            type(declared) not in (int, float)
            or not math.isfinite(declared)
            or not math.isclose(declared, value, rel_tol=1e-6, abs_tol=1e-6)
        ):
            report["errors"].append(f"Inconsistent native {key}")
    if report["errors"]:
        report["status"] = "failed"
    elif seen == set(expected) and count == len(expected):
        report.update(
            status="passed",
            success_rate=successes / count,
            partial_score_percent=100 * total_score / count,
        )
    return report


def main() -> int:
    """Expose CPU preparation, explicit plugin installation, and result auditing."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "runtime", "probe-reference", "install-policy"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "install-policy":
            command.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("audit")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "install-policy":
        print(install_policy(yaml.safe_load(args.config.read_text())))
        return 0
    if args.command in {"prepare", "runtime", "probe-reference"}:
        function = {
            "prepare": prepare,
            "runtime": runtime_inventory,
            "probe-reference": probe_reference,
        }[args.command]
        result = function(yaml.safe_load(args.config.read_text()))
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        result = audit(read_json(args.plan), args.result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return {
        "prepared": 0,
        "packages_present": 0,
        "reference_imports_passed": 0,
        "passed": 0,
        "failed": 2,
        "blocked": 3,
        "incomplete": 3,
    }[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
