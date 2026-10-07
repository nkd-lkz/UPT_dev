# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Inspect release integrity or replay recorded actions without loading a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

from toolkits.rlt.libero_audit import file_hash, observation_record
from toolkits.rlt.libero_reproduction import (
    ASSETS,
    check_source,
    episode_plan,
    paired_summary,
)


def compare_preprocessing(baseline: Path, candidate: Path) -> dict:
    """Compare complete matched evaluations, separating the inspected pilot cases."""
    manifests = [
        json.loads((p / "manifest.json").read_text()) for p in (baseline, candidate)
    ]
    for manifest in manifests:
        if (
            manifest.get("complete") is not True
            or manifest.get("mode") != "evaluate"
            or manifest.get("assistance") is not False
            or manifest.get("learner_dir") is not None
        ):
            raise ValueError(
                "Require complete unassisted public-checkpoint evaluations"
            )
    for key in (
        "source_revision",
        "assets",
        "tasks",
        "states",
        "seed",
        "variant",
        "runtime_versions",
    ):
        if key not in manifests[0] or manifests[0][key] != manifests[1].get(key):
            raise ValueError(f"Unmatched evaluation contract: {key}")
    if (
        manifests[0].get("input_image_size") is not None
        or manifests[1].get("input_image_size") != 224
    ):
        raise ValueError(
            "Require the original path versus the explicit 224-pixel diagnostic"
        )
    plan = set(episode_plan(manifests[0]["tasks"], manifests[0]["states"]))
    result = {"baseline": str(baseline), "candidate": str(candidate), "arms": {}}
    pilot = {(task, state) for task in (5, 6, 9) for state in (0, 1, 2)}
    for arm in ("reference", "rlt_a"):
        rows = [
            json.loads((p / f"{arm}.json").read_text()) for p in (baseline, candidate)
        ]
        paired_summary(*rows)  # Validate identity uniqueness and boolean outcomes.
        if {(r["task"], r["state"]) for r in rows[0]} != plan:
            raise ValueError(
                "Completed results do not cover the declared task/state plan"
            )
        views = {
            "all": plan,
            "inspected_pilot": plan & pilot,
            "outside_pilot": plan - pilot,
        }
        arm_result = {}
        for name, cases in views.items():
            if not cases:
                continue
            paired = paired_summary(
                *[
                    [r for r in side if (r["task"], r["state"]) in cases]
                    for side in rows
                ]
            )
            arm_result[name] = {
                "episodes": paired["episodes_per_arm"],
                "original_success": paired["reference_success"],
                "resized_success": paired["rlt_a_success"],
                "new_successes": paired["rlt_a_only_successes"],
                "lost_successes": paired["reference_only_successes"],
            }
        before = {(r["task"], r["state"]): r["success"] for r in rows[0]}
        arm_result["changed_cases"] = [
            {**r, "original_success": before[r["task"], r["state"]]}
            for r in rows[1]
            if r["success"] != before[r["task"], r["state"]]
        ]
        result["arms"][arm] = arm_result
    result["limitation"] = (
        "Fixed-weight preprocessing diagnostic, not RL training benefit or a reproduction of the reported 500-episode score. Outside-pilot cases are still public benchmark states, not unseen test data."
    )
    return result


def dependency_conflicts(package: str) -> list[dict]:
    """Return unsatisfied active requirements in this interpreter's search path."""
    conflicts = []
    for text in metadata.requires(package) or []:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed is None or installed not in requirement.specifier:
            conflicts.append({"requirement": text, "installed": installed})
    return conflicts


def git_blob_hash(path: Path) -> str:
    """Hash an ordinary Git blob, including the size header used by the Hub."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_integrity(storage: Path) -> list[dict]:
    """Compare every pinned release file against public Hub metadata."""
    rows = []
    for arm, (repo, revision) in ASSETS.items():
        url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            info = json.load(response)
        if info["sha"] != revision:
            raise ValueError(f"Hub returned a different revision for {repo}")
        for entry in info["siblings"]:
            name = entry["rfilename"]
            if name == ".gitattributes":
                continue
            path = storage / arm / name
            lfs = entry.get("lfs")
            expected = lfs["sha256"] if lfs else entry["blobId"]
            actual = (
                (file_hash(path) if lfs else git_blob_hash(path))
                if path.is_file()
                else None
            )
            rows.append(
                {
                    "arm": arm,
                    "file": name,
                    "revision": revision,
                    "expected": expected,
                    "actual": actual,
                    "match": actual == expected,
                }
            )
    return rows


def libero_tree_integrity(source: Path) -> dict:
    """Compare installed LIBERO source and assets with a local upstream Git tree."""
    from libero.libero import get_libero_path

    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "HEAD", "libero/libero"],
        text=True,
    )
    root = Path(get_libero_path("benchmark_root"))
    locations = {
        "assets": Path(get_libero_path("assets")),
        "bddl_files": Path(get_libero_path("bddl_files")),
        "init_files": Path(get_libero_path("init_states")),
    }
    counts, differences = {}, []
    for row in tree.splitlines():
        meta, name = row.split("\t")
        relative = Path(name).relative_to("libero/libero")
        category = relative.parts[0]
        if category in locations:
            local = locations[category] / Path(*relative.parts[1:])
        else:
            local = root / relative
            category = "source"
        expected = meta.split()[2]
        actual = git_blob_hash(local) if local.is_file() else None
        status = "match" if actual == expected else "different_or_missing"
        key = f"{category}/{status}"
        counts[key] = counts.get(key, 0) + 1
        if actual != expected:
            differences.append({"file": name, "expected": expected, "actual": actual})
    return {"upstream_revision": revision, "counts": counts, "differences": differences}


def recorded_episode(path: Path) -> tuple[dict, list[dict]]:
    """Require a complete, finite 7-D action stream before simulating it."""
    import numpy as np

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or rows[0].get("kind") != "reset" or rows[-1].get("kind") != "close":
        raise ValueError("Replay requires a complete reset-to-close trace")
    reset = rows[0]
    episode_plan([reset["task"]], [reset["initial_state"]])
    steps = [row for row in rows if row.get("kind") == "step"]
    if not steps or rows[-1].get("ticks") != len(steps):
        raise ValueError("Missing steps in recorded episode")
    for tick, row in enumerate(steps):
        action = np.asarray(row["action"], dtype=float)
        if row["tick"] != tick or action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(
                "Actions must have consecutive ticks and seven finite values"
            )
    return reset, steps


def replay(trace: Path, output: Path, snapshot_ticks: set[int] | None = None) -> dict:
    """Replay executed commands; expose full physics state and both camera hashes.

    This is open-loop simulation sensitivity, not a policy success-rate test.
    Continue the recorded action stream even if the new simulator succeeds early.
    No model, replay buffer, optimizer or training data is modified.
    """
    import numpy as np
    from AlphaBrain.training.reinforcement_learning.envs.libero_env_worker import (
        _quat2axisangle,
    )
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    reset, steps = recorded_episode(trace)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task, state = reset["task"], reset["initial_state"]
    initial_states = suite.get_task_init_states(task)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task),
        camera_heights=256,
        camera_widths=256,
    )
    first_difference = None
    matching_observations = 0
    success_once = False

    def record(obs, expected, tick, reward=0.0, done=False):
        nonlocal first_difference, matching_observations
        converted = {
            "primary_image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "wrist_image": np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            ),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ).astype(np.float32),
        }
        observation = observation_record(converted)
        changed = [key for key in observation if observation[key] != expected[key]]
        matching_observations += not changed
        save_images = tick == -1 or tick in (snapshot_ticks or set())
        if changed and first_difference is None:
            first_difference = {"tick": tick, "fields": changed}
            save_images = True
        if save_images:
            for camera in ("primary_image", "wrist_image"):
                Image.fromarray(converted[camera]).save(
                    output / f"tick_{tick}_{camera}.png"
                )
        data = env.sim.data
        return {
            "tick": tick,
            "observation": observation,
            "qpos": np.asarray(data.qpos).tolist(),
            "qvel": np.asarray(data.qvel).tolist(),
            "ctrl": np.asarray(data.ctrl).tolist(),
            "qacc_warmstart": np.asarray(data.qacc_warmstart).tolist(),
            "time": float(data.time),
            "contacts": [
                {"geom1": int(c.geom1), "geom2": int(c.geom2), "dist": float(c.dist)}
                for c in data.contact[: data.ncon]
            ],
            "reward": float(reward),
            "done": bool(done),
            "changed_from_recording": changed,
        }

    try:
        env.seed(reset["seed"])
        env.reset()
        obs = env.set_init_state(initial_states[state])
        model = env.sim.model
        (output / "model.json").write_text(
            json.dumps(
                {
                    "geom_names": [model.geom_id2name(i) for i in range(model.ngeom)],
                    "geom_contype": np.asarray(model.geom_contype).tolist(),
                    "geom_conaffinity": np.asarray(model.geom_conaffinity).tolist(),
                    "body_mass": np.asarray(model.body_mass).tolist(),
                    "body_inertia": np.asarray(model.body_inertia).tolist(),
                    "control_freq": env.env.control_freq,
                    "timestep": float(model.opt.timestep),
                },
                indent=2,
            )
            + "\n"
        )
        with (output / "physics.jsonl").open("x", buffering=1) as stream:
            stream.write(json.dumps(record(obs, reset["observation"], -1)) + "\n")
            for row in steps:
                obs, reward, done, _ = env.step(row["action"])
                success_once |= bool(done and reward > 0.5)
                stream.write(
                    json.dumps(
                        record(obs, row["observation"], row["tick"], reward, done)
                    )
                    + "\n"
                )
    finally:
        env.close()
    return {
        "task": task,
        "state": state,
        "steps": len(steps),
        "matching_observations_including_reset": matching_observations,
        "first_difference": first_difference,
        "success_once": success_once,
        "limitation": "Open-loop replay of fixed commands; not closed-loop policy evaluation.",
    }


def main() -> None:
    """Record CPU-only integrity checks or replay one trace on an idle GPU 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["inventory", "replay", "compare-preprocessing"]
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--libero-source", type=Path)
    parser.add_argument("--snapshot-ticks", type=int, nargs="*", default=[])
    args = parser.parse_args()
    check_source(args.source)
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.mode == "replay" and args.trace is None:
        parser.error("replay requires --trace")
    if args.mode == "compare-preprocessing" and (
        args.baseline is None or args.candidate is None
    ):
        parser.error("compare-preprocessing requires --baseline and --candidate")
    sys.path.insert(0, str(args.source.resolve()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    versions = {}
    for name in (
        "mujoco",
        "rlinf-libero",
        "robosuite",
        "numpy",
        "torch",
        "transformers",
    ):
        versions[name] = metadata.version(name)
    args.output.mkdir(parents=True)
    manifest = {
        "complete": False,
        "mode": args.mode,
        "versions": versions,
        "libero_dependency_conflicts": dependency_conflicts("rlinf-libero"),
        "training": False,
        "source_revision": check_source(args.source),
        "diagnostic_sha256": file_hash(Path(__file__)),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.mode == "inventory":
        results = {"release_files": release_integrity(args.storage)}
        if args.libero_source is not None:
            results["libero_tree"] = libero_tree_integrity(args.libero_source)
    elif args.mode == "compare-preprocessing":
        results = compare_preprocessing(args.baseline, args.candidate)
    else:
        import fcntl

        from toolkits.rlt.libero_egl import egl_device_index

        with Path("/dev/shm/rlt-libero-audit-gpu2.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            used, uuid = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        "2",
                        "--query-gpu=memory.used,uuid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                .strip()
                .split(", ")
            )
            if int(used) > 512:
                raise RuntimeError(f"GPU 2 is busy ({used} MiB); no simulator created")
            index = egl_device_index(uuid)
            os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(index))
            manifest.update(
                gpu_uuid=uuid,
                egl_device_index=index,
                trace=str(args.trace.resolve()),
                trace_sha256=file_hash(args.trace),
            )
            manifest["snapshot_ticks"] = args.snapshot_ticks
            results = replay(args.trace, args.output, set(args.snapshot_ticks))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
