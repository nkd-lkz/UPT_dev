# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and summarize a matched LIBERO-Pro medium-difficulty gate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json

METHOD_NAMES = {
    "reference": "Frozen Pi0.5 reference",
    "alphabrain_rlt": "AlphaBrain RLT Q+BC",
    "action_recon": "Matched small RLT",
    "joint": "Physical latent",
    "pointwise_axis0": "Pointwise consequence control",
    "effect_axis0": "Paired-effect Physical Token",
}


def cases(
    task: int,
    perturbation: str,
    states: range,
    split: str,
    seed_base: int,
) -> list[dict]:
    """Build explicit published-state cases without modulo state reuse."""
    if not 0 <= task < 10 or perturbation not in {"lan", "object", "swap", "task"}:
        raise ValueError("Require a valid LIBERO-Pro task and perturbation")
    if split not in {"train", "validation", "development"}:
        raise ValueError("Unknown experiment split")
    return [
        {
            "id": f"medium-t{task}-{perturbation}-{split}-s{state}",
            "suite": "libero_10",
            "task": task,
            "kind": "published",
            "state": state,
            "perturbation": perturbation,
            "seed": seed_base + state,
            "split": split,
        }
        for state in states
    ]


def _job(
    name: str,
    cwd: Path,
    argv: list[str],
    *,
    dependencies: tuple[str, ...] = (),
    timeout: int = 7200,
    manifest: Path | None = None,
    requires_gpu: bool = True,
    environment: dict[str, str],
) -> dict:
    result = {
        "name": name,
        "cwd": str(cwd),
        "argv": argv,
        "depends_on": list(dependencies),
        "timeout_seconds": timeout,
        "requires_gpu": requires_gpu,
        "env": environment,
    }
    if manifest is not None:
        result["manifest"] = str(manifest)
    return result


def prepare(
    root: Path,
    source: Path,
    assets: Path,
    python: Path,
    code: Path,
    gpu: int,
    task: int = 5,
    perturbation: str = "object",
) -> dict:
    """Freeze one fail-fast comparison before observing its outcomes."""
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        raise FileExistsError("An existing medium-gate plan is immutable")
    train = cases(task, perturbation, range(0, 6), "train", 810000)
    validation = cases(task, perturbation, range(6, 10), "validation", 811000)
    paired_development = cases(task, perturbation, range(30, 32), "development", 812000)
    control_development = cases(
        task, perturbation, range(32, 36), "development", 813000
    )
    input_root = root / "inputs"
    input_root.mkdir()
    paths = {
        "train": input_root / "train.json",
        "physical": input_root / "physical.json",
        "control_development": input_root / "control_development.json",
    }
    write_json(paths["train"], train)
    write_json(paths["physical"], train + validation + paired_development)
    write_json(paths["control_development"], control_development)

    revision = subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip()
    dependency_path = (
        "/mnt/hdd_toshiba_1/luokz/libero_pro_baseline_20261025/python_deps"
    )
    environment = {
        "LIBERO_TYPE": "pro",
        "LIBERO_PERTURBATION": "all",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": f"{dependency_path}:{code}",
        "LD_LIBRARY_PATH": "/mnt/hdd_toshiba_1/luokz/libero_pro_baseline_20261025/system_libs:/usr/local/cuda-12.1/lib64",
    }
    common = [
        "--source",
        str(source),
        "--assets",
        str(assets),
        "--benchmark",
        "libero_pro",
        "--suite",
        "libero_10",
        "--task",
        str(task),
        "--wandb",
        "disabled",
    ]
    baseline = root / "baseline"
    physical = root / "physical"
    baseline_collection = baseline / "collection"
    baseline_warmup = baseline / "warmup"
    baseline_online = baseline / "online"
    baseline_reference = baseline / "reference_eval"
    baseline_actor = baseline / "actor_eval"
    physical_collection = physical / "collection"
    probes = physical / "probes"
    paired_train = physical / "paired_train_axis0"
    paired_eval = physical / "paired_development_axis0"
    effects = physical / "effect_readouts"
    control = physical / "control_seed31"

    jobs = [
        _job(
            "baseline-reference-eval",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.pi05_baseline",
                "reference",
                *common,
                "--output",
                str(baseline_reference),
                "--cases",
                str(paths["control_development"]),
                "--episodes",
                "4",
                "--min-successes",
                "0",
                "--evaluation-role",
                "development",
                "--seed",
                "91001",
                "--gpu",
                str(gpu),
            ],
            manifest=baseline_reference / "manifest.json",
            environment=environment,
        ),
        _job(
            "baseline-collection",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.pi05_baseline",
                "reference",
                *common,
                "--output",
                str(baseline_collection),
                "--cases",
                str(paths["train"]),
                "--episodes",
                "6",
                "--min-successes",
                "0",
                "--collect",
                "--evaluation-role",
                "training",
                "--seed",
                "91002",
                "--gpu",
                str(gpu),
            ],
            manifest=baseline_collection / "manifest.json",
            environment=environment,
        ),
        _job(
            "baseline-warmup",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.pi05_baseline",
                "warmup",
                *common,
                "--output",
                str(baseline_warmup),
                "--cache",
                str(baseline_collection / "features.pt"),
                "--encoder-steps",
                "400",
                "--bc-steps",
                "1500",
                "--seed",
                "31",
                "--gpu",
                str(gpu),
            ],
            dependencies=("baseline-collection",),
            manifest=baseline_warmup / "manifest.json",
            environment=environment,
        ),
        _job(
            "baseline-online",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.pi05_baseline",
                "online",
                *common,
                "--output",
                str(baseline_online),
                "--cases",
                str(paths["train"]),
                "--episodes",
                "6",
                "--min-successes",
                "0",
                "--checkpoint",
                str(baseline_warmup / "warmup.pt"),
                "--objective",
                "q_bc",
                "--interaction-ticks",
                "1040",
                "--evaluation-role",
                "training",
                "--seed",
                "31",
                "--gpu",
                str(gpu),
            ],
            dependencies=("baseline-warmup",),
            manifest=baseline_online / "manifest.json",
            environment=environment,
        ),
        _job(
            "baseline-actor-eval",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.pi05_baseline",
                "actor-eval",
                *common,
                "--output",
                str(baseline_actor),
                "--cases",
                str(paths["control_development"]),
                "--episodes",
                "4",
                "--min-successes",
                "0",
                "--checkpoint",
                str(baseline_online / "online.pt"),
                "--objective",
                "q_bc",
                "--evaluation-role",
                "development",
                "--seed",
                "91003",
                "--gpu",
                str(gpu),
            ],
            dependencies=("baseline-online",),
            manifest=baseline_actor / "manifest.json",
            environment=environment,
        ),
        _job(
            "physical-collection",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.physrlt.collect",
                "--source",
                str(source),
                "--assets",
                str(assets),
                "--cases",
                str(paths["physical"]),
                "--output",
                str(physical_collection),
                "--ticks",
                "320",
                "--noise",
                "0.03",
                "--gpu",
                str(gpu),
            ],
            timeout=10800,
            environment=environment,
        ),
        _job(
            "physical-probes",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.physrlt.probe",
                "--dataset",
                str(physical_collection / "transitions.pt"),
                "--output",
                str(probes),
                "--steps",
                "400",
                "--seeds",
                "31",
                "--methods",
                "action_recon",
                "joint",
                "shuffled_joint",
                "state_action_joint",
            ],
            dependencies=("physical-collection",),
            requires_gpu=False,
            environment=environment,
        ),
        _job(
            "paired-train-axis0",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.physrlt.counterfactual",
                "--source",
                str(source),
                "--dataset",
                str(physical_collection / "transitions.pt"),
                "--probes",
                str(probes),
                "--output",
                str(paired_train),
                "--cases",
                "2",
                "--ticks",
                "40",
                "100",
                "160",
                "--axis",
                "0",
                "--split",
                "train",
                "--gpu",
                str(gpu),
            ],
            dependencies=("physical-probes",),
            environment=environment,
        ),
        _job(
            "paired-development-axis0",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.physrlt.counterfactual",
                "--source",
                str(source),
                "--dataset",
                str(physical_collection / "transitions.pt"),
                "--probes",
                str(probes),
                "--output",
                str(paired_eval),
                "--cases",
                "2",
                "--ticks",
                "40",
                "100",
                "160",
                "--axis",
                "0",
                "--split",
                "development",
                "--gpu",
                str(gpu),
            ],
            dependencies=("physical-probes",),
            environment=environment,
        ),
        _job(
            "effect-readouts",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.physrlt.effect_probe",
                "--dataset",
                str(physical_collection / "transitions.pt"),
                "--measurements",
                str(paired_train / "measurements.pt"),
                "--paired-result",
                str(paired_train / "result.json"),
                "--evaluation-measurements",
                str(paired_eval / "measurements.pt"),
                "--evaluation-paired-result",
                str(paired_eval / "result.json"),
                "--checkpoint",
                str(probes / "joint-seed31" / "readout.pt"),
                "--output",
                str(effects),
                "--parameter-scope",
                "decoders",
                "--steps",
                "100",
                "--seed",
                "31",
            ],
            dependencies=("paired-train-axis0", "paired-development-axis0"),
            requires_gpu=False,
            environment=environment,
        ),
        _job(
            "physical-control",
            code,
            [
                str(python),
                "-u",
                "-m",
                "toolkits.rlt.physrlt.control",
                "--source",
                str(source),
                "--assets",
                str(assets),
                "--dataset",
                str(physical_collection / "transitions.pt"),
                "--probes",
                str(probes),
                "--output",
                str(control),
                "--seed",
                "31",
                "--warmup-steps",
                "1500",
                "--online-ticks",
                "1040",
                "--correction-limit",
                "0.1",
                "--correction-axes",
                "0",
                "--development-cases",
                str(paths["control_development"]),
                "--external-readouts",
                f"pointwise_axis0={effects / 'checkpoints' / 'pointwise.pt'}",
                f"effect_axis0={effects / 'checkpoints' / 'effect.pt'}",
                "--relative-critics",
                "pointwise_axis0",
                "effect_axis0",
                "--methods",
                "action_recon",
                "joint",
                "pointwise_axis0",
                "effect_axis0",
                "--gpu",
                str(gpu),
            ],
            dependencies=("effect-readouts",),
            timeout=14400,
            manifest=control / "result.json",
            environment=environment,
        ),
    ]
    plan = {
        "gpu": gpu,
        "hours": 12,
        "stop_on_failure": False,
        "wait_for_lease": True,
        "revision": revision,
        "require_clean_worktree": True,
        "purpose": "Fail-fast LIBERO-Pro medium-difficulty comparison; not final paper evidence",
        "benchmark_cell": f"task{task}-{perturbation}",
        "budgets": {
            "readout_updates": 400,
            "actor_bc_updates": 1500,
            "online_control_ticks": 1040,
            "training_seeds": [31],
            "control_development_states": [32, 33, 34, 35],
        },
        "advance_rule": "Physical arm must gain a success over its matched control, or tie successes with at least 5% lower capped mean ticks; no validation regression",
        "matched_attribution": {
            "joint": "action_recon",
            "effect_axis0": "pointwise_axis0",
        },
        "formal_confirmation_if_advanced": "add seeds 47 and 59, then evaluate published states10-29 once after configuration freeze",
        "jobs": jobs,
    }
    write_json(plan_path, plan)
    return plan


def _pi05_episodes(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("episodes", [])


def _aggregate(rows: list[dict]) -> tuple[int, int, float]:
    successes = sum(bool(row["success"]) for row in rows)
    ticks = statistics.mean(float(row["ticks"]) for row in rows)
    return successes, len(rows), ticks


def _physical_episodes(root: Path) -> list[dict]:
    """Read either the serial control result or disjoint parallel shards."""
    serial = root / "physical/control_seed31/episodes.json"
    if serial.exists():
        return json.loads(serial.read_text())
    rows = []
    for name in ("control_direct_seed31", "control_effect_seed31"):
        path = root / f"physical/{name}/episodes.json"
        if path.exists():
            rows.extend(json.loads(path.read_text()))
    return rows


def summarize(root: Path) -> list[dict]:
    """Publish a single table that keeps pending and negative outcomes visible."""
    plan = json.loads((root / "plan.json").read_text())
    task = plan["benchmark_cell"]
    observations: dict[str, tuple[int, int, float]] = {}
    for method, path in (
        ("reference", root / "baseline/reference_eval/episodes.json"),
        ("alphabrain_rlt", root / "baseline/actor_eval/episodes.json"),
    ):
        episodes = _pi05_episodes(path)
        if episodes:
            observations[method] = _aggregate(episodes)
    validation_successes: dict[str, int] = {}
    physical_episodes = _physical_episodes(root)
    if physical_episodes:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in physical_episodes:
            groups[(row["method"], row["stage"])].append(row)
        for method in ("action_recon", "joint", "pointwise_axis0", "effect_axis0"):
            frozen = groups.get((method, "online_frozen_eval"), [])
            selected = [row for row in frozen if row.get("split") == "development"]
            validation = [row for row in frozen if row.get("split") == "validation"]
            if selected:
                observations[method] = _aggregate(selected)
            if validation:
                validation_successes[method] = sum(
                    bool(row["success"]) for row in validation
                )
    reference_rate = (
        100 * observations["reference"][0] / observations["reference"][1]
        if "reference" in observations
        else None
    )
    matched = {
        "joint": "action_recon",
        "effect_axis0": "pointwise_axis0",
    }
    result = []
    for method in METHOD_NAMES:
        observed = observations.get(method)
        if observed is None:
            result.append(
                {
                    "benchmark_cell": task,
                    "method": METHOD_NAMES[method],
                    "seed_count": 1,
                    "validation_successes": "",
                    "eval_episodes": "",
                    "successes": "",
                    "success_rate_pct": "",
                    "capped_mean_ticks": "",
                    "delta_vs_reference_pp": "",
                    "delta_vs_matched_pp": "",
                    "status": "pending",
                }
            )
            continue
        successes, count, ticks = observed
        rate = 100 * successes / count
        comparator = observations.get(matched.get(method, ""))
        delta_matched = (
            rate - 100 * comparator[0] / comparator[1] if comparator else None
        )
        status = "measured"
        if comparator:
            comparator_rate = 100 * comparator[0] / comparator[1]
            development_gain = rate > comparator_rate or (
                rate == comparator_rate and ticks <= 0.95 * comparator[2]
            )
            comparator_name = matched[method]
            validation_known = (
                method in validation_successes
                and comparator_name in validation_successes
            )
            validation_ok = validation_known and (
                validation_successes[method] >= validation_successes[comparator_name]
            )
            if development_gain and validation_ok:
                status = "advance"
            elif development_gain and not validation_known:
                status = "pending_validation"
            else:
                status = "no_gain"
        result.append(
            {
                "benchmark_cell": task,
                "method": METHOD_NAMES[method],
                "seed_count": 1,
                "validation_successes": validation_successes.get(method, ""),
                "eval_episodes": count,
                "successes": successes,
                "success_rate_pct": round(rate, 2),
                "capped_mean_ticks": round(ticks, 2),
                "delta_vs_reference_pp": (
                    round(rate - reference_rate, 2)
                    if reference_rate is not None
                    else ""
                ),
                "delta_vs_matched_pp": (
                    round(delta_matched, 2) if delta_matched is not None else ""
                ),
                "status": status,
            }
        )
    output = root / "comparison.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    headings = list(result[0])
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[key]) for key in headings) + " |" for row in result
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--code", type=Path, default=Path.cwd())
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--task", type=int, default=5)
    parser.add_argument(
        "--perturbation", choices=("lan", "object", "swap", "task"), default="object"
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.source or not args.assets or not args.python:
            parser.error("prepare requires source, assets and python")
        prepare(
            args.root,
            args.source,
            args.assets,
            args.python,
            args.code,
            args.gpu,
            args.task,
            args.perturbation,
        )
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
