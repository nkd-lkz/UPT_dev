# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate geometry-evolution evidence without inflating sample counts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _percentile_interval(values: torch.Tensor) -> list[float]:
    quantiles = torch.quantile(values, values.new_tensor([0.025, 0.975]))
    return [float(value) for value in quantiles]


def _bootstrap_rmse(
    mse_by_layout: torch.Tensor,
    *,
    draws: int,
    generator: torch.Generator,
) -> tuple[float, list[float]]:
    if mse_by_layout.ndim != 1 or not len(mse_by_layout):
        raise ValueError("Geometry evidence requires one or more layouts")
    indices = torch.randint(
        len(mse_by_layout),
        (draws, len(mse_by_layout)),
        generator=generator,
    )
    samples = mse_by_layout[indices].mean(-1).sqrt()
    return float(mse_by_layout.mean().sqrt()), _percentile_interval(samples)


def _bootstrap_difference(
    target: torch.Tensor,
    comparator: torch.Tensor,
    *,
    draws: int,
    generator: torch.Generator,
) -> tuple[float, list[float]]:
    if target.shape != comparator.shape or target.ndim != 1:
        raise ValueError("Paired geometry arms must contain the same layouts")
    indices = torch.randint(
        len(target),
        (draws, len(target)),
        generator=generator,
    )
    differences = (
        comparator[indices].mean(-1).sqrt() - target[indices].mean(-1).sqrt()
    )
    point = float(comparator.mean().sqrt() - target.mean().sqrt())
    return point, _percentile_interval(differences)


def load_geometry_arms(label: str, path: Path) -> list[dict[str, Any]]:
    """Load layout-clustered transport arms from one completed probe result."""
    payload = json.loads(path.read_text())
    if payload.get("state") != "complete":
        raise ValueError(f"Geometry result is not complete: {path}")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in payload.get("results", []):
        comparison = row.get("layout_comparison")
        if not comparison:
            continue
        key = (
            label,
            payload.get("split"),
            row["method"],
            row["object"],
            int(row["rank"]),
        )
        grouped[key].append(row)
    if not grouped:
        raise ValueError(f"Geometry result has no layout comparisons: {path}")

    arms = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate readout seeds for geometry arm {key}")
        layouts = rows[0]["layout_comparison"]["layouts"]
        metric_sets = [
            set(row["layout_comparison"]["mse_mm2_by_layout"]) for row in rows
        ]
        metrics = set.intersection(*metric_sets)
        for row in rows:
            if row["layout_comparison"]["layouts"] != layouts:
                raise ValueError(f"Layout identities differ across readout seeds: {key}")
        for metric in sorted(metrics):
            values = []
            for row in rows:
                current = row["layout_comparison"]["mse_mm2_by_layout"][metric]
                if len(current) != len(layouts):
                    raise ValueError(f"Geometry errors do not align with layouts: {key}")
                values.append(current)
            seed_layout_mse = torch.tensor(values, dtype=torch.float64)
            arms.append(
                {
                    "source": key[0],
                    "split": key[1],
                    "readout": key[2],
                    "object": key[3],
                    "rank": key[4],
                    "metric": metric,
                    "readout_seeds": seeds,
                    "layouts": layouts,
                    "mse_mm2_by_layout": seed_layout_mse.mean(0).tolist(),
                }
            )
    return arms


def summarize_geometry(
    arms: list[dict[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = 20261029,
) -> dict[str, Any]:
    """Summarize arms and paired contrasts with layouts as independent units."""
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    summaries = []
    arm_index = {}
    for index, arm in enumerate(arms):
        values = torch.tensor(arm["mse_mm2_by_layout"], dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed + index)
        rmse, interval = _bootstrap_rmse(
            values,
            draws=draws,
            generator=generator,
        )
        summary = {
            key: value
            for key, value in arm.items()
            if key != "mse_mm2_by_layout"
        }
        summary.update(
            {
                "readout_seed_count": len(arm["readout_seeds"]),
                "layout_count": len(arm["layouts"]),
                "equal_layout_rmse_mm": rmse,
                "layout_bootstrap_95pct_rmse_mm": interval,
            }
        )
        summaries.append(summary)
        arm_index[
            (
                arm["source"],
                arm["split"],
                arm["readout"],
                arm["object"],
                arm["rank"],
                arm["metric"],
            )
        ] = values

    contrasts = []

    def add_contrast(
        target_key: tuple,
        comparator_key: tuple,
        name: str,
    ) -> None:
        target = arm_index.get(target_key)
        comparator = arm_index.get(comparator_key)
        if target is None or comparator is None:
            return
        generator = torch.Generator().manual_seed(seed + len(contrasts) + 100_000)
        difference, interval = _bootstrap_difference(
            target,
            comparator,
            draws=draws,
            generator=generator,
        )
        contrasts.append(
            {
                "source": target_key[0],
                "split": target_key[1],
                "object": target_key[3],
                "rank": target_key[4],
                "contrast": name,
                "target_readout": target_key[2],
                "target_metric": target_key[5],
                "comparator_readout": comparator_key[2],
                "comparator_metric": comparator_key[5],
                "rmse_improvement_mm": difference,
                "layout_bootstrap_95pct_improvement_mm": interval,
                "interval_excludes_zero": interval[0] > 0,
                "layout_count": len(target),
                "evidence_tier": (
                    "pilot"
                    if len(target) < 6
                    else "confirmation"
                    if target_key[1] == "confirmation"
                    else "development"
                ),
                "confirmation_screen_eligible": (
                    target_key[1] == "confirmation" and len(target) >= 6
                ),
                "passes_confirmation_screen": (
                    target_key[1] == "confirmation"
                    and len(target) >= 6
                    and interval[0] > 0
                ),
            }
        )

    groups = sorted({key[:5] for key in arm_index})
    for group in groups:
        coupled = (*group, "coupled")
        for metric, name in (
            ("direct", "response_transport_vs_direct"),
            ("shuffled_coupling", "response_transport_vs_shuffled"),
            ("state_action_coupling", "response_transport_vs_state_action"),
        ):
            add_contrast(coupled, (*group, metric), name)

    semantic_groups = sorted(
        {(key[0], key[1], key[3], key[4]) for key in arm_index}
    )
    for source, split, object_name, rank in semantic_groups:
        prefix = (source, split)
        suffix = (object_name, rank, "coupled")
        add_contrast(
            (*prefix, "joint", *suffix),
            (*prefix, "state_action_joint", *suffix),
            "action_readout_vs_state_action_readout",
        )
        add_contrast(
            (*prefix, "flow_joint", *suffix),
            (*prefix, "joint", *suffix),
            "flow_path_vs_final_action_readout",
        )

    return {
        "state": "complete",
        "protocol": (
            "average readout seeds within each held-out layout, then bootstrap "
            "layouts as the independent units; positive contrast values favour "
            "the target arm"
        ),
        "limitation": (
            "Prediction evidence is an offline mechanism screen and does not "
            "establish task-success or closed-loop control gains."
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "arms": summaries,
        "contrasts": contrasts,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Geometry-evolution evidence",
        "",
        result["protocol"],
        "",
        f"Limitation: {result['limitation']}",
        "",
        "## Paired contrasts",
        "",
        "| Source | Split | Readout | Object | Contrast | Layouts | Improvement (mm) | 95% interval | Screen |",
        "|---|---|---|---|---|---:|---:|---:|:---:|",
    ]
    for row in result["contrasts"]:
        low, high = row["layout_bootstrap_95pct_improvement_mm"]
        lines.append(
            f"| {row['source']} | {row['split']} | {row['target_readout']} | "
            f"{row['object']} | {row['contrast']} | {row['layout_count']} | "
            f"{row['rmse_improvement_mm']:.3f} | [{low:.3f}, {high:.3f}] | "
            f"{'pass' if row['passes_confirmation_screen'] else 'ineligible/fail'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_geometry_report(result: dict[str, Any], output_dir: Path) -> None:
    """Write machine-readable and human-readable geometry evidence."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_csv(output_dir / "arms.csv", result["arms"])
    _write_csv(output_dir / "contrasts.csv", result["contrasts"])
    _write_markdown(output_dir / "comparison.md", result)


def parse_input(value: str) -> tuple[str, Path]:
    """Parse a LABEL=PATH geometry result argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Input must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError("Input label and result file must exist")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    arms = [
        arm
        for label, path in args.input
        for arm in load_geometry_arms(label, path)
    ]
    result = summarize_geometry(
        arms,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    write_geometry_report(result, args.output_dir)


if __name__ == "__main__":
    main()
