# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fit small physical readouts and audit frozen representations on new layouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from toolkits.rlt.physrlt.data import HORIZONS, check_split, masked_loss
from toolkits.rlt.physrlt.model import PhysicalReadout
from toolkits.rlt.pi05_baseline import write_json

METHODS = (
    "action_recon",
    "geometry",
    "response",
    "joint",
    "shuffled_joint",
    "prefix_joint",
    "state_action_joint",
    "hybrid_recon",
    "hybrid_joint",
    "joint_recon",
    "factorized_joint",
    "residual_joint",
    "arm_q_gripper_joint",
    "eef_gripper_joint",
    "balanced_joint",
    "flow_recon",
    "flow_joint",
)

RESPONSE_GROUPS = {
    "arm_joint_position": tuple(range(7)),
    "eef_position": tuple(range(7, 10)),
    "gripper_position": tuple(range(10, 12)),
}


def dataset_sha256(path: Path) -> str:
    """Hash the immutable transition artifact used to fit a readout."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_indices(method: str) -> tuple[int, ...]:
    """Select measured response fields for a redundancy ablation.

    Collection retains every commonly available proprioceptive field. These
    variants change only the supervision target, so their input, bottleneck,
    geometry loss and optimization budget remain matched.
    """
    if method == "arm_q_gripper_joint":
        return (
            RESPONSE_GROUPS["arm_joint_position"] + RESPONSE_GROUPS["gripper_position"]
        )
    if method == "eef_gripper_joint":
        return RESPONSE_GROUPS["eef_position"] + RESPONSE_GROUPS["gripper_position"]
    return tuple(range(12))


def mismatched_targets(index: torch.Tensor, episode_id: torch.Tensor) -> dict[int, int]:
    """Choose a physical target from another episode for every train row."""
    result = {}
    for source in index:
        candidates = index[episode_id[index] != episode_id[source]]
        if not len(candidates):
            raise ValueError("Shuffled controls require multiple training episodes")
        target = candidates[torch.randint(len(candidates), ())]
        result[int(source)] = int(target)
    return result


def execution_response_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Balance response groups or apply the declared field ablation."""
    if method == "balanced_joint":
        return torch.stack(
            [
                masked_loss(prediction[..., indices], target[..., indices], mask)
                for indices in RESPONSE_GROUPS.values()
            ]
        ).mean()
    selected = response_indices(method)
    return masked_loss(prediction[..., selected], target[..., selected], mask)


def tensor_data(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Load only completed-horizon labels and pre-execution inputs."""
    keys = (
        "action_feature",
        "prefix_feature",
        "ref",
        "prop",
        "commands",
        "response",
        "geometry",
        "horizon_mask",
    )
    result = {key: torch.stack([r[key] for r in rows]) for key in keys}
    if all("flow_feature" in row for row in rows):
        result["flow_feature"] = torch.stack([row["flow_feature"] for row in rows])
    cases = sorted({r["case_id"] for r in rows})
    result["episode_id"] = torch.tensor([cases.index(r["case_id"]) for r in rows])
    return result


def normalization(data: dict, train: torch.Tensor, feature_key: str) -> dict:
    """Estimate scaling exclusively on training layouts, with physical floors."""
    result = {}
    for key in ("action_feature", "prefix_feature", "flow_feature", "prop"):
        if key not in data:
            continue
        values = data[key][train]
        result[key + "_mean"] = values.mean(0)
        result[key + "_scale"] = values.std(0).clamp_min(0.05)
    for key in ("response", "geometry"):
        values = data[key][train][data["horizon_mask"][train]]
        result[key + "_scale"] = values.std(0).clamp_min(0.01)
    return result


def inputs(data: dict, stats: dict, method: str) -> tuple:
    """Apply train-only normalization identically in fitting and deployment."""
    if method.startswith("flow_"):
        key = "flow_feature"
    else:
        key = "prefix_feature" if method == "prefix_joint" else "action_feature"
    if key not in data:
        raise ValueError(f"Dataset does not contain the requested {key}")
    feature = (data[key] - stats[key + "_mean"]) / stats[key + "_scale"]
    if method.startswith("hybrid_"):
        prefix = (data["prefix_feature"] - stats["prefix_feature_mean"]) / stats[
            "prefix_feature_scale"
        ]
        # Preserve four bins from each pathway without increasing input width.
        feature = torch.cat(
            (
                feature.reshape(-1, 4, 128)[..., :64].flatten(1),
                prefix.reshape(-1, 4, 128)[..., :64].flatten(1),
            ),
            -1,
        )
    if method == "state_action_joint":
        feature = torch.zeros_like(feature)
    prop = (data["prop"] - stats["prop_mean"]) / stats["prop_scale"]
    return feature, prop, data["ref"]


def fit_readout(data: dict, train: torch.Tensor, method: str, seed: int, steps: int):
    """Train fixed-budget, equal-capacity variants; future states are labels only."""
    torch.manual_seed(seed)
    model = PhysicalReadout(
        data["response"].shape[-1],
        data["geometry"].shape[-1],
        factorized=method == "factorized_joint",
    )
    if method.startswith("flow_"):
        feature_key = "flow_feature"
    else:
        feature_key = "prefix_feature" if method == "prefix_joint" else "action_feature"
    stats = normalization(data, train, feature_key)
    response_target = data["response"]
    if method == "residual_joint":
        response_target = crossfitted_response_residual(data, train)
    features, props, refs = inputs(data, stats, method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    index = train.nonzero().flatten()
    shuffled = mismatched_targets(index, data["episode_id"])
    losses = []
    for step in range(steps):
        selected = index[torch.randint(len(index), (min(64, len(index)),))]
        targets = (
            torch.tensor([shuffled[int(i)] for i in selected])
            if method == "shuffled_joint"
            else selected
        )
        z = model.encode(features[selected], props[selected], refs[selected])
        pred = model.consequences(z, data["commands"][selected])
        mask = data["horizon_mask"][targets]
        loss = z.sum() * 0
        terms = {}
        if method in {"action_recon", "hybrid_recon", "flow_recon"}:
            loss = (model.reconstruct(z) - features[selected]).square().mean()
        else:
            for key in ("response", "geometry"):
                if method == "geometry" and key == "response":
                    continue
                if method == "response" and key == "geometry":
                    continue
                target = (response_target if key == "response" else data[key])[
                    targets
                ] / stats[key + "_scale"]
                if key == "geometry":
                    # Prevent 64 depth bins from overwhelming the manipulated
                    # objects' few position coordinates.
                    terms[key] = 0.5 * (
                        masked_loss(pred[key][..., :64], target[..., :64], mask)
                        + masked_loss(pred[key][..., 64:], target[..., 64:], mask)
                    )
                else:
                    terms[key] = execution_response_loss(
                        pred[key], target, mask, method
                    )
                loss = loss + terms[key]
            if method == "joint_recon":
                loss = (
                    loss
                    + 0.1 * (model.reconstruct(z) - features[selected]).square().mean()
                )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {method} objective at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite physical readout gradients")
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            losses.append(
                {"step": step + 1, "loss": float(loss.detach()), "grad": float(grad)}
            )
    return model.eval().requires_grad_(False), stats, losses


def ridge_predict(x: torch.Tensor, y: torch.Tensor, train: torch.Tensor, ridge: float):
    """Fit a common linear probe; learned targets never enter its input."""
    mean = x[train].mean(0)
    scale = x[train].std(0).clamp_min(0.1)
    x = (x - mean) / scale
    x = torch.cat((x, torch.ones(len(x), 1)), dim=-1).double()
    penalty = torch.eye(x.shape[-1], dtype=x.dtype) * ridge
    penalty[-1, -1] = 1e-6
    weights = torch.linalg.solve(
        x[train].T @ x[train] + penalty, x[train].T @ y[train].double()
    )
    return (x @ weights).float()


def crossfitted_response_residual(data: dict, train: torch.Tensor) -> torch.Tensor:
    """Subtract a state/action predictor fitted without the target episode.

    This is an empirical local response prior, not a known physical law.
    Only training residuals supervise the encoder. Other splits are never
    used to fit the prior or the representation.
    """
    residual = torch.zeros_like(data["response"])
    for column, horizon in enumerate(HORIZONS):
        commands = data["commands"].clone()
        commands[:, horizon:] = 0
        x = torch.cat((data["prop"], commands.flatten(1)), -1)
        target = data["response"][:, column]
        valid = data["horizon_mask"][:, column]
        for episode in data["episode_id"][train].unique():
            selected = train & (data["episode_id"] == episode) & valid
            others = train & (data["episode_id"] != episode) & valid
            if others.sum() < 2:
                raise ValueError(
                    "Residual supervision requires multiple training episodes"
                )
            prediction = ridge_predict(x, target, others, ridge=10.0)
            residual[selected, column] = target[selected] - prediction[selected]
    return residual


@torch.no_grad()
def evaluate_representation(model, data, stats, method, rows) -> dict:
    """Use the same frozen linear readout for every representation variant."""
    train = torch.tensor([r["split"] == "train" for r in rows])
    validation = torch.tensor([r["split"] == "validation" for r in rows])
    development = torch.tensor([r["split"] == "development" for r in rows])
    z = model.encode(*inputs(data, stats, method))
    result = {"probe": "frozen representation + matched ridge probe", "horizons": {}}
    for column, horizon in enumerate(HORIZONS):
        commands = data["commands"].clone()
        commands[:, horizon:] = 0
        candidate = commands.flatten(1)
        # Direct current-proprio + action is included as an honest simple baseline.
        variants = {
            "token_and_action": torch.cat((z, candidate, data["prop"]), -1),
            "token_without_candidate": torch.cat((z, data["prop"]), -1),
            "state_and_action": torch.cat((candidate, data["prop"]), -1),
        }
        valid = data["horizon_mask"][:, column]
        measures = {}
        for field in ("response", "geometry"):
            target = data[field][:, column]
            field_results = {}
            for name, x in variants.items():
                trials = []
                for ridge in (0.1, 10.0, 100.0):
                    pred = ridge_predict(x, target, train & valid, ridge)
                    score = (
                        ((pred - target) / stats[field + "_scale"]).square().mean(-1)
                    )
                    trials.append(
                        (float(score[validation & valid].mean()), ridge, pred)
                    )
                val, chosen, pred = min(trials, key=lambda value: value[0])
                error = (
                    pred[development & valid] - target[development & valid]
                ).square()
                zero = target[development & valid].square()
                field_results[name] = {
                    "validation_scaled_mse": val,
                    "ridge_from_validation_only": chosen,
                    "development_rmse": float(error.mean().sqrt()),
                    "development_persistence_rmse": float(zero.mean().sqrt()),
                    "development_scaled_mse": float(
                        (error / stats[field + "_scale"].square()).mean()
                    ),
                    "development_samples": int((development & valid).sum()),
                }
                if field == "response":
                    field_results[name]["groups"] = {
                        group: {
                            "development_rmse": float(error[:, indices].mean().sqrt()),
                            "development_scaled_mse": float(
                                (
                                    error[:, indices]
                                    / stats["response_scale"][list(indices)].square()
                                ).mean()
                            ),
                        }
                        for group, indices in RESPONSE_GROUPS.items()
                    }
                if field == "geometry":
                    field_results[name]["depth_rmse_metres"] = float(
                        error[:, :64].mean().sqrt()
                    )
                    field_results[name]["object_rmse_metres"] = float(
                        error[:, 64:].mean().sqrt()
                    )
                    moving = target[development & valid, 64:].abs().amax(-1) > 0.001
                    field_results[name]["moving_object_chunks"] = int(moving.sum())
                    field_results[name]["moving_object_rmse_metres"] = (
                        float(error[moving, 64:].mean().sqrt())
                        if moving.any()
                        else None
                    )
            measures[field] = field_results
        result["horizons"][str(horizon)] = measures
    return result


def run(args: argparse.Namespace) -> list[dict]:
    torch.set_num_threads(args.threads)
    payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    rows = [r for r in payload["rows"] if bool(r["horizon_mask"].any())]
    check_split(rows)
    for split in ("train", "validation", "development"):
        cases = {r["case_id"] for r in rows if r["split"] == split}
        if len(cases) < 2:
            raise ValueError(f"Require at least two whole episodes for {split}")
    if len({tuple(r["object_keys"]) for r in rows}) != 1:
        raise ValueError("Run probes per task with a fixed geometry target schema")
    data = tensor_data(rows)
    train = torch.tensor([r["split"] == "train" for r in rows])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    data_digest = dataset_sha256(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        for method in args.methods:
            run_dir = args.output / f"{method}-seed{seed}"
            run_dir.mkdir(exist_ok=True)
            if (run_dir / "result.json").exists():
                all_results.append(json.loads((run_dir / "result.json").read_text()))
                continue
            started = time.monotonic()
            model, stats, losses = fit_readout(data, train, method, seed, args.steps)
            evaluation = evaluate_representation(model, data, stats, method, rows)
            result = {
                "state": "complete",
                "revision": revision,
                "dataset_sha256": data_digest,
                "method": method,
                "seed": seed,
                "factorized": model.factorized,
                "response_target": "crossfitted_state_action_residual"
                if method == "residual_joint"
                else "measured_delta",
                "response_supervision": [
                    group
                    for group, indices in RESPONSE_GROUPS.items()
                    if set(indices).issubset(response_indices(method))
                ],
                "response_weighting": "equal_physical_groups"
                if method == "balanced_joint"
                else "equal_selected_dimensions",
                "steps": args.steps,
                "training_rows": int(train.sum()),
                "all_rows": len(rows),
                "seconds": time.monotonic() - started,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "loss_trace": losses,
                **evaluation,
                "interpretation": "offline development evidence, NOT control improvement",
            }
            torch.save(
                {
                    "schema": 1,
                    "revision": revision,
                    "dataset_sha256": data_digest,
                    "model": model.state_dict(),
                    "stats": stats,
                    "method": method,
                    "seed": seed,
                    "factorized": model.factorized,
                    "response_target": result["response_target"],
                    "response_supervision": result["response_supervision"],
                    "response_weighting": result["response_weighting"],
                    "response_dim": data["response"].shape[-1],
                    "geometry_dim": data["geometry"].shape[-1],
                    "dataset": str(args.dataset),
                },
                run_dir / "readout.pt",
            )
            write_json(run_dir / "result.json", result)
            all_results.append(result)
            write_json(args.output / "results.json", all_results)
            print(
                json.dumps(
                    {"method": method, "seed": seed, "seconds": result["seconds"]}
                ),
                flush=True,
            )
    with (args.output / "table.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method",
                "seed",
                "horizon",
                "target",
                "token_scaled_mse",
                "state_action_scaled_mse",
                "rmse",
                "persistence_rmse",
                "n",
            ]
        )
        for result in all_results:
            for horizon, fields in result["horizons"].items():
                for field, probes in fields.items():
                    token = probes["token_and_action"]
                    writer.writerow(
                        [
                            result["method"],
                            result["seed"],
                            horizon,
                            field,
                            token["development_scaled_mse"],
                            probes["state_and_action"]["development_scaled_mse"],
                            token["development_rmse"],
                            token["development_persistence_rmse"],
                            token["development_samples"],
                        ]
                    )
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS[:7])
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 47, 59])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("Positive steps/threads required")
    run(args)


if __name__ == "__main__":
    main()
