# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Physical readout causality, units, split isolation and experiment contracts."""

import copy
from argparse import Namespace

import numpy as np
import pytest
import torch

from toolkits.rlt.physrlt.campaign import prepare, ready_jobs
from toolkits.rlt.physrlt.data import (
    check_split,
    compact_features,
    masked_loss,
    metric_depth,
    transition_targets,
)
from toolkits.rlt.physrlt.model import PhysicalReadout
from toolkits.rlt.physrlt.probe import inputs, normalization, run, tensor_data


def synthetic_rows():
    """Make explicit synthetic fixtures; never report them as robot evidence."""
    rng = torch.Generator().manual_seed(19)
    rows = []
    for state in range(6):
        for tick in range(4):
            command = torch.randn(10, 7, generator=rng) * 0.1
            delta = torch.stack([command[:h, :3].sum(0) for h in (3, 10)])
            rows.append(
                {
                    "case_id": f"synthetic-{state}",
                    "suite": "libero_10",
                    "task": 0,
                    "state_id": state,
                    "reset_hash": str(state),
                    "split": ("train", "validation", "development")[state // 2],
                    "action_feature": torch.randn(512, generator=rng),
                    "prefix_feature": torch.randn(512, generator=rng),
                    "ref": command.clone(),
                    "commands": command,
                    "prop": torch.randn(8, generator=rng),
                    "response": delta.repeat(1, 4),
                    "geometry": torch.cat((torch.zeros(2, 64), delta), -1),
                    "horizon_mask": torch.ones(2, dtype=torch.bool),
                    "object_keys": ["synthetic_object_pos"],
                }
            )
    return rows


def test_layout_split_ignores_seed_and_condition_names():
    rows = synthetic_rows()
    check_split(rows)
    duplicate = copy.deepcopy(rows[0])
    duplicate.update(case_id="different-name", seed=999, split="development")
    with pytest.raises(ValueError, match="layout"):
        check_split(rows + [duplicate])
    duplicate["state_id"] = 49
    with pytest.raises(ValueError, match="simulator reset"):
        check_split(rows + [duplicate])


def test_depth_is_metric_and_invalid_values_fail():
    np.testing.assert_allclose(metric_depth(np.array([0, 1]), 0.1, 10), [0.1, 10])
    assert metric_depth(np.array([0.5]), 0.1, 10).item() == pytest.approx(0.1980198)
    with pytest.raises(ValueError):
        metric_depth(np.array([1.1]), 0.1, 10)


def test_unexecuted_horizons_cannot_contribute_loss():
    before = {"response": torch.ones(12), "geometry": torch.ones(67)}
    after = {"response": torch.ones(12) * 3, "geometry": torch.ones(67) * 4}
    targets = transition_targets(before, {3: after})
    assert targets["horizon_mask"].tolist() == [True, False]
    assert targets["response"][0, 0] == 2
    prediction = torch.zeros(1, 2, 12, requires_grad=True)
    target = targets["response"][None]
    target[:, 1] = 999
    loss = masked_loss(prediction, target, targets["horizon_mask"][None])
    loss.backward()
    assert loss.item() == 4
    assert prediction.grad[:, 1].count_nonzero() == 0


def test_fixed_feature_budget_does_not_modify_global_rng():
    torch.manual_seed(3)
    hidden = torch.randn(10, 1024)
    state = torch.random.get_rng_state()
    first = compact_features(hidden)
    assert first.shape == (512,)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    torch.testing.assert_close(first, compact_features(hidden))
    assert compact_features(torch.randn(512, 2048)).shape == first.shape


def test_response_decoder_depends_on_candidate_but_not_its_future_suffix():
    torch.manual_seed(13)
    model = PhysicalReadout(12, 67)
    token = model.encode(torch.randn(2, 512), torch.randn(2, 8), torch.randn(2, 10, 7))
    commands = torch.randn(2, 10, 7, requires_grad=True)
    response = model.consequences(token, commands)["response"]
    response[:, 0].sum().backward(retain_graph=True)
    assert commands.grad[:, 3:].count_nonzero() == 0
    assert commands.grad[:, :3].abs().sum() > 0
    with torch.no_grad():
        changed = commands.detach().clone()
        changed[:, 3:] += 10
        torch.testing.assert_close(
            response[:, 0], model.consequences(token, changed)["response"][:, 0]
        )


def test_training_scaling_and_policy_inputs_do_not_read_future_labels():
    rows = synthetic_rows()
    data = tensor_data(rows)
    train = torch.tensor([r["split"] == "train" for r in rows])
    stats = normalization(data, train, "action_feature")
    original = inputs(data, stats, "joint")
    altered = {key: value.clone() for key, value in data.items()}
    altered["action_feature"][~train] = 1e6
    changed_stats = normalization(altered, train, "action_feature")
    for key in stats:
        torch.testing.assert_close(stats[key], changed_stats[key])
    data["geometry"][:] = 1e6
    data["response"][:] = 1e6
    for actual, expected in zip(inputs(data, stats, "joint"), original):
        torch.testing.assert_close(actual, expected)


def test_small_experiment_writes_auditable_results(tmp_path):
    path = tmp_path / "synthetic.pt"
    torch.save({"rows": synthetic_rows()}, path)
    results = run(
        Namespace(
            dataset=path,
            output=tmp_path / "probe",
            methods=["action_recon", "joint"],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    assert len(results) == 2
    assert results[0]["trainable_parameters"] == results[1]["trainable_parameters"]
    assert "NOT control improvement" in results[0]["interpretation"]
    assert (tmp_path / "probe/table.csv").is_file()
    assert (tmp_path / "probe/joint-seed3/readout.pt").is_file()


def test_failed_job_blocks_only_its_dependents():
    jobs = [
        {"id": "a", "dependencies": []},
        {"id": "b", "dependencies": ["a"]},
        {"id": "c", "dependencies": []},
    ]
    state = {
        "a": {"state": "failed"},
        "b": {"state": "pending"},
        "c": {"state": "pending"},
    }
    ready, blocked = ready_jobs(jobs, state)
    assert [job["id"] for job in ready] == ["c"]
    assert blocked == ["b"]


def test_campaign_requires_smoke_and_keeps_layout_splits(tmp_path):
    import json

    prepare(tmp_path, tmp_path / "source", tmp_path / "assets")
    plan = json.loads((tmp_path / "plan.json").read_text())
    for job in plan["jobs"]:
        if job["id"].endswith("-collect"):
            assert job["dependencies"] == ["capture-smoke"]
    assert plan["global_wall_time_limit"] is None
    cases = json.loads((tmp_path / "task0-object/cases.json").read_text())
    assert len({case["state"] for case in cases}) == len(cases)
    with pytest.raises(FileExistsError):
        prepare(tmp_path, tmp_path / "source", tmp_path / "assets")
