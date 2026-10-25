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
from toolkits.rlt.physrlt.probe import (
    crossfitted_response_residual,
    inputs,
    normalization,
    run,
    tensor_data,
)


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


def test_factorized_decoders_have_disjoint_token_gradients():
    model = PhysicalReadout(12, 67, factorized=True)
    token = torch.randn(2, 128, requires_grad=True)
    prediction = model.consequences(token, torch.randn(2, 10, 7))
    prediction["response"].sum().backward()
    assert token.grad[:, :64].count_nonzero() == 0
    assert token.grad[:, 64:].abs().sum() > 0


def test_crossfitted_response_prior_never_reads_dev_labels():
    data = tensor_data(synthetic_rows())
    train = data["episode_id"] < 2
    first = crossfitted_response_residual(data, train)
    data["response"][~train] = 1e9
    second = crossfitted_response_residual(data, train)
    torch.testing.assert_close(first[train], second[train])
    assert first[~train].count_nonzero() == 0


@pytest.mark.parametrize(
    "method",
    [
        "hybrid_recon",
        "hybrid_joint",
        "joint_recon",
        "factorized_joint",
        "residual_joint",
    ],
)
def test_followup_methods_fit_and_restore(tmp_path, method):
    from toolkits.rlt.physrlt.control import FrozenReadout

    path = tmp_path / "synthetic.pt"
    rows = synthetic_rows()
    torch.save({"rows": rows}, path)
    run(
        Namespace(
            dataset=path,
            output=tmp_path / "probes",
            methods=[method],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    readout = FrozenReadout(tmp_path / "probes" / f"{method}-seed3/readout.pt")
    encoded = readout.encode(rows[0], rows[0]["prop"])
    assert encoded["z"].shape == (128,)
    assert torch.isfinite(encoded["z"]).all()
    assert readout.model.factorized == (method == "factorized_joint")


@pytest.mark.parametrize("limit", [7, 10])
@pytest.mark.parametrize("gain", [0.7, 1.0])
def test_control_records_proposal_and_execution_domains_and_real_successor(
    monkeypatch, limit, gain
):
    from toolkits.rlt.physrlt import control

    class SimulatorEdge:
        pilot_reset_hashes = {}
        instruction = "synthetic integration test"

        def reset(self, case):
            self.tick = 0
            return {"state": np.zeros(8, np.float32)}

        def step(self, command):
            self.tick += 1
            return {"state": np.full(8, self.tick, np.float32)}, 0.0, False

    class ReferenceEdge:
        def predict(self, obs, instruction, seed):
            return {"ref": np.full((10, 7), 0.2, np.float32)}

    class Readout:
        def encode(self, prediction, prop):
            return {
                "z": torch.full((128,), float(prop[0])),
                "ref": torch.from_numpy(prediction["ref"]),
                "prop": torch.as_tensor(prop),
            }

    class Learner:
        def command(self, features, deterministic):
            return features["ref"].clone()

        def update(self, replay):
            return {"updated": 1}

    monkeypatch.setattr(control, "physical_reset_hash", lambda env: "synthetic-reset")
    case = {"id": "synthetic", "split": "train", "state": 0, "seed": 3}
    replay = []
    result = control.rollout(
        SimulatorEdge(),
        ReferenceEdge(),
        Readout(),
        Learner(),
        case,
        limit,
        replay,
        execution_gain=gain,
    )
    assert result["ticks"] == limit
    assert len(replay) == 1
    assert replay[0]["action"][0, 6].item() == pytest.approx(0.2)
    assert replay[0]["accepted_commands"][0, 6].item() == 1
    assert replay[0]["accepted_commands"][0, 0].item() == pytest.approx(0.2 * gain)
    assert replay[0]["accepted_commands"][limit:].count_nonzero() == 0
    assert replay[0]["prop"][0] == 9
    assert replay[0]["next_prop"][0] == 9 + limit
    assert len(result["updates"]) == int(limit == 10)


def test_structured_physics_token_remains_causal(tmp_path):
    from toolkits.rlt.physrlt.control import FrozenReadout

    rows = synthetic_rows()
    path = tmp_path / "synthetic.pt"
    torch.save({"rows": rows}, path)
    run(
        Namespace(
            dataset=path,
            output=tmp_path / "probes",
            methods=["joint"],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    for mode in ("outcomes", "mixed"):
        readout = FrozenReadout(tmp_path / "probes/joint-seed3/readout.pt", mode=mode)
        original = readout.encode(rows[0], rows[0]["prop"])["z"]
        altered = copy.deepcopy(rows[0])
        altered["response"][:] = 1e9
        altered["geometry"][:] = 1e9
        torch.testing.assert_close(
            readout.encode(altered, altered["prop"])["z"], original
        )
        assert original.shape == (128,)


def test_delayed_online_rl_requires_state_augmentation():
    from toolkits.rlt.physrlt.control import rollout

    with pytest.raises(ValueError, match="state augmentation"):
        rollout(None, None, None, None, {}, 10, replay=[], delay_ticks=2)


def test_action_prior_and_grounded_readout_have_one_causal_token_budget(tmp_path):
    from toolkits.rlt.physrlt.control import FrozenReadout

    rows = synthetic_rows()
    dataset = tmp_path / "synthetic.pt"
    torch.save({"rows": rows}, dataset)
    run(
        Namespace(
            dataset=dataset,
            output=tmp_path / "probes",
            methods=["action_recon", "joint"],
            seeds=[31, 59],
            steps=2,
            threads=2,
        )
    )
    prior = tmp_path / "probes/action_recon-seed59/readout.pt"
    physical = tmp_path / "probes/joint-seed31/readout.pt"
    combined = FrozenReadout(physical, anchor=prior)
    result = combined.encode(rows[0], rows[0]["prop"])["z"]
    assert result.shape == (128,) and torch.isfinite(result).all()
    assert not torch.allclose(
        result, FrozenReadout(physical).encode(rows[0], rows[0]["prop"])["z"]
    )
    changed = copy.deepcopy(rows[0])
    changed["geometry"][:] = 999
    changed["response"][:] = 999
    torch.testing.assert_close(result, combined.encode(changed, changed["prop"])["z"])
    payload = torch.load(prior, weights_only=True)
    payload["dataset"] = "different-training-data"
    torch.save(payload, tmp_path / "mismatch.pt")
    with pytest.raises(ValueError, match="same training dataset"):
        FrozenReadout(physical, anchor=tmp_path / "mismatch.pt")


def test_new_development_cases_exclude_all_original_layouts(tmp_path):
    import json

    from toolkits.rlt.physrlt.control import development_cases

    original = [
        {
            "suite": "libero_10",
            "task": 0,
            "state": 0,
            "perturbation": "object",
            "id": "old",
        }
    ]
    fresh = original[0] | {"state": 10, "id": "new", "split": "development"}
    path = tmp_path / "new.json"
    path.write_text(json.dumps([fresh]))
    assert development_cases(path, original) == [fresh]
    for change in ({"state": 0}, {"id": "old"}, {"split": "train"}, {"task": 3}):
        path.write_text(json.dumps([fresh | change]))
        with pytest.raises(ValueError):
            development_cases(path, original)
    path.write_text(json.dumps([fresh, fresh]))
    with pytest.raises(ValueError):
        development_cases(path, original)


def test_repeat_plan_has_matched_families_and_small_budgets(tmp_path):
    import json

    from toolkits.rlt.physrlt.campaign import prepare_repeat

    base = tmp_path / "pilot-v1"
    followup = tmp_path / "pilot-v2"
    followup.mkdir()
    (followup / "plan.json").write_text("{}")
    output = tmp_path / "repeat"
    prepare_repeat(
        output, base, tmp_path / "after", tmp_path / "source", tmp_path / "assets"
    )
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan["jobs"]) == 16
    for job in plan["jobs"]:
        args = job["args"]
        assert args[args.index("--online-ticks") + 1] == "1040"
        methods = args[args.index("--methods") + 1 :]
        assert methods in (["action_recon", "joint"], ["hybrid_recon", "hybrid_joint"])
    assert plan["new_development_layouts"] == list(range(10, 16))


def test_report_retains_invalidated_online_and_failure_costs(tmp_path):
    import json

    from toolkits.rlt.physrlt.report import build

    for version in (1, 2):
        root = tmp_path / f"pilot-v{version}"
        path = root / "task0-object/control-seed31"
        path.mkdir(parents=True)
        episodes = [
            {
                "method": "joint",
                "stage": "online_frozen_eval",
                "split": "development",
                "success": success,
                "ticks": ticks,
                "seconds": 1.0,
                "state_id": i,
            }
            for i, (success, ticks) in enumerate(((True, 100), (False, 200)))
        ]
        (path / "episodes.json").write_text(json.dumps(episodes))
        (path / "run_manifest.json").write_text(
            json.dumps({"control_protocol_version": version})
        )
    (tmp_path / "pilot-v1/INVALIDATION.json").write_text("{}")
    result = build(tmp_path)
    old, new = result["control"]
    assert old["evidence"] == "INVALIDATED_online_protocol"
    assert new["evidence"] == "development_only"
    assert new["mean_ticks_all"] == 150 and new["successes"] == 1
    assert new["episodes"] == new["unique_layouts"] == 2
    assert not new["run_complete"]
    assert (tmp_path / "overview/index.html").is_file()


def test_paired_action_effects_require_correct_effect_not_state_prediction():
    from toolkits.rlt.physrlt.counterfactual import effect_errors

    plus = {
        "response": torch.ones(2, 12),
        "geometry": torch.ones(2, 67),
        "horizon_mask": torch.tensor([True, False]),
    }
    minus = {
        key: value.clone() if key == "horizon_mask" else -value
        for key, value in plus.items()
    }
    predictions = {key: plus[key][None] for key in ("response", "geometry")}
    negative = {key: -value for key, value in predictions.items()}
    rows = effect_errors(
        plus, minus, predictions, negative, {"response_scale": torch.ones(12)}
    )
    assert len(rows) == 1 and rows[0]["horizon"] == 3
    assert rows[0]["response_effect_scaled_mse"] == rows[0]["object_effect_rmse_m"] == 0
    assert rows[0]["zero_effect_response_scaled_mse"] == 4
    assert rows[0]["zero_effect_object_rmse_m"] == 2
    unchanged = effect_errors(
        plus, minus, predictions, predictions, {"response_scale": torch.ones(12)}
    )
    assert unchanged[0]["response_effect_scaled_mse"] == 4


def test_quiet_stdout_with_new_episode_is_not_a_stalled_job(tmp_path):
    import os

    from toolkits.rlt.physrlt.campaign import job_stalled

    log = tmp_path / "stdout.log"
    log.write_text("old startup log")
    os.utime(log, (100, 100))
    assert job_stalled(tmp_path, 1000)
    episode = tmp_path / "episodes.json"
    episode.write_text("[]")
    os.utime(episode, (999, 999))
    assert not job_stalled(tmp_path, 1000)
    assert job_stalled(tmp_path, 2000)


def test_two_supervisors_cannot_claim_the_same_gpu(tmp_path):
    from toolkits.rlt.physrlt.campaign import claim_gpu

    first = claim_gpu(0, tmp_path)
    assert first is not None
    assert claim_gpu(0, tmp_path) is None
    second = claim_gpu(1, tmp_path)
    assert second is not None
    second.close()
    first.close()
    replacement = claim_gpu(0, tmp_path)
    assert replacement is not None
    replacement.close()


def test_gpu_lease_survives_parent_release_until_child_exits(tmp_path):
    import subprocess
    import sys

    from toolkits.rlt.physrlt.campaign import claim_gpu

    lease = claim_gpu(0, tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        pass_fds=(lease.fileno(),),
    )
    lease.close()
    try:
        assert claim_gpu(0, tmp_path) is None
        child.stdin.close()
        child.wait(timeout=5)
        available = claim_gpu(0, tmp_path)
        assert available is not None
        available.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
