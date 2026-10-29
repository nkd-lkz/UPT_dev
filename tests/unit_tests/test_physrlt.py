# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Physical readout causality, units, split isolation and experiment contracts."""

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from toolkits.rlt.physrlt import response_adaptation as response_adaptation_module
from toolkits.rlt.physrlt.campaign import prepare, ready_jobs
from toolkits.rlt.physrlt.conditioning import (
    CandidateConsequenceConditioner,
    SparseMotionConditioner,
    differentiable_env_commands,
    load_sparse_motion_conditioner,
)
from toolkits.rlt.physrlt.consequence_critic_gate import (
    load_replay as load_consequence_gate_replay,
)
from toolkits.rlt.physrlt.consequence_critic_gate import (
    parse_evaluations as parse_consequence_gate_evaluations,
)
from toolkits.rlt.physrlt.consequence_critic_gate import summarize_gate
from toolkits.rlt.physrlt.control import (
    attach_actor_view,
    deferred_update_summary,
    load_gated_online_checkpoint,
    merge_replay_views,
    parse_external_readouts,
    replay_trace_sha256,
    skip_deferred_updates,
    validate_fixed_replay,
)
from toolkits.rlt.physrlt.control import warmup_gate as physical_warmup_gate
from toolkits.rlt.physrlt.control_report import summarize as summarize_control_runs
from toolkits.rlt.physrlt.critic_audit import (
    binary_auc,
    discounted_success_returns,
    ranking_concordance,
)
from toolkits.rlt.physrlt.data import (
    check_split,
    compact_features,
    flow_path_features,
    masked_loss,
    measured_response_snapshot,
    metric_depth,
    projection,
    transition_targets,
)
from toolkits.rlt.physrlt.effect_probe import (
    paired_records,
)
from toolkits.rlt.physrlt.effect_probe import (
    run as run_effect_probe,
)
from toolkits.rlt.physrlt.episode_fold_gate import (
    complete_positive_episodes,
    leave_one_episode_out_folds,
)
from toolkits.rlt.physrlt.gain_probe import (
    realized_translation_score,
)
from toolkits.rlt.physrlt.gain_probe import (
    summarize as summarize_gain_probe,
)
from toolkits.rlt.physrlt.goal_ranking import distance_progress, pairwise_summary
from toolkits.rlt.physrlt.medium_gate import prepare as prepare_medium_gate
from toolkits.rlt.physrlt.medium_gate import summarize as summarize_medium_gate
from toolkits.rlt.physrlt.medium_gate import summarize_followups
from toolkits.rlt.physrlt.model import PhysicalReadout, SparseObjectMotionHead
from toolkits.rlt.physrlt.object_motion_probe import (
    dominant_object,
    fit_feature,
)
from toolkits.rlt.physrlt.probe import (
    crossfitted_response_residual,
    execution_response_loss,
    inputs,
    mismatched_targets,
    normalization,
    response_indices,
    run,
    tensor_data,
)
from toolkits.rlt.physrlt.response_adaptation import (
    AxisTranslationCalibrator,
    DiagonalResponseRegressor,
    NominalResponseRegressor,
    TranslationCalibrator,
    control_correction,
    response_context,
)
from toolkits.rlt.physrlt.response_adaptation import (
    rollout as rollout_response_adaptation,
)
from toolkits.rlt.physrlt.response_adaptation import (
    run as run_response_adaptation,
)
from toolkits.rlt.physrlt.response_matrix import CausalResponseMatrix
from toolkits.rlt.physrlt.response_matrix_control_gate import (
    evaluate as evaluate_matrix_gate,
)
from toolkits.rlt.physrlt.response_matrix_control_gate import (
    summarize as summarize_matrix_gate,
)
from toolkits.rlt.physrlt.sparse_motion_report import aggregate
from toolkits.rlt.physrlt.tick_selection import select_ticks
from toolkits.rlt.pi05_baseline import env_commands


def test_realized_translation_score_uses_issued_command_direction():
    commands = torch.zeros(3, 7)
    commands[:, 0] = 0.1
    response = torch.zeros(12)
    response[7] = 0.15
    assert realized_translation_score(commands, response) == pytest.approx(0.5)
    response[7] = -0.15
    assert realized_translation_score(commands, response) == pytest.approx(-0.5)


def test_causal_response_matrix_starts_at_identity_without_future_data():
    estimator = CausalResponseMatrix(ridge=1e-3)
    expected = torch.tensor([0.01, -0.02, 0.03])
    assert estimator.observations == 0
    assert torch.allclose(estimator.matrix(), torch.eye(3))
    assert torch.allclose(estimator.predict(expected), expected)
    assert torch.equal(estimator.support(), torch.zeros(3))
    assert estimator.token().shape == (12,)


def test_continuous_return_ranking_concordance_is_tie_aware():
    assert ranking_concordance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert ranking_concordance([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert ranking_concordance([1.0, 1.0], [0.0, 1.0]) == 0.5
    assert ranking_concordance([1.0, 2.0], [0.0, 0.0]) is None
    with pytest.raises(ValueError, match="equal length"):
        ranking_concordance([1.0], [1.0, 2.0])


def test_causal_response_matrix_recovers_coupled_execution_map():
    estimator = CausalResponseMatrix(ridge=1e-6)
    physical = torch.tensor([[0.5, 0.1, 0.0], [0.0, 1.2, -0.2], [0.1, 0.0, 0.8]])
    observations = torch.tensor(
        [[0.03, 0.00, 0.00], [0.00, 0.04, 0.00], [0.00, 0.00, 0.05]]
    )
    for expected in observations:
        estimator.observe(expected, expected @ physical)
    probe = torch.tensor([0.02, -0.01, 0.03])
    assert estimator.observations == 3
    assert torch.allclose(estimator.predict(probe), probe @ physical, atol=2e-4)
    assert torch.all(estimator.support() > 0)


def test_causal_response_matrix_returns_bounded_inverse_and_negative_control():
    adaptive = CausalResponseMatrix(ridge=1e-8)
    reversed_control = CausalResponseMatrix(ridge=1e-8)
    physical = torch.diag(torch.tensor([0.5, 1.0, 1.3]))
    observations = torch.eye(3) * 0.1
    assert torch.equal(adaptive.correction(), torch.eye(3))
    for expected in observations:
        measured = expected @ physical
        adaptive.observe(expected, measured)
        reversed_control.observe(expected, measured)
    np.testing.assert_allclose(
        adaptive.correction().numpy(), np.diag([1.3, 1.0, 0.8]), atol=1e-5
    )
    np.testing.assert_allclose(
        reversed_control.correction(reversed_control=True).numpy(),
        np.diag([0.8, 1.0, 1.3]),
        atol=1e-5,
    )


def test_causal_response_matrix_correction_handles_axis_coupling():
    estimator = CausalResponseMatrix(ridge=1e-8)
    physical = torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for expected in torch.eye(3) * 0.1:
        estimator.observe(expected, expected @ physical)
    correction = estimator.correction(singular_bounds=(0.5, 2.0))
    torch.testing.assert_close(correction @ physical, torch.eye(3), atol=2e-5, rtol=0)


@pytest.mark.parametrize(
    ("method", "expected_scale"),
    [("matrix_adaptive", 1.3), ("matrix_reversed", 0.8)],
)
def test_response_matrix_rollout_uses_only_completed_chunks(
    monkeypatch, method, expected_scale
):
    class Environment:
        def __init__(self):
            self.instruction = "move"
            self.position = torch.zeros(3)
            self.steps = 0

        def reset(self, case):
            del case
            self.position.zero_()
            self.steps = 0
            return {"state": np.zeros(8, np.float32)}

        def step(self, command):
            self.steps += 1
            if self.steps > 9:
                self.position += torch.as_tensor(command[:3])
            done = self.steps == 39
            return {"state": np.zeros(8, np.float32)}, float(done), done

    class Reference:
        def predict(self, obs, instruction, seed):
            del obs, instruction, seed
            commands = np.zeros((10, 7), np.float32)
            commands[:, 0] = 0.01
            return {"commands": commands}

    class Nominal:
        def predict(self, commands, proprio):
            del commands, proprio
            return torch.tensor([0.1, 0.0, 0.0])

    def snapshot(env):
        response = torch.zeros(12)
        response[7:10] = env.position
        return response

    monkeypatch.setattr(
        response_adaptation_module, "measured_response_snapshot", snapshot
    )
    monkeypatch.setattr(
        response_adaptation_module, "physical_reset_hash", lambda env: "reset"
    )
    result = rollout_response_adaptation(
        Environment(),
        Reference(),
        {"id": "case", "split": "confirmation", "seed": 1},
        method,
        np.array([0.5, 1.0, 1.0]),
        0.01,
        30,
        Nominal(),
    )
    assert result["success"] is True
    assert result["ticks"] == 30
    assert len(result["diagnostics"]) == 3
    np.testing.assert_allclose(result["diagnostics"][0]["correction"], np.eye(3))
    np.testing.assert_allclose(result["diagnostics"][1]["correction"], np.eye(3))
    assert result["diagnostics"][2]["correction"][0][0] == pytest.approx(
        expected_scale, abs=1e-5
    )


def test_response_matrix_control_gate_uses_success_then_capped_ticks():
    def episodes(scenario, method, successes, ticks):
        return [
            {
                "scenario": scenario,
                "case_id": str(index),
                "method": method,
                "success": index < successes,
                "ticks": tick,
            }
            for index, tick in enumerate(ticks)
        ]

    stress = []
    nominal = []
    for method, successes, ticks in (
        ("reference", 4, [200] * 6),
        ("adaptive", 4, [190] * 6),
        ("matrix_adaptive", 5, [180] * 6),
        ("matrix_reversed", 4, [210] * 6),
        ("fixed_high", 5, [175] * 6),
        ("delayed_high", 4, [190] * 6),
    ):
        stress.extend(episodes("stress", method, successes, ticks))
    for method, successes, ticks in (
        ("reference", 5, [200] * 6),
        ("adaptive", 5, [195] * 6),
        ("matrix_adaptive", 5, [205] * 6),
        ("matrix_reversed", 4, [210] * 6),
        ("fixed_high", 5, [190] * 6),
        ("delayed_high", 5, [195] * 6),
    ):
        nominal.extend(episodes("nominal", method, successes, ticks))
    rows, passed = evaluate_matrix_gate(stress, nominal)
    assert passed is True
    assert len(summarize_matrix_gate(stress, "stress")) == 6
    matrix = next(
        row
        for row in rows
        if row.get("scenario") == "stress" and row.get("method") == "matrix_adaptive"
    )
    assert matrix["success_rate"] == pytest.approx(5 / 6)
    failed = copy.deepcopy(stress)
    for row in failed:
        if row["method"] == "matrix_adaptive":
            row["success"] = False
    _, passed = evaluate_matrix_gate(failed, nominal)
    assert passed is False


@pytest.mark.parametrize(
    ("expected", "measured"),
    [
        (torch.zeros(2), torch.zeros(3)),
        (torch.zeros(3), torch.zeros(4)),
        (torch.tensor([0.0, float("nan"), 0.0]), torch.zeros(3)),
    ],
)
def test_causal_response_matrix_rejects_invalid_observations(expected, measured):
    with pytest.raises(ValueError):
        CausalResponseMatrix().observe(expected, measured)


def test_gain_probe_gate_uses_only_paired_development_records():
    records = []
    for split, count in (("train", 2), ("development", 6)):
        for state in range(count):
            for horizon in (3, 10):
                for gain, score in ((1.0, 1.0), (0.7, 0.7)):
                    records.append(
                        {
                            "split": split,
                            "case_id": f"{split}-{state}",
                            "start_tick": 10 * state,
                            "horizon": horizon,
                            "execution_gain": gain,
                            "translation_score": score,
                        }
                    )
    result = summarize_gain_probe(records, 0.7)
    assert result["development_gate"]["passed"]
    assert result["summaries"]["development"]["3"][
        "paired_order_accuracy"
    ] == pytest.approx(1.0)
    changed = copy.deepcopy(records)
    for row in changed:
        if row["split"] == "development" and row["execution_gain"] == 0.7:
            row["translation_score"] = 1.1
    assert not summarize_gain_probe(changed, 0.7)["development_gate"]["passed"]


def test_gain_probe_does_not_pass_without_development_pairs():
    records = [
        {
            "split": "train",
            "case_id": "train-0",
            "start_tick": 0,
            "horizon": horizon,
            "execution_gain": gain,
            "translation_score": score,
        }
        for horizon in (3, 10)
        for gain, score in ((1.0, 1.0), (0.7, 0.7))
    ]
    assert not summarize_gain_probe(records, 0.7)["development_gate"]["passed"]


def test_translation_calibrator_is_causal_bounded_and_has_negative_control():
    commands = torch.zeros(10, 7)
    commands[:, 0] = 0.1
    low_response = torch.zeros(12)
    low_response[7] = 0.006
    adaptive = TranslationCalibrator(0.01)
    reversed_control = TranslationCalibrator(0.01, reversed_control=True)
    assert adaptive.correction() == reversed_control.correction() == 1.0
    for calibrator in (adaptive, reversed_control):
        calibrator.observe(commands, low_response)
        assert calibrator.correction() == 1.0
        calibrator.observe(commands, low_response)
    assert adaptive.correction() == pytest.approx(1.3)
    assert reversed_control.correction() == pytest.approx(0.8)


def test_vector_calibrator_removes_the_known_issued_scale():
    calibrator = TranslationCalibrator(0.01)
    nominal = torch.tensor([0.1, 0.0, 0.0])
    measured = nominal * 1.3 * 0.7
    for _ in range(2):
        assert calibrator.observe_vector(nominal, measured, 1.3) == pytest.approx(0.7)
    assert calibrator.correction() == pytest.approx(1.3)


def test_axis_calibrator_separates_anisotropic_execution_response():
    nominal = torch.tensor([0.1, 0.2, 0.3])
    issued_scale = np.array([1.3, 1.0, 0.8])
    physical_gain = torch.tensor([0.5, 1.0, 1.3])
    measured = nominal * torch.as_tensor(issued_scale) * physical_gain
    adaptive = AxisTranslationCalibrator(prior_energy=1e-8)
    reversed_control = AxisTranslationCalibrator(
        reversed_control=True, prior_energy=1e-8
    )
    for calibrator in (adaptive, reversed_control):
        for _ in range(2):
            np.testing.assert_allclose(
                calibrator.observe(nominal, measured, issued_scale),
                physical_gain,
                atol=1e-6,
            )
    np.testing.assert_allclose(adaptive.correction(), [1.3, 1.0, 0.8])
    np.testing.assert_allclose(reversed_control.correction(), [0.8, 1.0, 1.3])


def test_axis_calibrator_deadband_preserves_nominal_axes():
    calibrator = AxisTranslationCalibrator(prior_energy=1e-8)
    nominal = torch.ones(3)
    for _ in range(2):
        calibrator.observe(nominal, torch.tensor([0.79, 0.81, 1.19]), np.ones(3))
    np.testing.assert_allclose(calibrator.correction(), [1 / 0.79, 1, 1])


def test_response_controls_separate_feedback_from_fixed_action_scaling():
    calibrator = TranslationCalibrator(0.01)
    assert control_correction("reference", calibrator, 0.7) == 1
    assert control_correction("adaptive", calibrator, 0.7) == 1
    assert control_correction("fixed_high", calibrator, 1.0) == 1.3
    assert control_correction("delayed_high", calibrator, 0.7) == 1
    assert control_correction("delayed_oracle", calibrator, 0.7) == 1
    calibrator.observations = 2
    assert control_correction("delayed_high", calibrator, 0.7) == 1.3
    assert control_correction("delayed_oracle", calibrator, 0.7) == 1.3
    assert control_correction("delayed_oracle", calibrator, 1.5) == 0.8
    assert control_correction("oracle", calibrator, 0.85) == pytest.approx(1 / 0.85)
    assert control_correction("oracle", calibrator, 1.5) == 0.8
    with pytest.raises(ValueError, match="Unknown"):
        control_correction("missing", calibrator, 0.7)


def test_response_adaptation_rejects_the_wrong_environment_before_loading(
    monkeypatch,
):
    monkeypatch.delenv("LIBERO_TYPE", raising=False)
    with pytest.raises(RuntimeError, match="LIBERO_TYPE=pro"):
        run_response_adaptation(SimpleNamespace())


def test_nominal_response_regressor_uses_train_and_validation_only(tmp_path):
    rows = []
    for split, count, offset in (("train", 30, 0), ("validation", 12, 30)):
        for index in range(count):
            value = (index + offset) / 50
            commands = torch.zeros(10, 7)
            commands[:, 0] = 0.1
            prop = torch.zeros(8)
            prop[0] = value
            score = 0.01 + 0.004 * value
            response = torch.zeros(2, 12)
            response[1, 7] = score
            rows.append(
                {
                    "split": split,
                    "commands": commands,
                    "prop": prop,
                    "response": response,
                    "horizon_mask": torch.tensor([True, True]),
                }
            )
    poisoned = copy.deepcopy(rows[0])
    poisoned.update(split="development")
    poisoned["response"][1, 7] = 1e6
    confirmation = copy.deepcopy(poisoned)
    confirmation.update(split="confirmation")
    dataset = tmp_path / "transitions.pt"
    torch.save({"rows": rows + [poisoned, confirmation]}, dataset)
    model, metadata = NominalResponseRegressor.fit(dataset)
    assert metadata["development_rows_read"] == 0
    commands = torch.zeros(10, 7)
    commands[:, 0] = 0.1
    prop = torch.zeros(8)
    prop[0] = 0.5
    assert response_context(commands, prop).shape == (58,)
    prediction = model.predict(commands, prop)
    assert prediction[0] == pytest.approx(0.012, abs=5e-4)
    assert prediction[1:].abs().max() < 1e-6
    assert metadata["validation_rmse"] < metadata["zero_motion_validation_rmse"]
    assert metadata["validation_cosine"] > 0
    assert metadata["evaluation_rows_read"] == 0


def test_diagonal_response_regressor_fits_physical_gain_without_development(tmp_path):
    rows = []
    physical_gain = torch.tensor([0.01, 0.02, 0.03])
    for split, count in (("train", 30), ("validation", 12)):
        for index in range(count):
            commands = torch.zeros(10, 7)
            commands[:, :3] = torch.tensor([0.1, 0.2, 0.3]) + index * 1e-3
            displacement = commands[:, :3].sum(0) * physical_gain
            response = torch.zeros(2, 12)
            response[1, 7:10] = displacement
            rows.append(
                {
                    "split": split,
                    "commands": commands,
                    "response": response,
                    "horizon_mask": torch.tensor([True, True]),
                }
            )
    poisoned = copy.deepcopy(rows[0])
    poisoned.update(split="development")
    poisoned["response"][1, 7:10] = 1e6
    dataset = tmp_path / "transitions.pt"
    torch.save({"rows": rows + [poisoned]}, dataset)
    model, metadata = DiagonalResponseRegressor.fit(dataset)
    prediction = model.predict(rows[0]["commands"], torch.zeros(8))
    torch.testing.assert_close(prediction, rows[0]["response"][1, 7:10])
    assert metadata["development_rows_read"] == 0
    assert metadata["validation_gain"] == pytest.approx([1, 1, 1])


def test_measured_response_snapshot_uses_deployable_robot_state_only():
    raw = {
        "robot0_joint_pos": np.arange(7, dtype=np.float32),
        "robot0_eef_pos": np.arange(3, dtype=np.float32) + 7,
        "robot0_gripper_qpos": np.arange(2, dtype=np.float32) + 10,
    }
    environment = SimpleNamespace(env=SimpleNamespace(current_raw_obs=[raw]))
    torch.testing.assert_close(
        measured_response_snapshot(environment), torch.arange(12).float()
    )


def test_flow_path_feature_uses_fixed_generation_points_and_width():
    hidden = [torch.full((10, 16), float(index)) for index in range(10)]
    feature = flow_path_features(hidden)
    assert feature.shape == (512,)
    expected = [0, 3, 6, 9]
    for chunk, index in zip(feature.reshape(4, 128), expected, strict=True):
        torch.testing.assert_close(
            chunk, torch.full((10, 16), float(index)).mean(0) @ projection(16)
        )
    with pytest.raises(ValueError, match="at least four"):
        flow_path_features(hidden[:3])


def test_sparse_object_motion_probe_selects_training_object_and_crossfits():
    def records(split: str) -> list[dict]:
        result = []
        for case in range(3):
            for index in range(4):
                moving = index % 2 == 0
                value = 0.004 * (1 + 0.1 * case) if moving else 0.0
                effect = torch.tensor([value, value / 2, 0.0])
                branches = {}
                for sign in (-1, 1):
                    geometry = torch.zeros(2, 70)
                    geometry[1, 64:67] = sign * effect / 2
                    branches[sign] = {"geometry": geometry}
                prop = torch.zeros(8)
                prop[0] = float(moving)
                feature_value = float(moving) + case * 0.1
                result.append(
                    {
                        "case_id": f"{split}-{case}",
                        "branches": branches,
                        "row": {
                            "prop": prop,
                            "ref": torch.zeros(10, 7),
                            "action_feature": torch.full((512,), feature_value),
                            "flow_feature": torch.full((512,), feature_value),
                            "prefix_feature": torch.full((512,), feature_value),
                            "object_keys": ["active_pos", "static_pos"],
                        },
                    }
                )
        return result

    train = records("train")
    evaluation = records("development")
    assert dominant_object(train) == 0
    with pytest.raises(ValueError, match="no measurable effect"):
        dominant_object(train, rank=1)
    secondary = copy.deepcopy(train)
    for record in secondary:
        for branch in record["branches"].values():
            branch["geometry"][:, 67:70] = branch["geometry"][:, 64:67] * 0.5
    assert dominant_object(secondary, rank=1) == 1
    summary = fit_feature(train, evaluation, "flow_path", 0)
    assert summary["evaluation_pairs"] == 12
    assert summary["evaluation_moving_pairs"] == 6
    assert summary["gated_rmse_m"] < summary["zero_motion_rmse_m"]
    assert summary["event_auc"] == 1


def test_sparse_object_motion_head_exposes_stable_gated_outputs():
    token = torch.randn(2, 128)
    commands = torch.randn(2, 10, 7)
    gated = SparseObjectMotionHead(gated=True)
    event_logit, conditional, expected = gated(token, commands, horizon=3)
    assert event_logit.shape == (2,)
    assert conditional.shape == (2, 3)
    assert expected.shape == (2, 3)
    torch.testing.assert_close(expected, event_logit.sigmoid()[:, None] * conditional)
    expected.square().mean().backward()
    assert all(parameter.grad is not None for parameter in gated.parameters())

    direct = SparseObjectMotionHead(gated=False)
    event_logit, conditional, expected = direct(token, commands, horizon=10)
    assert event_logit is None
    torch.testing.assert_close(conditional, expected)


def test_sparse_motion_checkpoint_is_bound_to_readout_version(tmp_path):
    head = SparseObjectMotionHead(gated=True)
    path = tmp_path / "sparse.pt"
    torch.save(
        {
            "schema": 1,
            "gated": True,
            "arm": "gated_pair",
            "readout_sha256": "readout-v1",
            "model": head.state_dict(),
        },
        path,
    )
    conditioner, payload = load_sparse_motion_conditioner(
        path, expected_readout_sha256="readout-v1"
    )
    assert isinstance(conditioner, SparseMotionConditioner)
    assert payload["arm"] == "gated_pair"
    with pytest.raises(ValueError, match="versions differ"):
        load_sparse_motion_conditioner(path, expected_readout_sha256="readout-v2")


def test_sparse_motion_shared_axes_retains_train_provenance(tmp_path):
    import json

    from toolkits.rlt.physrlt.sparse_motion import run as run_sparse

    rows = synthetic_rows()
    for index, row in enumerate(rows):
        row["start_tick"] = index % 4 * 10
    dataset = tmp_path / "transitions.pt"
    torch.save({"rows": rows}, dataset)
    run(
        Namespace(
            dataset=dataset,
            output=tmp_path / "readouts",
            methods=["state_action_joint"],
            seeds=[3],
            steps=2,
            threads=1,
        )
    )
    for split, selected in (
        ("train", [rows[0], rows[4]]),
        ("development", [rows[16], rows[20]]),
    ):
        for axis in (0, 1):
            directory = tmp_path / f"{split}-axis{axis}"
            directory.mkdir()
            measurements = [
                {
                    "case": {"id": row["case_id"]},
                    "start_tick": row["start_tick"],
                    "branches": {
                        sign: {
                            "horizon_mask": torch.ones(2, dtype=torch.bool),
                            "geometry": row["geometry"] + sign * 0.01,
                            "response": row["response"] + sign * 0.01,
                        }
                        for sign in (-1, 1)
                    },
                }
                for row in selected
            ]
            torch.save(measurements, directory / "measurements.pt")
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "no_training": True,
                        "axis": axis,
                        "offset": 0.05,
                    }
                )
            )
    args = Namespace(
        dataset=dataset,
        evaluation_dataset=None,
        checkpoint=tmp_path / "readouts/state_action_joint-seed3/readout.pt",
        train_measurements=tmp_path / "train-axis0/measurements.pt",
        train_result=tmp_path / "train-axis0/result.json",
        additional_train_pairs=[tmp_path / "train-axis1"],
        evaluation_measurements=tmp_path / "development-axis1/measurements.pt",
        evaluation_result=tmp_path / "development-axis1/result.json",
        evaluation_split="development",
        output=tmp_path / "heads",
        steps=2,
        seed=3,
        cpu_threads=1,
        motion_threshold_m=0.002,
        arms=["gated_pair"],
        object_rank=0,
    )
    result = run_sparse(args)
    assert result["training_axes"] == [0, 1]
    assert result["train_pairs"] == 4 and result["evaluation_pairs"] == 2
    payload = torch.load(args.output / "gated_pair.pt", weights_only=True)
    assert payload["training_axes"] == [0, 1] and "axis" not in payload
    assert len(payload["all_train_measurements_sha256"]) == 2
    args.output = tmp_path / "duplicate"
    args.additional_train_pairs = [tmp_path / "train-axis0"]
    with pytest.raises(ValueError, match="Duplicate training intervention axis"):
        run_sparse(args)


def test_sparse_motion_conditioner_is_relative_and_has_zero_control():
    token = torch.randn(2, 128)
    reference = torch.zeros(2, 10, 7)
    candidate = reference.clone()
    candidate[:, :, 1] = 0.2
    head = SparseObjectMotionHead(gated=True)
    conditioned = SparseMotionConditioner(head)(token, candidate, reference)
    control = SparseMotionConditioner(head, zero_channel=True)(
        token, candidate, reference
    )
    assert conditioned.shape == control.shape == (2, 256)
    torch.testing.assert_close(conditioned[:, :128], token)
    torch.testing.assert_close(control[:, :128], token)
    assert conditioned[:, 128:].abs().sum() > 0
    assert control[:, 128:].count_nonzero() == 0
    matched = SparseMotionConditioner(head)(token, reference, reference)
    assert matched[:, 128:].count_nonzero() == 0


def test_goal_progress_and_pairwise_ranking_preserve_physical_direction():
    source = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([1.0, 0.0, 0.0])
    toward = distance_progress(
        source, target, torch.tensor([0.1, 0.0, 0.0]), torch.zeros(3)
    )
    away = distance_progress(
        source, target, torch.tensor([-0.1, 0.0, 0.0]), torch.zeros(3)
    )
    assert toward == pytest.approx(0.1)
    assert away == pytest.approx(-0.1)
    rows = [
        {
            "actual_progress_difference_m": 0.2,
            "predicted_sign": 1,
            "oracle_sign": 1,
            "selected_actual_progress_m": 0.1,
            "oracle_actual_progress_m": 0.1,
            "random_actual_progress_m": 0.0,
        },
        {
            "actual_progress_difference_m": -0.1,
            "predicted_sign": 1,
            "oracle_sign": -1,
            "selected_actual_progress_m": -0.05,
            "oracle_actual_progress_m": 0.05,
            "random_actual_progress_m": 0.0,
        },
        {
            "actual_progress_difference_m": 1e-5,
            "predicted_sign": -1,
            "oracle_sign": 1,
            "selected_actual_progress_m": 0.0,
            "oracle_actual_progress_m": 1e-5,
            "random_actual_progress_m": 5e-6,
        },
    ]
    result = pairwise_summary(rows, tie_threshold_m=1e-4)
    assert result["pairs"] == 3
    assert result["resolvable_pairs"] == 2
    assert result["rank_accuracy_pct"] == 50
    assert result["constant_sign_accuracy_pct"] == {"-1": 50, "1": 50}
    assert result["selected_advantage_vs_random_mm"] == pytest.approx(25)
    assert result["oracle_regret_mm"] == pytest.approx(50)


def test_goal_candidate_comparison_exposes_constant_direction_and_ties():
    from toolkits.rlt.physrlt.goal_ranking import candidate_summary, sum_object_progress

    common = {
        "evaluation_dataset_sha256": "data",
        "evaluation_split": "development",
        "source_object_key": "source",
        "target_object_key": "target",
        "readout_sha256": "readout",
        "sparse_arm": "gated_pair",
        "offset": 0.1,
    }
    results = []
    for axis in (0, 1):
        rows = []
        for tick in (10, 20):
            scores = {"-1": 0.0, "1": 0.01 if axis == 1 and tick == 10 else 0.0}
            rows.append(
                {
                    "case_id": "case",
                    "start_tick": tick,
                    "origin_sha256": str(tick),
                    "actual_progress_m": scores,
                    "predicted_progress_m": scores,
                }
            )
        results.append(common | {"axis": axis, "rows": rows})
    summary = candidate_summary(results, 0.0001)
    assert summary["states"] == 2
    assert summary["resolvable_states"] == 1
    assert summary["resolvable_accuracy_pct"] == 100
    assert summary["fixed_candidates"]["1:1"]["resolvable_accuracy_pct"] == 100
    assert summary["advantage_over_best_fixed_envelope_mm"] == 0
    other_object = copy.deepcopy(results[1])
    other_object["source_object_key"] = "second"
    results[1]["sparse_head_sha256"] = "first-head"
    other_object["sparse_head_sha256"] = "second-head"
    combined = sum_object_progress([results[1], other_object])
    assert combined["rows"][0]["actual_progress_m"]["1"] == pytest.approx(0.02)
    assert combined["source_heads_sha256"] == ["first-head", "second-head"]
    altered = copy.deepcopy(results)
    altered[1]["rows"][0]["origin_sha256"] = "wrong-origin"
    with pytest.raises(ValueError, match="origins differ"):
        candidate_summary(altered, 0.0001)
    altered = copy.deepcopy(results)
    altered[1]["rows"].pop()
    with pytest.raises(ValueError, match="state sets differ"):
        candidate_summary(altered, 0.0001)


def test_transport_prior_is_bounded_and_inputs_exclude_measured_outcomes():
    from toolkits.rlt.physrlt.transport_probe import (
        response_features,
        transport_coefficient,
    )

    response = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    effects = torch.tensor(
        [[0.5, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(
        transport_coefficient(response, effects),
        torch.tensor([[0.5], [0.0], [0.0], [1.0]]),
    )
    row = synthetic_rows()[0]
    record = {
        "row": row,
        "branches": {
            sign: {
                "commands": row["commands"] + sign * 0.01,
                "response": row["response"].clone(),
            }
            for sign in (-1, 1)
        },
    }
    before = response_features([record])
    record["row"]["geometry"][:] = 999
    for branch in record["branches"].values():
        branch["response"][:] = 999
    torch.testing.assert_close(before, response_features([record]), atol=0, rtol=0)


def test_sparse_token_allows_nonflow_readouts_without_flow_artifacts():
    from toolkits.rlt.physrlt.sparse_motion import _token

    class Encoder:
        def encode(self, action, proprioception, reference):
            return action + reference.mean((1, 2))[:, None] + proprioception

    row = synthetic_rows()[0]
    row.pop("flow_feature")
    row["action_feature"] = torch.ones(2)
    row["prefix_feature"] = torch.ones(2)
    row["prop"] = torch.ones(2)
    row["ref"] = torch.ones(1, 1)
    readout = SimpleNamespace(
        model=Encoder(),
        payload={
            "method": "joint",
            "stats": {
                "action_feature_mean": torch.zeros(2),
                "action_feature_scale": torch.ones(2),
                "prop_mean": torch.zeros(2),
                "prop_scale": torch.ones(2),
            },
        },
    )
    assert _token(readout, {"row": row}).shape == (1, 2)


def test_paired_case_batches_cover_layouts_once_and_preserve_split():
    from toolkits.rlt.physrlt.counterfactual import select_cases

    cases = [{"id": "train", "split": "train"}] + [
        {"id": str(i), "split": "development"} for i in range(10)
    ]
    first = select_cases(cases, "development", 5)
    second = select_cases(cases, "development", 5, 5)
    assert first + second == cases[1:]
    assert select_cases(cases, "train", 6) == cases[:1]
    with pytest.raises(ValueError, match="empty"):
        select_cases(cases, "development", 5, 10)
    with pytest.raises(ValueError, match="nonnegative"):
        select_cases(cases, "development", 5, -1)


def test_transport_uncertainty_clusters_layouts_not_correlated_rows():
    from toolkits.rlt.physrlt.transport_probe import (
        active_object_labels,
        apply_transport_jacobian,
        apply_transport_residual,
        apply_transport_vector,
        assert_disjoint_records,
        classification_layout_comparison,
        classification_report,
        gated_coefficient,
        layout_comparison,
        layout_ensemble_uncertainty,
        nested_layout_uncertainty,
        out_of_layout_gated_predictions,
        out_of_layout_predictions,
        phase_expand,
        phase_ratio_tensor,
        select_incremental_ratio,
        select_phase_incremental_ratios,
        select_residual_ratio,
        select_uncertainty_trust,
        stratified_scores,
        transport_jacobian,
        transport_residual,
        transport_vector,
        uncertainty_trust_tensor,
    )

    target = torch.zeros(4, 3)
    direct = torch.tensor([[0.001] * 3] * 3 + [[0.003] * 3])
    arms = dict.fromkeys(
        ("direct", "constant_coupling", "state_action_coupling", "shuffled_coupling"),
        direct,
    )
    arms["coupled"] = target.clone()
    report = layout_comparison(arms, target, ["a", "a", "a", "b"])
    comparison = report["coupled_vs"]["direct"]
    assert comparison["mse_improvement_mm2"] == pytest.approx(5.0)
    assert comparison["layouts_improved"] == 2
    assert comparison["exploratory_layout_bootstrap_95pct"] == pytest.approx([1, 9])
    with pytest.raises(ValueError, match="layout identity"):
        layout_comparison(arms, target, ["a"])

    def record(case, reset):
        return {"case_id": case, "row": {"reset_hash": reset}}

    audit = assert_disjoint_records(
        [record("train", "train-reset")],
        [record("evaluation", "evaluation-reset")],
    )
    assert audit["train_cases"] == ["train"]
    with pytest.raises(ValueError, match="layouts overlap"):
        assert_disjoint_records(
            [record("same", "train-reset")],
            [record("same", "evaluation-reset")],
        )
    with pytest.raises(ValueError, match="reset hashes overlap"):
        assert_disjoint_records(
            [record("train", "same-reset")],
            [record("evaluation", "same-reset")],
        )

    features = torch.tensor([[1.0], [2.0], [3.0]])
    phased = phase_expand(
        features,
        [
            {"start_tick": 40},
            {"start_tick": 140},
            {"start_tick": 240},
        ],
    )
    assert phased.shape == (3, 7)
    torch.testing.assert_close(phased[:, 1:4], torch.eye(3))
    assert not torch.equal(
        phased,
        phase_expand(
            features,
            [
                {"start_tick": 40},
                {"start_tick": 140},
                {"start_tick": 240},
            ],
            shuffle_seed=31,
        ),
    )

    records = []
    for first, second in ((0.0, 0.0), (0.002, 0.0), (0.0, 0.003)):
        geometry = torch.zeros(2, 70)
        branches = {}
        for sign in (-1, 1):
            branch_geometry = geometry.clone()
            branch_geometry[-1, 64] = sign * first / 2
            branch_geometry[-1, 67] = sign * second / 2
            branches[sign] = {"geometry": branch_geometry}
        records.append({"branches": branches})
    labels = active_object_labels(records, [0, 1], 0.0001)
    assert labels.tolist() == [0, 1, 2]
    report = classification_report(torch.eye(3), labels)
    assert report["accuracy_pct"] == 100
    assert report["macro_recall_pct"] == 100
    scores = {
        "action_side": torch.eye(3),
        "state_action": torch.zeros(3, 3),
        "time_phase": torch.zeros(3, 3),
        "shuffled_labels": torch.zeros(3, 3),
        "fixed_train_majority": torch.zeros(3, 3),
    }
    comparison = classification_layout_comparison(scores, labels, ["a", "a", "b"])
    assert comparison["action_side_vs"]["state_action"][
        "accuracy_advantage_pct"
    ] == pytest.approx(75)
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.double)
    y = 2 * x + 1
    prediction = out_of_layout_predictions(x, y, ["a", "a", "b", "b"], 0.1)
    assert prediction.shape == y.shape
    assert torch.isfinite(prediction).all()
    with pytest.raises(ValueError, match="multiple layouts"):
        out_of_layout_predictions(x, y, ["a"] * 4, 0.1)
    coefficient, audit = gated_coefficient(
        x,
        x,
        torch.tensor([[0.0], [0.5], [0.0], [1.0]], dtype=torch.double),
        torch.tensor([False, True, False, True]),
        ["a", "a", "b", "b"],
    )
    assert coefficient.shape == (4, 1)
    assert torch.all((0 <= coefficient) & (coefficient <= 1))
    assert audit["moving_rows"] == 2
    all_moving, moving_audit = gated_coefficient(
        x,
        x,
        torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.double),
        torch.ones(4, dtype=torch.bool),
        ["a", "a", "b", "b"],
    )
    assert all_moving.shape == (4, 1)
    assert moving_audit["event_cv"] == {"constant_label": 1}
    no_motion, stationary_audit = gated_coefficient(
        x,
        x,
        torch.zeros(4, 1, dtype=torch.double),
        torch.zeros(4, dtype=torch.bool),
        ["a", "a", "b", "b"],
    )
    torch.testing.assert_close(no_motion, torch.zeros(4, 1, dtype=torch.double))
    assert stationary_audit["event_cv"] == {"constant_label": 0}
    gated_oof = out_of_layout_gated_predictions(
        torch.arange(8, dtype=torch.double)[:, None],
        torch.tensor(
            [[0.0], [0.0], [0.5], [0.5], [0.7], [0.7], [1.0], [1.0]],
            dtype=torch.double,
        ),
        torch.tensor([False, False, True, True, True, True, True, True]),
        ["a", "a", "b", "b", "c", "c", "d", "d"],
    )
    assert gated_oof.shape == (8, 1)
    assert torch.isfinite(gated_oof).all()
    ratio, losses = select_incremental_ratio(
        torch.tensor([[0.2], [0.4]], dtype=torch.double),
        torch.tensor([[0.2], [-0.2]], dtype=torch.double),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.double),
        torch.tensor([[0.4, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.double),
    )
    assert ratio == 1.0
    assert losses["1.0"] == 0.0
    ensemble_x = torch.arange(8, dtype=torch.double)[:, None]
    ensemble_y = torch.cat((ensemble_x, 2 * ensemble_x), dim=1)
    ensemble_cases = [f"layout{index // 2}" for index in range(8)]
    nested_uncertainty = nested_layout_uncertainty(
        ensemble_x, ensemble_y, ensemble_cases, 1e-6
    )
    evaluation_uncertainty = layout_ensemble_uncertainty(
        ensemble_x, ensemble_y, ensemble_cases, ensemble_x[:2], 1e-6
    )
    assert nested_uncertainty.shape == (8, 1)
    assert evaluation_uncertainty.shape == (2, 1)
    assert (nested_uncertainty >= 0).all()
    selection, selection_losses = select_uncertainty_trust(
        torch.zeros(8, 1, dtype=torch.double),
        torch.ones(8, 1, dtype=torch.double),
        nested_uncertainty,
        torch.ones(8, 3, dtype=torch.double),
        torch.ones(8, 3, dtype=torch.double),
    )
    assert selection["ratio"] == 1.0
    assert selection["uncertainty_scale"] > 0
    assert len(selection_losses) == 20
    trust = uncertainty_trust_tensor(evaluation_uncertainty, selection)
    assert trust.shape == (2, 1)
    assert (trust >= 0).all() and (trust <= 1).all()
    phase_records = [{"start_tick": tick} for tick in (40, 90, 140, 190, 240, 290)]
    phase_ratio, phase_audit = select_phase_incremental_ratios(
        torch.full((6, 1), 0.2, dtype=torch.double),
        torch.full((6, 1), 0.2, dtype=torch.double),
        torch.tensor([[1.0, 0.0, 0.0]] * 6, dtype=torch.double),
        torch.full((6, 3), 0.0, dtype=torch.double).index_fill(
            1, torch.tensor([0]), 0.4
        ),
        phase_records,
    )
    assert phase_ratio == {"early": 1.0, "late": 1.0, "mid": 1.0}
    assert set(phase_audit) == {"early", "late", "mid"}
    torch.testing.assert_close(
        phase_ratio_tensor(
            [{"start_tick": 40}, {"start_tick": 140}],
            {"mid": 0.5},
            torch.zeros(2, 1, dtype=torch.double),
        ),
        torch.tensor([[0.0], [0.5]], dtype=torch.double),
    )
    stratified = stratified_scores(
        {"candidate": torch.zeros(3, 3, dtype=torch.double)},
        torch.ones(3, 3, dtype=torch.double),
        [
            {
                "start_tick": tick,
                "branches": {
                    -1: {"commands": torch.zeros(2, 3)},
                    1: {
                        "commands": torch.nn.functional.one_hot(
                            torch.tensor([axis, axis]), 3
                        ).float()
                    },
                },
            }
            for tick, axis in ((40, 0), (140, 1), (240, 1))
        ],
    )
    assert stratified["axis"]["axis1"]["pairs"] == 2
    assert stratified["phase"]["late"]["pairs"] == 1
    response = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    effect = torch.tensor([[0.0, 0.5, 0.0], [0.0, 2.0, 0.0]])
    jacobian = transport_jacobian(response, effect)
    transported = apply_transport_jacobian(jacobian, response)
    torch.testing.assert_close(
        transported,
        torch.tensor([[0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]),
    )
    vector = transport_vector(response, effect)
    torch.testing.assert_close(
        apply_transport_vector(vector, response),
        torch.tensor([[0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]),
    )
    residual = transport_residual(response, effect)
    torch.testing.assert_close(residual, effect)
    torch.testing.assert_close(
        apply_transport_residual(torch.zeros_like(effect), residual, response),
        torch.tensor([[0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]),
    )
    ratio, losses = select_residual_ratio(
        torch.zeros_like(effect), residual, response, effect
    )
    assert ratio == 1.0
    assert losses["1.0"] < losses["0.0"]
    torch.testing.assert_close(
        apply_transport_residual(
            torch.zeros_like(effect), residual, response, max_response_ratio=0
        ),
        torch.zeros_like(effect),
    )
    with pytest.raises(ValueError, match="nonnegative"):
        apply_transport_residual(
            torch.zeros_like(effect), residual, response, max_response_ratio=-1
        )


def test_sparse_motion_report_keeps_all_seeds_and_rejects_duplicates(tmp_path):
    paths = []
    for seed, rmse in ((31, 0.008), (47, 0.007)):
        payload = {
            "state": "complete",
            "axis": 1,
            "readout_method": "flow_joint",
            "evaluation_split": "confirmation",
            "evaluation_dataset_sha256": "same",
            "seed": seed,
            "results": {
                "gated_pair": {
                    "zero_motion_rmse_m": 0.01,
                    "rmse_m": rmse,
                    "improvement_vs_zero_pct": 100 * (0.01 - rmse) / 0.01,
                    "event_auc": 0.9,
                    "effect_cosine": 0.5,
                    "sign_reversed_prediction_rmse_m": 0.012,
                }
            },
        }
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    result = aggregate(paths)
    assert result["source_runs"] == 2
    assert result["rows"][0]["rmse_mm_mean"] == pytest.approx(7.5)
    assert result["rows"][0]["improvement_pct_min"] == pytest.approx(20)
    assert result["rows"][0]["all_seeds_beat_zero"]
    with pytest.raises(ValueError, match="repeats"):
        aggregate([paths[0], paths[0]])


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
                    "flow_feature": torch.randn(512, generator=rng),
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


def test_discounted_success_returns_stop_at_tick_cap_and_terminal():
    replay = [
        {"ticks": 10, "terminal": False, "reward": 0.0},
        {"ticks": 10, "terminal": False, "reward": 0.0},
        {"ticks": 5, "terminal": True, "reward": 5.0},
        {"ticks": 10, "terminal": False, "reward": 0.0},
        {"ticks": 10, "terminal": False, "reward": 0.0},
    ]
    returns = discounted_success_returns(replay, gamma=0.9, episode_tick_cap=20)
    assert returns == [0.0, 0.0, 5.0, 0.0, 0.0]


def test_discounted_success_returns_propagate_within_successful_episode():
    replay = [
        {"ticks": 10, "terminal": False, "reward": 0.0},
        {"ticks": 5, "terminal": True, "reward": 5.0},
        {"ticks": 10, "terminal": False, "reward": 0.0},
    ]
    returns = discounted_success_returns(replay, gamma=0.9)
    assert returns == pytest.approx([5 * 0.9**10, 5.0, 0.0])


def test_binary_auc_is_tie_aware_and_requires_both_classes():
    assert binary_auc([0.1, 0.5, 0.2], [False, True, False]) == 1
    assert binary_auc([0.1, 0.1], [True, False]) == 0.5
    assert binary_auc([0.1], [True]) is None
    with pytest.raises(ValueError, match="equal length"):
        binary_auc([0.1], [True, False])


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
    confirmation = copy.deepcopy(rows[0])
    confirmation.update(state_id=50, reset_hash="confirmation", split="confirmation")
    check_split(rows + [confirmation])
    confirmation["split"] = "final-test"
    with pytest.raises(ValueError, match="Unknown physical-data split"):
        check_split(rows + [confirmation])


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


def test_candidate_consequences_match_commands_and_preserve_action_gradients():
    torch.manual_seed(23)
    actions = torch.linspace(-2, 2, 140).reshape(2, 10, 7).requires_grad_(True)
    commands = differentiable_env_commands(actions)
    expected = torch.from_numpy(
        np.stack([env_commands(row.detach().numpy()) for row in actions])
    )
    torch.testing.assert_close(commands.detach(), expected)
    model = PhysicalReadout(12, 67)
    conditioner = CandidateConsequenceConditioner(model, "joint", mode="mixed")
    token = torch.randn(2, 128)
    result = conditioner(token, actions, torch.zeros_like(actions))
    assert result.shape == token.shape
    result.square().mean().backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad[..., :6].abs().sum() > 0
    assert actions.grad[..., 6].abs().sum() > 0
    assert not any(parameter.requires_grad for parameter in conditioner.parameters())


def test_relative_candidate_consequences_are_zero_at_reference():
    torch.manual_seed(27)
    model = PhysicalReadout(12, 67)
    conditioner = CandidateConsequenceConditioner(model, "joint", mode="relative")
    token = torch.randn(2, 128)
    reference = torch.randn(2, 10, 7)
    result = conditioner(token, reference, reference)
    torch.testing.assert_close(result[:, :64], token[:, :64])
    assert result[:, 64:].count_nonzero() == 0
    candidate = (reference + 0.1).requires_grad_(True)
    changed = conditioner(token, candidate, reference)
    assert changed[:, 64:].count_nonzero() > 0
    changed.square().mean().backward()
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0


def test_consequence_concat_preserves_the_complete_base_token():
    torch.manual_seed(31)
    model = PhysicalReadout(12, 67)
    token = torch.randn(2, 128)
    reference = torch.randn(2, 10, 7)
    candidate = reference + 0.1
    control = CandidateConsequenceConditioner(model, "joint", mode="zero_concat")
    physical = CandidateConsequenceConditioner(model, "joint", mode="relative_concat")
    unit_physical = CandidateConsequenceConditioner(
        model, "joint", mode="relative_unit_concat"
    )
    control_result = control(token, candidate, reference)
    physical_result = physical(token, candidate, reference)
    assert control_result.shape == physical_result.shape == (2, 256)
    torch.testing.assert_close(control_result[:, :128], token)
    torch.testing.assert_close(physical_result[:, :128], token)
    assert control_result[:, 128:].count_nonzero() == 0
    assert physical_result[:, 128:].count_nonzero() > 0
    unit_result = unit_physical(token, candidate, reference)
    torch.testing.assert_close(unit_result[:, :128], token)
    torch.testing.assert_close(unit_result[:, 128:196].square().mean(-1), torch.ones(2))
    assert unit_result[:, 196:].count_nonzero() == 0
    unchanged = unit_physical(token, reference, reference)
    assert unchanged[:, 128:].count_nonzero() == 0
    separate_source = unit_physical(token, candidate, reference, token + 1)
    torch.testing.assert_close(separate_source[:, :128], token)
    assert not torch.allclose(separate_source[:, 128:], unit_result[:, 128:])


def test_fixed_replay_views_require_the_same_executed_trace():
    base = [
        {
            "z": torch.zeros(128),
            "ref": torch.zeros(10, 7),
            "prop": torch.zeros(8),
            "action": torch.zeros(10, 7),
            "accepted_commands": torch.zeros(10, 7),
            "next_z": torch.ones(128),
            "next_ref": torch.zeros(10, 7),
            "next_prop": torch.zeros(8),
            "ticks": 10,
            "terminal": False,
            "reward": 0.0,
        }
    ]
    physical = copy.deepcopy(base)
    physical[0]["z"] += 2
    merged = merge_replay_views(base, physical)
    torch.testing.assert_close(merged[0]["z"], base[0]["z"])
    torch.testing.assert_close(merged[0]["critic_z"], physical[0]["z"])
    assert "critic_z" not in base[0]
    matched = copy.deepcopy(base)
    matched[0]["critic_z"] = torch.full((128,), 3.0)
    matched[0]["next_critic_z"] = torch.full((128,), 4.0)
    merged = merge_replay_views(base, matched)
    torch.testing.assert_close(merged[0]["critic_z"], matched[0]["critic_z"])
    torch.testing.assert_close(
        merged[0]["next_critic_z"], matched[0]["next_critic_z"]
    )
    physical[0]["action"][0, 0] = 0.1
    with pytest.raises(ValueError, match="traces differ"):
        merge_replay_views(base, physical)


def test_gated_online_checkpoint_requires_exact_provenance(tmp_path):
    from toolkits.rlt.physrlt.control import file_sha256

    base = tmp_path / "base.pt"
    consequence = tmp_path / "consequence.pt"
    base.write_bytes(b"base")
    consequence.write_bytes(b"physical")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = {
        "state_dict": {"value": torch.tensor(1)},
        "gate_passed": True,
        "arm": "correct",
        "optimizer_seed": 31,
        "train_trace_sha256": "trace",
        "base_readout_sha256": file_sha256(base),
        "consequence_readout_sha256": file_sha256(consequence),
    }
    torch.save(payload, checkpoint)
    loaded = load_gated_online_checkpoint(
        checkpoint,
        optimizer_seed=31,
        base_readout=base,
        consequence_readout=consequence,
    )
    assert loaded["train_trace_sha256"] == "trace"
    with pytest.raises(ValueError, match="optimizer seed differs"):
        load_gated_online_checkpoint(
            checkpoint,
            optimizer_seed=47,
            base_readout=base,
            consequence_readout=consequence,
        )
    payload["gate_passed"] = False
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="did not pass"):
        load_gated_online_checkpoint(
            checkpoint,
            optimizer_seed=31,
            base_readout=base,
            consequence_readout=consequence,
        )


def test_consequence_gate_requires_explicit_matched_replay_views(tmp_path):
    base = [
        {
            "z": torch.zeros(128),
            "ref": torch.zeros(10, 7),
            "prop": torch.zeros(8),
            "action": torch.zeros(10, 7),
            "accepted_commands": torch.zeros(10, 7),
            "next_z": torch.ones(128),
            "next_ref": torch.zeros(10, 7),
            "next_prop": torch.zeros(8),
            "ticks": 10,
            "terminal": False,
            "reward": 0.0,
        }
    ]
    physical = copy.deepcopy(base)
    physical[0]["critic_z"] = torch.full((128,), 2.0)
    physical[0]["next_critic_z"] = torch.full((128,), 3.0)
    base_path, physical_path = tmp_path / "base.pt", tmp_path / "physical.pt"
    torch.save(base, base_path)
    torch.save(physical, physical_path)
    parsed = parse_consequence_gate_evaluations(
        [f"heldout={base_path},{physical_path}"]
    )
    assert parsed == {"heldout": (base_path, physical_path)}
    merged = load_consequence_gate_replay(base_path, physical_path)
    torch.testing.assert_close(merged[0]["z"], base[0]["z"])
    torch.testing.assert_close(merged[0]["critic_z"], physical[0]["critic_z"])
    physical[0]["action"][0, 0] = 0.1
    torch.save(physical, physical_path)
    with pytest.raises(ValueError, match="traces differ"):
        load_consequence_gate_replay(base_path, physical_path)


def test_consequence_critic_gate_requires_correct_arm_best_in_every_seed():
    rows = []
    for seed in (31, 47, 59):
        for arm, auc, concordance in (
            ("correct", 0.8, 0.7),
            ("reversed", 0.6, 0.5),
            ("zero", 0.55, 0.45),
        ):
            rows.append(
                {
                    "evaluation": "heldout",
                    "optimizer_seed": seed,
                    "arm": arm,
                    "return_ranking_auc": auc,
                    "return_order_concordance": concordance,
                }
            )
    summaries, passes = summarize_gate(rows)
    assert passes
    assert len(summaries) == 8
    assert all(row["correct_best"] for row in summaries)
    next(
        row
        for row in rows
        if row["optimizer_seed"] == 47 and row["arm"] == "reversed"
    )["return_order_concordance"] = 0.9
    summaries, passes = summarize_gate(rows)
    assert not passes
    failed = [row for row in summaries if not row["correct_best"]]
    assert any(
        row["optimizer_seed"] == 47
        and row["metric"] == "return_order_concordance"
        for row in failed
    )
    next(
        row
        for row in rows
        if row["optimizer_seed"] == 31 and row["arm"] == "correct"
    )["return_ranking_auc"] = None
    _, passes = summarize_gate(rows)
    assert not passes


def test_episode_fold_gate_excludes_open_tail_and_holds_out_unique_traces():
    replay = []
    for episode in range(3):
        for step in range(2):
            replay.append(
                {
                    "z": torch.full((128,), float(episode)),
                    "ref": torch.zeros(10, 7),
                    "prop": torch.zeros(8),
                    "action": torch.full((10, 7), 0.01 * episode),
                    "accepted_commands": torch.full((10, 7), 0.01 * episode),
                    "next_z": torch.full((128,), float(episode + 1)),
                    "next_ref": torch.zeros(10, 7),
                    "next_prop": torch.zeros(8),
                    "ticks": 10,
                    "terminal": step == 1,
                    "reward": 5.0 if step == 1 else 0.0,
                }
            )
    open_tail = copy.deepcopy(replay[0])
    open_tail.update(terminal=False, reward=0.0, ticks=7)
    replay.append(open_tail)
    episodes = complete_positive_episodes(replay)
    assert [len(episode) for episode in episodes] == [2, 2, 2]
    folds = leave_one_episode_out_folds(replay)
    assert len(folds) == 3
    assert all(len(fold["train"]) == 4 for fold in folds)
    assert all(len(fold["evaluation"]) == 2 for fold in folds)
    assert all(
        fold["train_summary"]["trace_sha256"]
        != fold["evaluation_summary"]["trace_sha256"]
        for fold in folds
    )


def test_episode_fold_gate_requires_three_complete_successful_episodes():
    replay = []
    for episode in range(2):
        row = {
            "z": torch.full((128,), float(episode)),
            "ref": torch.zeros(10, 7),
            "prop": torch.zeros(8),
            "action": torch.full((10, 7), 0.01 * episode),
            "accepted_commands": torch.full((10, 7), 0.01 * episode),
            "next_z": torch.full((128,), float(episode + 1)),
            "next_ref": torch.zeros(10, 7),
            "next_prop": torch.zeros(8),
            "ticks": 10,
            "terminal": True,
            "reward": 5.0,
        }
        replay.append(row)
    with pytest.raises(ValueError, match="at least three"):
        leave_one_episode_out_folds(replay)


def test_actor_replay_view_has_a_capacity_matched_control():
    base = [
        {
            "z": torch.zeros(128),
            "next_z": torch.ones(128),
            "critic_z": torch.full((128,), 2.0),
            "next_critic_z": torch.full((128,), 3.0),
            "ref": torch.zeros(10, 7),
            "next_ref": torch.ones(10, 7) * 0.1,
        }
    ]
    control = attach_actor_view(base, "zero_concat")
    physical = attach_actor_view(base, "physical_concat")
    readout = SimpleNamespace(
        consequence=SimpleNamespace(
            model=PhysicalReadout(12, 67), payload={"method": "joint"}
        )
    )
    outcomes = attach_actor_view(base, "physical_outcomes_concat", readout)
    assert control[0]["actor_z"].shape == physical[0]["actor_z"].shape == (256,)
    assert outcomes[0]["actor_z"].shape == (256,)
    torch.testing.assert_close(control[0]["actor_z"][:128], base[0]["z"])
    assert control[0]["actor_z"][128:].count_nonzero() == 0
    torch.testing.assert_close(physical[0]["actor_z"][128:], base[0]["critic_z"])
    torch.testing.assert_close(
        physical[0]["next_actor_z"][128:], base[0]["next_critic_z"]
    )
    assert outcomes[0]["actor_z"][128:].count_nonzero() > 0
    with pytest.raises(ValueError, match="lacks consequence tokens"):
        attach_actor_view(
            [{"z": torch.zeros(128), "next_z": torch.zeros(128)}], "physical_concat"
        )


def test_external_readout_labels_are_explicit_and_unique(tmp_path):
    expected = (tmp_path / "effect.pt").resolve()
    assert parse_external_readouts([f"effect_candidate={expected}"]) == {
        "effect_candidate": expected
    }
    for invalid in (["effect.pt"], ["=effect.pt"], ["effect="], ["a=x", "a=y"]):
        with pytest.raises(ValueError, match="unique LABEL=PATH"):
            parse_external_readouts(invalid)


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
            methods=["action_recon", "joint", "flow_recon", "flow_joint"],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    assert len(results) == 4
    assert len({result["trainable_parameters"] for result in results}) == 1
    assert "NOT control improvement" in results[0]["interpretation"]
    assert (tmp_path / "probe/table.csv").is_file()
    assert (tmp_path / "probe/joint-seed3/readout.pt").is_file()
    assert (tmp_path / "probe/flow_joint-seed3/readout.pt").is_file()


def test_medium_gate_is_disjoint_and_pre_registered(tmp_path):
    plan = prepare_medium_gate(
        tmp_path / "medium",
        tmp_path,
        tmp_path,
        tmp_path / "python",
        Path.cwd(),
        2,
    )
    inputs = tmp_path / "medium/inputs"
    physical = json.loads((inputs / "physical.json").read_text())
    control = json.loads((inputs / "control_development.json").read_text())
    assert {row["state"] for row in physical} == {*range(10), 30, 31}
    assert {row["state"] for row in control} == {32, 33, 34, 35}
    assert not ({row["state"] for row in physical} & {row["state"] for row in control})
    assert plan["benchmark_cell"] == "task5-object"
    assert plan["budgets"]["online_control_ticks"] == 1040
    assert plan["matched_attribution"]["effect_axis0"] == "pointwise_axis0"
    names = [job["name"] for job in plan["jobs"]]
    assert names[-1] == "physical-control"
    assert "baseline-actor-eval" in names
    probe = next(job for job in plan["jobs"] if job["name"] == "physical-probes")
    methods = probe["argv"][probe["argv"].index("--methods") + 1 :]
    assert methods == [
        "action_recon",
        "joint",
        "shuffled_joint",
        "state_action_joint",
    ]
    assert probe["env"]["PYTHONPATH"] == str(Path.cwd())
    assert "LD_LIBRARY_PATH" not in probe["env"]


def test_residual_warmup_gate_accepts_exact_reference_without_lucky_success():
    fidelity = {
        "after": {"arm_mse": 0.0, "gripper_disagreement": 0.0},
    }
    failed_rollouts = [{"success": False} for _ in range(4)]
    assert not physical_warmup_gate(fidelity, failed_rollouts, False)[0]
    assert physical_warmup_gate(fidelity, failed_rollouts, True)[0]
    fidelity["after"]["arm_mse"] = 0.03
    assert not physical_warmup_gate(fidelity, failed_rollouts, True)[0]


def test_medium_gate_summary_exposes_matched_gain(tmp_path):
    root = tmp_path / "medium"
    root.mkdir()
    (root / "plan.json").write_text('{"benchmark_cell":"task5-object"}')
    for relative, successes in (
        ("baseline/reference_eval", 2),
        ("baseline/actor_eval", 3),
    ):
        output = root / relative
        output.mkdir(parents=True)
        episodes = [
            {"success": index < successes, "ticks": 100 + index} for index in range(4)
        ]
        (output / "episodes.json").write_text(json.dumps({"episodes": episodes}))
    output = root / "physical/control_seed31"
    output.mkdir(parents=True)
    records = []
    for method, successes, ticks in (
        ("action_recon", 2, 200),
        ("joint", 2, 180),
        ("pointwise_axis0", 2, 200),
        ("effect_axis0", 3, 220),
    ):
        records.extend(
            {
                "method": method,
                "stage": "online_frozen_eval",
                "split": "development",
                "success": index < successes,
                "ticks": ticks,
            }
            for index in range(4)
        )
        records.extend(
            {
                "method": method,
                "stage": "online_frozen_eval",
                "split": "validation",
                "success": index < 3,
                "ticks": ticks,
            }
            for index in range(4)
        )
    (output / "episodes.json").write_text(json.dumps(records))
    rows = summarize_medium_gate(root)
    keyed = {row["method"]: row for row in rows}
    assert keyed["Physical latent"]["status"] == "advance"
    assert keyed["Paired-effect Physical Token"]["delta_vs_matched_pp"] == 25
    assert keyed["Paired-effect Physical Token"]["status"] == "advance"
    assert (root / "comparison.csv").is_file()


def test_medium_gate_summary_reads_parallel_control_shards(tmp_path):
    root = tmp_path / "medium"
    root.mkdir()
    (root / "plan.json").write_text('{"benchmark_cell":"task5-object"}')
    for directory, method in (
        ("control_direct_seed31", "joint"),
        ("control_effect_seed31", "effect_axis0"),
    ):
        output = root / "physical" / directory
        output.mkdir(parents=True)
        rows = [
            {
                "method": method,
                "stage": "online_frozen_eval",
                "split": "development",
                "success": True,
                "ticks": 100,
            }
        ]
        (output / "episodes.json").write_text(json.dumps(rows))
    rows = summarize_medium_gate(root)
    keyed = {row["method"]: row for row in rows}
    assert keyed["Physical latent"]["success_rate_pct"] == 100
    assert keyed["Paired-effect Physical Token"]["success_rate_pct"] == 100


def test_control_report_keeps_axes_denominators_and_skipped_updates(tmp_path):
    runs = []
    for axis, skipped in ((0, False), (1, True)):
        run = tmp_path / f"axis{axis}"
        run.mkdir()
        rows = []
        rows.extend(
            {
                "case_id": f"validation-{index}",
                "reset_hash": f"reset-validation-{index}",
                "method": "reference",
                "stage": "frozen",
                "split": "validation",
                "success": index == 0,
                "ticks": 130 + index,
                "seconds": 13 + index,
            }
            for index in range(2)
        )
        for method in ("zero", "physical"):
            rows.extend(
                {
                    "case_id": f"validation-{index}",
                    "reset_hash": f"reset-validation-{index}",
                    "method": method,
                    "stage": "warmup",
                    "split": "development",
                    "success": index == 0,
                    "ticks": 100 + index,
                }
                for index in range(2)
            )
            rows.extend(
                {
                    "case_id": f"validation-{index}",
                    "reset_hash": f"reset-validation-{index}",
                    "method": method,
                    "stage": "online_frozen_eval",
                    "split": "validation",
                    "success": index < axis + 1,
                    "ticks": 120 + index,
                    "seconds": 12 + index,
                    "candidate_selection_calls": 2,
                    "candidate_nonreference_fraction": 0.25 * (index + 1),
                    "candidate_scale_mean": 0.125 * (index + 1),
                    "candidate_score_improvement_mean": 0.01 * (index + 1),
                }
                for index in range(2)
            )
            (run / f"{method}-gate.json").write_text('{"passed": true}')
            (run / f"{method}-deferred.json").write_text(
                json.dumps(
                    {
                        "positive_transitions": 0 if skipped else 3,
                        "updates": 0 if skipped else 20,
                        "updates_skipped": skipped,
                        "skip_reason": "no_positive_transition" if skipped else None,
                    }
                )
            )
        (run / "episodes.json").write_text(json.dumps(rows))
        (run / "result.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "state": "complete",
                    "seed": 31,
                    "correction_axes": [axis],
                    "online_tick_cap": 1040,
                }
            )
        )
        runs.append(run)
    result = summarize_control_runs(runs)
    assert len(result["rows"]) == 4
    keyed = {(row["correction_axes"], row["method"]): row for row in result["rows"]}
    assert keyed[("0", "physical")]["online_validation_episodes"] == 2
    assert keyed[("0", "physical")]["online_validation_success_rate_pct"] == 50
    assert keyed[("1", "physical")]["online_validation_success_rate_pct"] == 100
    assert keyed[("1", "physical")][
        "online_validation_candidate_nonreference_fraction_mean"
    ] == pytest.approx(0.375)
    assert keyed[("1", "physical")][
        "online_validation_candidate_selection_calls_total"
    ] == 4
    assert keyed[("1", "physical")][
        "reference_validation_success_rate_pct"
    ] == 50
    assert keyed[("1", "physical")][
        "paired_validation_success_rate_delta_pct"
    ] == 50
    assert keyed[("1", "physical")][
        "paired_validation_failure_capped_ticks_delta_mean"
    ] == -10
    assert keyed[("1", "physical")]["online_validation_ticks_per_wall_second"] == (
        241 / 25
    )
    assert keyed[("1", "physical")]["updates_skipped"] is True
    assert keyed[("1", "physical")]["skip_reason"] == "no_positive_transition"


def test_control_report_rejects_unpaired_reference_resets(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "case_id": "case",
            "reset_hash": "reference",
            "method": "reference",
            "stage": "frozen",
            "split": "validation",
            "success": True,
            "ticks": 10,
        },
        {
            "case_id": "case",
            "reset_hash": "different",
            "method": "physical",
            "stage": "online_frozen_eval",
            "split": "validation",
            "success": True,
            "ticks": 10,
        },
    ]
    (run / "episodes.json").write_text(json.dumps(rows))
    (run / "result.json").write_text(
        '{"complete":true,"state":"complete","seed":31}'
    )
    with pytest.raises(ValueError, match="reset hashes differ"):
        summarize_control_runs([run])


def test_tick_selection_reads_training_geometry_only():
    rows = []
    for split, cases in (("train", 3), ("development", 1)):
        for case in range(cases):
            for tick, motion in ((10, 0.01), (20, 0.02), (30, 0.03)):
                if split == "development":
                    motion = 1.0 if tick == 10 else 0.0
                geometry = torch.zeros(2, 70)
                geometry[-1, 64] = motion
                rows.append(
                    {
                        "split": split,
                        "case_id": f"{split}-{case}",
                        "start_tick": tick,
                        "horizon_mask": torch.ones(2, dtype=torch.bool),
                        "geometry": geometry,
                    }
                )
    result = select_ticks({"rows": rows}, count=2, minimum_cases=2, minimum_spacing=10)
    assert result["selected_ticks"] == [20, 30]
    assert result["training_cases"] == 3
    assert result["role"].startswith("train-only")


def test_followup_summary_separates_actor_update_protocols(tmp_path):
    root = tmp_path / "medium"
    settings = (
        ("control_direct_seed31", False, False, False, False, 1),
        ("control_residual_seed31", True, False, False, False, 3),
        ("control_deferred_seed31", True, True, False, False, 2),
        ("control_gated_seed31", True, True, True, False, 4),
        ("control_fixed_seed31", True, True, True, True, 2),
    )
    for name, residual, deferred, positive_gate, fixed, successes in settings:
        output = root / "physical" / name
        output.mkdir(parents=True)
        rows = []
        for stage in ("warmup", "online_frozen_eval"):
            rows.extend(
                {
                    "method": "joint",
                    "stage": stage,
                    "split": "development",
                    "success": index < successes,
                    "ticks": 100 + index,
                }
                for index in range(4)
            )
            if fixed:
                rows.extend(
                    {
                        "method": "action_fixed",
                        "stage": stage,
                        "split": "development",
                        "success": index < 1,
                        "ticks": 120 + index,
                    }
                    for index in range(4)
                )
        (output / "episodes.json").write_text(json.dumps(rows))
        (output / "result.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "state": "complete",
                    "seed": 31,
                    "residual_output": residual,
                    "deferred_online_updates": deferred,
                    "require_positive_replay": positive_gate,
                    "fixed_replays": (
                        {"joint": "joint.pt", "action_fixed": "action.pt"}
                        if fixed
                        else {}
                    ),
                }
            )
        )
        (output / "joint-gate.json").write_text('{"passed":true}')
        if deferred:
            (output / "joint-deferred.json").write_text(
                json.dumps(
                    {
                        "updates": 4,
                        "transitions": 4,
                        "positive_transitions": 0 if positive_gate else 1,
                        "updates_skipped": positive_gate,
                        "trace_sha256": "a" * 64,
                    }
                )
            )
        if fixed:
            (output / "action_fixed-gate.json").write_text('{"passed":true}')
            (output / "action_fixed-deferred.json").write_text(
                json.dumps(
                    {
                        "updates": 4,
                        "transitions": 4,
                        "positive_transitions": 0,
                        "updates_skipped": True,
                        "trace_sha256": "a" * 64,
                    }
                )
            )
    rows = summarize_followups(root)
    keyed = {(row["protocol"], row["method"]): row for row in rows}
    assert keyed[("direct_action", "joint")]["online_dev_successes"] == 1
    assert keyed[("zero_residual", "joint")]["online_dev_successes"] == 3
    assert keyed[("deferred_residual", "joint")]["online_dev_successes"] == 2
    assert keyed[("deferred_residual", "joint")]["matched_collection_seeds"] == 0
    gated = keyed[("positive_gated_deferred_residual", "joint")]
    assert gated["online_dev_successes"] == 4
    assert gated["zero_positive_replay_seeds"] == 1
    assert gated["updates_skipped_seeds"] == 1
    fixed_joint = keyed[("fixed_positive_replay_residual", "joint")]
    assert fixed_joint["online_dev_successes"] == 2
    assert fixed_joint["delta_dev_vs_action_recon_pp"] == 25
    assert (root / "followup_comparison.csv").is_file()


def test_followup_summary_rejects_unmatched_deferred_collection(tmp_path):
    root = tmp_path / "medium"
    for method, trace in (("action_recon", "a" * 64), ("joint", "b" * 64)):
        output = root / "physical" / f"control_{method}_seed31"
        output.mkdir(parents=True)
        (output / "episodes.json").write_text("[]")
        (output / "result.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "seed": 31,
                    "residual_output": True,
                    "deferred_online_updates": True,
                }
            )
        )
        (output / f"{method}-gate.json").write_text('{"passed":true}')
        (output / f"{method}-deferred.json").write_text(
            json.dumps({"transitions": 4, "trace_sha256": trace})
        )
    with pytest.raises(ValueError, match="Deferred collection traces differ"):
        summarize_followups(root)


def test_followup_summary_reports_only_seed_matched_control_delta(tmp_path):
    root = tmp_path / "medium"
    output = root / "physical" / "control_direct_seed31"
    output.mkdir(parents=True)
    rows = []
    for method, successes, ticks in (
        ("action_recon", 1, 500),
        ("joint", 2, 450),
    ):
        rows.extend(
            {
                "method": method,
                "stage": "online_frozen_eval",
                "split": split,
                "success": index < successes,
                "ticks": ticks,
            }
            for split in ("development", "validation")
            for index in range(4)
        )
        (output / f"{method}-gate.json").write_text('{"passed":true}')
    (output / "episodes.json").write_text(json.dumps(rows))
    (output / "result.json").write_text(json.dumps({"complete": True, "seed": 31}))
    keyed = {row["method"]: row for row in summarize_followups(root)}
    assert keyed["joint"]["evaluated_seed_ids"] == "31"
    assert keyed["joint"]["delta_dev_vs_action_recon_pp"] == 25
    assert keyed["joint"]["delta_validation_vs_action_recon_pp"] == 25
    assert keyed["joint"]["delta_dev_ticks_vs_action_recon"] == -50


def test_response_redundancy_ablation_keeps_gripper_and_matched_capacity(tmp_path):
    assert response_indices("joint") == tuple(range(12))
    assert response_indices("arm_q_gripper_joint") == (*range(7), 10, 11)
    assert response_indices("eef_gripper_joint") == (7, 8, 9, 10, 11)
    path = tmp_path / "synthetic.pt"
    torch.save({"rows": synthetic_rows()}, path)
    results = run(
        Namespace(
            dataset=path,
            output=tmp_path / "response-ablation",
            methods=["arm_q_gripper_joint", "eef_gripper_joint"],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    assert results[0]["trainable_parameters"] == results[1]["trainable_parameters"]
    assert results[0]["response_supervision"] == [
        "arm_joint_position",
        "gripper_position",
    ]
    assert results[1]["response_supervision"] == [
        "eef_position",
        "gripper_position",
    ]
    groups = results[0]["horizons"]["3"]["response"]["token_and_action"]["groups"]
    assert set(groups) == {
        "arm_joint_position",
        "eef_position",
        "gripper_position",
    }


def test_shuffled_physical_targets_always_use_another_episode():
    torch.manual_seed(7)
    episodes = torch.tensor([0, 0, 1, 1, 2, 2])
    index = torch.arange(len(episodes))
    mapping = mismatched_targets(index, episodes)
    assert set(mapping) == set(index.tolist())
    assert all(
        episodes[source] != episodes[target] for source, target in mapping.items()
    )


def test_balanced_response_loss_gives_each_physical_group_equal_weight():
    prediction = torch.zeros(1, 2, 12)
    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 2, dtype=torch.bool)
    target[..., :7] = 1
    target[..., 7:10] = 2
    target[..., 10:] = 3
    loss = execution_response_loss(prediction, target, mask, "balanced_joint")
    assert loss.item() == pytest.approx((1 + 4 + 9) / 3)
    dimension_weighted = execution_response_loss(prediction, target, mask, "joint")
    assert dimension_weighted.item() == pytest.approx((7 + 3 * 4 + 2 * 9) / 12)
    assert loss != dimension_weighted


def test_paired_effect_probe_crossfits_layout_labels(tmp_path):
    rows = synthetic_rows()
    for index, row in enumerate(rows):
        row["start_tick"] = index % 4 * 10
    dataset = tmp_path / "synthetic.pt"
    torch.save({"rows": rows}, dataset)
    probe_root = tmp_path / "probes"
    run(
        Namespace(
            dataset=dataset,
            output=probe_root,
            methods=["joint"],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    measurements = []
    for row in (rows[0], rows[4]):
        branches = {}
        for sign in (-1, 1):
            branches[sign] = {
                "horizon_mask": row["horizon_mask"].clone(),
                "response": row["response"] + sign * 0.01,
                "geometry": row["geometry"] + sign * 0.02,
            }
        measurements.append(
            {
                "case": {"id": row["case_id"]},
                "start_tick": row["start_tick"],
                "branches": branches,
                "predictions": {},
            }
        )
    measurement_path = tmp_path / "measurements.pt"
    torch.save(measurements, measurement_path)
    paired_result = tmp_path / "paired_result.json"
    paired_result.write_text(
        '{"state":"complete","no_training":true,"axis":0,"offset":0.05,'
        '"paired_states":2}'
    )
    joined = paired_records({"rows": rows}, measurements, axis=0, offset=0.05)
    assert len(joined) == 2
    from toolkits.rlt.physrlt.control import FrozenReadout
    from toolkits.rlt.physrlt.sparse_motion import evaluate_object_bank, fit_head

    sparse_records = copy.deepcopy(joined)
    for record in sparse_records:
        record["row"]["flow_feature"] = record["row"]["action_feature"].clone()
        for branch in record["branches"].values():
            branch["horizon_mask"][0] = False
    readout = FrozenReadout(probe_root / "joint-seed3/readout.pt")
    first, _ = fit_head(readout, sparse_records, 0, "gated_pair", 7, 2, 0.002)
    for record in sparse_records:
        for branch in record["branches"].values():
            branch["geometry"][0] = 1e6
    second, _ = fit_head(readout, sparse_records, 0, "gated_pair", 7, 2, 0.002)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    bank_records = copy.deepcopy(joined)
    for record in bank_records:
        for branch in record["branches"].values():
            branch["geometry"] = torch.cat(
                (branch["geometry"], torch.zeros(2, 3)), dim=1
            )
    heads = [SparseObjectMotionHead(gated=True) for _ in range(2)]
    for head in heads:
        for parameter in head.parameters():
            parameter.data.zero_()
    bank = evaluate_object_bank(readout, heads, [0, 1], bank_records, 0.002)
    assert bank["accuracy_pct"] == 0
    assert bank["fixed_dominant_accuracy_pct"] == 100
    assert bank["effect_rmse_mm"] == bank["zero_effect_rmse_mm"]
    assert all(row["predicted_label"] == -1 for row in bank["rows"])
    torch.testing.assert_close(
        joined[0]["branches"][1]["commands"][:, 0]
        - joined[0]["branches"][-1]["commands"][:, 0],
        torch.full((10,), 0.1),
    )
    saturated = copy.deepcopy(rows)
    saturated[0]["ref"][:, 0] = 100
    saturated[4]["ref"][:, 0] = 100
    with pytest.raises(ValueError, match="clipping"):
        paired_records({"rows": saturated}, measurements, axis=0, offset=0.05)
    result = run_effect_probe(
        Namespace(
            dataset=dataset,
            measurements=measurement_path,
            paired_result=paired_result,
            checkpoint=probe_root / "joint-seed3/readout.pt",
            output=tmp_path / "effect-probe",
            seed=7,
            steps=2,
        )
    )
    assert result["state"] == "complete" and len(result["folds"]) == 2
    for fold in result["folds"]:
        assert fold["heldout_paired_case"] not in fold["training_paired_cases"]
        assert set(fold["arms"]) == {
            "base",
            "pointwise",
            "effect",
            "reversed_effect",
            "object_effect",
            "reversed_object_effect",
        }
        for arm in fold["arms"].values():
            assert arm["summary"]["10"]["paired_states"] == 1
    paired_result.write_text(
        '{"state":"complete","no_training":true,"axis":0,"offset":0.05,'
        '"paired_states":1}'
    )
    with pytest.raises(ValueError, match="count differs"):
        run_effect_probe(
            Namespace(
                dataset=dataset,
                measurements=measurement_path,
                paired_result=paired_result,
                checkpoint=probe_root / "joint-seed3/readout.pt",
                output=tmp_path / "invalid-effect-probe",
                seed=7,
                steps=2,
            )
        )
    evaluation_measurements = []
    for row in (rows[16], rows[20]):
        evaluation_measurements.append(
            {
                "case": {"id": row["case_id"]},
                "start_tick": row["start_tick"],
                "branches": {
                    sign: {
                        "horizon_mask": row["horizon_mask"].clone(),
                        "response": row["response"] + sign * 0.01,
                        "geometry": row["geometry"] + sign * 0.02,
                    }
                    for sign in (-1, 1)
                },
                "predictions": {},
            }
        )
    evaluation_path = tmp_path / "evaluation_measurements.pt"
    torch.save(evaluation_measurements, evaluation_path)
    evaluation_result = tmp_path / "evaluation_result.json"
    evaluation_result.write_text(
        '{"state":"complete","no_training":true,"split":"development",'
        '"axis":0,"offset":0.05,"paired_states":2}'
    )
    paired_result.write_text(
        '{"state":"complete","no_training":true,"axis":0,"offset":0.05,'
        '"paired_states":2}'
    )
    transfer = run_effect_probe(
        Namespace(
            dataset=dataset,
            measurements=measurement_path,
            paired_result=paired_result,
            evaluation_measurements=evaluation_path,
            evaluation_paired_result=evaluation_result,
            checkpoint=probe_root / "joint-seed3/readout.pt",
            output=tmp_path / "effect-transfer",
            seed=7,
            steps=2,
            parameter_scope="decoders",
        )
    )
    assert transfer["evaluation_paired_states"] == 2
    assert len(transfer["folds"]) == 1
    assert transfer["folds"][0]["heldout_paired_case"] == "development"
    for arm in (
        "pointwise",
        "effect",
        "reversed_effect",
        "object_effect",
        "reversed_object_effect",
    ):
        checkpoint = tmp_path / "effect-transfer/checkpoints" / f"{arm}.pt"
        assert checkpoint.is_file()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["effect_supervision"] == arm
        assert payload["effect_steps"] == 2
        assert payload["effect_parameter_scope"] == "decoders"
        assert payload["effect_measurements_sha256"]
        restored = PhysicalReadout(
            payload["response_dim"],
            payload["geometry_dim"],
            factorized=payload["factorized"],
        )
        restored.load_state_dict(payload["model"], strict=True)
        original = torch.load(
            probe_root / "joint-seed3/readout.pt",
            map_location="cpu",
            weights_only=True,
        )["model"]
        for name, value in payload["model"].items():
            if name.startswith(("encoder.", "reconstruct.")):
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_response_control_plan_is_bc_only_and_capacity_matched(tmp_path):
    import json

    from toolkits.rlt.physrlt.campaign import prepare_response_check

    base = tmp_path / "base"
    probes = tmp_path / "probes"
    for task, perturbation in ((0, "object"), (3, "lan")):
        name = f"task{task}-{perturbation}"
        (base / name / "collection").mkdir(parents=True)
        for method in ("joint", "arm_q_gripper_joint", "eef_gripper_joint"):
            path = probes / name / "probes" / f"{method}-seed31"
            path.mkdir(parents=True)
            (path / "readout.pt").touch()
    output = tmp_path / "output"
    prepare_response_check(output, base, probes, tmp_path, tmp_path)
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan["jobs"]) == 2
    assert "contact-pair" in plan["contact_scope"]
    for job in plan["jobs"]:
        args = job["args"]
        assert args[args.index("--online-ticks") + 1] == "0"
        assert args[args.index("--warmup-steps") + 1] == "1500"
        assert args[args.index("--methods") + 1 :] == [
            "joint",
            "arm_q_gripper_joint",
            "eef_gripper_joint",
        ]


def test_paired_action_plan_can_reserve_development_layouts(tmp_path):
    import json

    from toolkits.rlt.physrlt.campaign import prepare_paired_actions

    base = tmp_path / "base"
    for task, perturbation in ((0, "object"), (3, "lan")):
        name = f"task{task}-{perturbation}"
        collection = base / name / "collection"
        collection.mkdir(parents=True)
        (collection / "transitions.pt").touch()
        for method in ("joint", "shuffled_joint", "state_action_joint"):
            checkpoint = base / name / "probes" / f"{method}-seed31/readout.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
    output = tmp_path / "paired-development"
    prepare_paired_actions(output, base, tmp_path, split="development")
    plan = json.loads((output / "plan.json").read_text())
    assert plan["paired_split"] == "development"
    assert len(plan["jobs"]) == 2
    assert all("development" in job["id"] for job in plan["jobs"])
    assert all(
        job["args"][job["args"].index("--split") + 1] == "development"
        for job in plan["jobs"]
    )
    with pytest.raises(ValueError, match="train or development"):
        prepare_paired_actions(tmp_path / "invalid", base, tmp_path, split="test")


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
        "factorized_recon",
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
    assert readout.model.factorized == method.startswith("factorized_")
    assert readout.payload["geometry_weight"] == 1.0
    assert readout.payload["reconstruction_weight"] == 0.1


@pytest.mark.parametrize("limit", [7, 10])
@pytest.mark.parametrize("gain", [0.7, 1.0])
def test_control_records_proposal_and_execution_domains_and_real_successor(
    monkeypatch, limit, gain
):
    from toolkits.rlt.physrlt import control

    class SimulatorEdge:
        collection_reset_hashes = {}
        control_reset_states = {}
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
    monkeypatch.setattr(
        control,
        "physical_reset_state",
        lambda env: np.array([0.5, 1.0], dtype=np.float64),
    )
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
    assert len(replay) == int(limit == 10)
    if replay:
        assert replay[0]["action"][0, 6].item() == pytest.approx(0.2)
        assert replay[0]["accepted_commands"][0, 6].item() == 1
        assert replay[0]["accepted_commands"][0, 0].item() == pytest.approx(0.2 * gain)
        assert replay[0]["accepted_commands"][limit:].count_nonzero() == 0
        assert replay[0]["prop"][0] == 9
        assert replay[0]["next_prop"][0] == 9 + limit
    assert len(result["updates"]) == int(limit == 10)


def test_deferred_control_keeps_replay_but_skips_inline_update(monkeypatch):
    from toolkits.rlt.physrlt import control

    class Environment:
        collection_reset_hashes = {}
        control_reset_states = {}
        instruction = "deferred synthetic test"

        def reset(self, case):
            self.tick = 0
            return {"state": np.zeros(8, np.float32)}

        def step(self, command):
            self.tick += 1
            return {"state": np.full(8, self.tick, np.float32)}, 0.0, False

    class Reference:
        def predict(self, obs, instruction, seed):
            return {"ref": np.zeros((10, 7), np.float32)}

    class Readout:
        def encode(self, prediction, prop):
            return {
                "z": torch.zeros(128),
                "ref": torch.from_numpy(prediction["ref"]),
                "prop": torch.as_tensor(prop),
            }

    class Learner:
        def command(self, features, deterministic):
            return features["ref"].clone()

        def update(self, replay):
            raise AssertionError("Deferred collection must not update inline")

    monkeypatch.setattr(control, "physical_reset_hash", lambda env: "reset")
    monkeypatch.setattr(
        control, "physical_reset_state", lambda env: np.array([0.5], np.float64)
    )
    replay = []
    result = control.rollout(
        Environment(),
        Reference(),
        Readout(),
        Learner(),
        {"id": "case", "split": "train", "state": 0, "seed": 1},
        10,
        replay,
        update_online=False,
    )
    assert len(replay) == 1 and result["updates"] == []
    assert replay_trace_sha256(replay) == replay_trace_sha256(copy.deepcopy(replay))
    changed = copy.deepcopy(replay)
    changed[0]["accepted_commands"][0, 0] = 1
    assert replay_trace_sha256(changed) != replay_trace_sha256(replay)
    summary = validate_fixed_replay(replay)
    assert summary["transitions"] == 1
    assert summary["trace_sha256"] == replay_trace_sha256(replay)
    invalid = copy.deepcopy(replay)
    invalid[0]["ticks"] = 3
    with pytest.raises(ValueError, match="not TD eligible"):
        validate_fixed_replay(invalid)


def test_deferred_update_summary_keeps_reward_and_drift_diagnostics():
    metrics = [
        {"critic_loss": 4.0, "correction_rms": 0.01},
        {"critic_loss": 2.0, "correction_rms": 0.03},
    ]
    replay = [{"reward": 0.0}, {"reward": 5.0}]
    summary = deferred_update_summary(metrics, replay)
    assert summary["updates"] == summary["transitions"] == 2
    assert summary["positive_transitions"] == 1
    assert summary["reward_sum"] == 5
    assert summary["metrics"]["critic_loss"]["mean"] == 3
    assert summary["metrics"]["correction_rms"]["max"] == 0.03
    assert skip_deferred_updates(replay, True) is False
    assert skip_deferred_updates([{"reward": 0.0}], True) is True
    assert skip_deferred_updates([{"reward": 0.0}], False) is False


def test_control_reset_pairing_rejects_physics_drift(monkeypatch):
    from toolkits.rlt.physrlt import control

    class Environment:
        control_reset_states = {}

    environment = Environment()
    states = iter(
        [
            np.array([0.5, 1.0], dtype=np.float64),
            np.array([0.5, 1.0 + 5e-10], dtype=np.float64),
            np.array([0.5, 1.0 + 2e-8], dtype=np.float64),
        ]
    )
    monkeypatch.setattr(control, "physical_reset_state", lambda env: next(states))
    assert control.paired_reset_delta(environment, "case") == 0
    assert control.paired_reset_delta(environment, "case") == pytest.approx(5e-10)
    with pytest.raises(RuntimeError, match="Paired control reset differs"):
        control.paired_reset_delta(environment, "case")


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
    checkpoint = tmp_path / "probes/joint-seed3/readout.pt"
    readout = FrozenReadout(
        checkpoint,
        consequence=checkpoint,
        actor_conditioning="physical_outcomes_concat",
    )
    original = readout.encode(rows[0], rows[0]["prop"])["actor_z"]
    changed = copy.deepcopy(rows[0])
    changed["response"][:] = 1e9
    changed["geometry"][:] = 1e9
    torch.testing.assert_close(
        readout.encode(changed, changed["prop"])["actor_z"], original
    )
    assert original.shape == (256,)


def test_outcome_token_masks_unsupervised_response_decoder_rows(tmp_path):
    from toolkits.rlt.physrlt.control import FrozenReadout

    rows = synthetic_rows()
    path = tmp_path / "synthetic.pt"
    torch.save({"rows": rows}, path)
    run(
        Namespace(
            dataset=path,
            output=tmp_path / "probes",
            methods=["arm_q_gripper_joint"],
            seeds=[3],
            steps=2,
            threads=2,
        )
    )
    checkpoint = tmp_path / "probes/arm_q_gripper_joint-seed3/readout.pt"
    original = FrozenReadout(checkpoint, mode="outcomes").encode(
        rows[0], rows[0]["prop"]
    )["z"]
    payload = torch.load(checkpoint, weights_only=True)
    payload["model"]["response.2.weight"][7:10] = 1e6
    payload["model"]["response.2.bias"][7:10] = 1e6
    altered = tmp_path / "altered.pt"
    torch.save(payload, altered)
    actual = FrozenReadout(altered, mode="outcomes").encode(rows[0], rows[0]["prop"])[
        "z"
    ]
    torch.testing.assert_close(actual, original)


def test_delayed_online_rl_requires_state_augmentation():
    from toolkits.rlt.physrlt.control import rollout

    with pytest.raises(ValueError, match="state augmentation"):
        rollout(None, None, None, None, {}, 10, replay=[], delay_ticks=2)


def test_action_prior_and_grounded_readout_have_one_causal_token_budget(tmp_path):
    from toolkits.rlt.physrlt.control import FrozenReadout
    from toolkits.rlt.physrlt.goal_ranking import load_audited_readout

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
    assert load_audited_readout(physical).payload["method"] == "joint"
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
    with pytest.raises(ValueError, match="different dataset"):
        FrozenReadout(physical, expected_dataset=tmp_path / "other.pt")
    with pytest.raises(ValueError, match="requested arm"):
        FrozenReadout(physical, expected_method="geometry")
    dataset.write_bytes(dataset.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="contents have changed"):
        FrozenReadout(physical, expected_dataset=dataset, expected_method="joint")


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


def test_geometry_evidence_clusters_layouts_before_readout_seeds(tmp_path):
    import json

    from toolkits.rlt.physrlt.geometry_evidence import (
        load_geometry_arms,
        summarize_geometry,
    )

    def row(method, seed, coupled, direct):
        return {
            "method": method,
            "seed": seed,
            "object": "object_pos",
            "rank": 0,
            "layout_comparison": {
                "layouts": ["layout-a", "layout-b"],
                "mse_mm2_by_layout": {
                    "coupled": coupled,
                    "direct": direct,
                    "shuffled_coupling": [9.0, 16.0],
                    "state_action_coupling": [4.0, 9.0],
                },
            },
        }

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "state": "complete",
                "split": "confirmation",
                "results": [
                    row("joint", 1, [1.0, 4.0], [4.0, 9.0]),
                    row("joint", 2, [1.0, 4.0], [4.0, 9.0]),
                    row("state_action_joint", 1, [4.0, 9.0], [4.0, 9.0]),
                    row("state_action_joint", 2, [4.0, 9.0], [4.0, 9.0]),
                ],
            }
        )
    )
    result = summarize_geometry(
        load_geometry_arms("task", path), draws=1000, seed=7
    )
    coupled = next(
        arm
        for arm in result["arms"]
        if arm["readout"] == "joint" and arm["metric"] == "coupled"
    )
    assert coupled["readout_seed_count"] == 2
    assert coupled["layout_count"] == 2
    assert coupled["equal_layout_rmse_mm"] == pytest.approx(2.5**0.5)
    direct = next(
        contrast
        for contrast in result["contrasts"]
        if contrast["contrast"] == "response_transport_vs_direct"
        and contrast["target_readout"] == "joint"
    )
    assert direct["rmse_improvement_mm"] == pytest.approx(6.5**0.5 - 2.5**0.5)
    assert direct["interval_excludes_zero"]
    assert not direct["confirmation_screen_eligible"]
    assert direct["evidence_tier"] == "pilot"
    semantic = next(
        contrast
        for contrast in result["contrasts"]
        if contrast["contrast"] == "action_readout_vs_state_action_readout"
    )
    assert semantic["interval_excludes_zero"]


def test_geometry_evidence_rejects_seed_layout_mismatch(tmp_path):
    import json

    from toolkits.rlt.physrlt.geometry_evidence import load_geometry_arms

    def row(seed, layouts):
        return {
            "method": "joint",
            "seed": seed,
            "object": "object_pos",
            "rank": 0,
            "layout_comparison": {
                "layouts": layouts,
                "mse_mm2_by_layout": {"coupled": [1.0, 2.0]},
            },
        }

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "state": "complete",
                "split": "development",
                "results": [row(1, ["a", "b"]), row(2, ["a", "c"])],
            }
        )
    )
    with pytest.raises(ValueError, match="Layout identities differ"):
        load_geometry_arms("task", path)
