# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise experience records, grounding and perturbations without Isaac."""

import numpy as np
import pytest
import torch

from toolkits.rlt.robodojo_experience import (
    EXPERIENCE_DIM,
    ExperienceHistory,
    experience_features,
    history_view,
    response_valid,
    segment_replay,
)
from toolkits.rlt.robodojo_grounding import (
    TOKEN_DIM,
    GroundedTokenizer,
    action_permutation_margin,
    applicability_weights,
    build_samples,
    ground_replay_rows,
    grounding_loss,
    load_grounding,
    save_grounding,
)
from toolkits.rlt.robodojo_perturb import (
    DelayLine,
    GainShaper,
    make_perturber,
    parse_perturb_schedule,
    phase_for_episode,
)


def make_row(
    *,
    prop_value=0.0,
    next_value=0.1,
    reward=0.0,
    ticks=10,
    terminated=False,
    truncated=False,
    with_successor=True,
):
    row = {
        "prop": np.full(14, prop_value, np.float32),
        "ref": torch.zeros(10, 14),
        "z": torch.zeros(2048),
        "action": np.zeros((10, 14), np.float32),
        "executed_action": np.zeros((ticks, 14), np.float32),
        "reward": reward,
        "ticks": ticks,
        "terminated": terminated,
        "truncated": truncated,
    }
    if with_successor:
        row["next_prop"] = np.full(14, next_value, np.float32)
        row["next_ref"] = torch.zeros(10, 14)
        row["next_z"] = torch.zeros(2048)
    return row


def test_experience_features_separate_measured_response_from_grippers():
    row = make_row(prop_value=0.0, next_value=0.5)
    features = experience_features(row)
    assert features.shape == (EXPERIENCE_DIM,)
    response = features[14 + 140 : 14 + 140 + 14]
    assert response[0] == pytest.approx(0.5)
    assert response[6] == 0.0 and response[13] == 0.0


def test_experience_features_zero_response_without_a_valid_successor():
    row = make_row(terminated=True, reward=1.0, ticks=4)
    row["executed_action"] = np.zeros((4, 14), np.float32)
    assert not response_valid(row)
    features = experience_features(row)
    assert not features[14 + 140 : 14 + 140 + 14].any()
    assert features[-1] == 1.0
    with pytest.raises(ValueError, match="successor"):
        experience_features(make_row(with_successor=False))


def test_history_window_pads_and_respects_capacity():
    history = ExperienceHistory(4)
    features, mask = history.features()
    assert not mask.any()
    for value in range(6):
        history.append(make_row(prop_value=float(value)))
    features, mask = history.features()
    assert mask.sum() == 4
    assert features[0, 0] == pytest.approx(2.0)
    history.reset()
    assert len(history) == 0


def test_shuffled_history_view_breaks_command_response_pairing():
    history = ExperienceHistory(4)
    for value in range(4):
        history.append(make_row(prop_value=0.0, next_value=float(value + 1)))
    features, mask = history.features()
    shuffled, _ = history_view(features, mask, "shuffled", np.random.default_rng(3))
    start = 14 + 140
    assert sorted(shuffled[:, start].tolist()) == sorted(features[:, start].tolist())
    assert not np.array_equal(shuffled[:, start:], features[:, start:])
    assert np.array_equal(shuffled[:, :start], features[:, :start])
    zero, zero_mask = history_view(features, mask, "zero")
    assert not zero.any() and not zero_mask.any()
    with pytest.raises(ValueError, match="generator"):
        history_view(features, mask, "shuffled")


def test_segment_replay_recovers_episode_boundaries():
    first = [
        make_row(prop_value=0.0, next_value=1.0),
        make_row(prop_value=1.0, next_value=2.0, terminated=True, reward=1.0),
    ]
    second = [make_row(prop_value=9.0, next_value=10.0)]
    segments = segment_replay(first + second)
    assert [len(segment) for segment in segments] == [2, 1]


def test_grounding_predictions_depend_on_the_candidate_action():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rows = []
    prop = np.zeros(14, np.float32)
    # The measured response equals the commanded first-tick increment, so a
    # predictor can only fit it by reading the action.
    for _ in range(60):
        command = rng.normal(0, 0.3, (10, 14)).astype(np.float32)
        next_prop = prop + command[0]
        rows.append(
            make_row(prop_value=0.0, next_value=0.0)
            | {
                "prop": prop.copy(),
                "action": command,
                "executed_action": command,
                "next_prop": next_prop.astype(np.float32),
            }
        )
    samples = build_samples(rows, history_length=4)
    tokenizer = GroundedTokenizer(4)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    from toolkits.rlt.robodojo_grounding import _collate

    for _ in range(200):
        batch = _collate(samples, "cpu")
        context = tokenizer.encoder(batch["history"], batch["mask"])
        prediction = tokenizer.predictor(batch["prop"], context, batch["action"])
        loss, _ = grounding_loss(prediction, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    audit = action_permutation_margin(
        tokenizer.eval(), samples, "cpu", __import__("random").Random(0)
    )
    assert audit["margin"] > 0


def test_grounding_artifact_round_trip(tmp_path):
    tokenizer = GroundedTokenizer(8)
    digest = save_grounding(tmp_path / "grounding.pt", tokenizer, {"gate_passed": True})
    loaded, loaded_digest = load_grounding(tmp_path / "grounding.pt")
    assert loaded_digest == digest
    assert loaded.history_length == 8
    token = loaded.token(
        torch.zeros(14),
        np.zeros((8, EXPERIENCE_DIM), np.float32),
        np.zeros(8, np.float32),
        torch.zeros(10, 14),
    )
    assert token.shape == (TOKEN_DIM,)


def test_ground_replay_rows_widen_tokens_and_keep_successor_contract():
    rows = [
        make_row(prop_value=0.0, next_value=1.0),
        make_row(prop_value=1.0, next_value=2.0, terminated=True, reward=1.0),
    ]
    grounded = ground_replay_rows(rows, GroundedTokenizer(4).eval())
    assert all(row["z"].shape == (2048 + TOKEN_DIM,) for row in grounded)
    torch.testing.assert_close(grounded[0]["next_z"], grounded[1]["z"])
    torch.testing.assert_close(grounded[1]["next_z"], grounded[1]["z"])


def test_applicability_weights_decay_with_prediction_error():
    tokenizer = GroundedTokenizer(4).eval()
    experiences = np.stack(
        [
            experience_features(make_row(prop_value=0.0, next_value=0.0)),
            experience_features(make_row(prop_value=0.0, next_value=5.0)),
        ]
    )
    recent = (
        np.zeros((4, EXPERIENCE_DIM), np.float32),
        np.zeros(4, np.float32),
    )
    weights = applicability_weights(tokenizer, experiences, recent, scale=0.5)
    assert weights.shape == (2,)
    assert (weights > 0).all() and (weights <= 1).all()
    assert applicability_weights(
        tokenizer, np.zeros((0, EXPERIENCE_DIM), np.float32), recent
    ).shape == (0,)


def test_perturb_schedule_parses_aba_phases():
    schedule = parse_perturb_schedule("none:2,delay=2:2,none")
    assert [phase["kind"] for phase in schedule] == ["none", "delay", "none"]
    assert phase_for_episode(schedule, 0)["kind"] == "none"
    assert phase_for_episode(schedule, 2)["kind"] == "delay"
    assert phase_for_episode(schedule, 4)["kind"] == "none"
    assert phase_for_episode(schedule, 99)["kind"] == "none"
    assert parse_perturb_schedule("gain=1.5")[0]["value"] == 1.5
    for bad in ("", "delay=0", "delay=21:2", "gain=3", "none,delay=2"):
        with pytest.raises(ValueError):
            parse_perturb_schedule(bad)


def test_delay_line_holds_then_replays_the_command_stream():
    hold = np.zeros(14, np.float32)
    line = DelayLine(2, hold)
    first = np.full(14, 1.0, np.float32)
    second = np.full(14, 2.0, np.float32)
    np.testing.assert_array_equal(line.step(first), hold)
    np.testing.assert_array_equal(line.step(second), hold)
    np.testing.assert_array_equal(line.step(np.full(14, 3.0, np.float32)), first)
    assert line.describe() == {"kind": "delay", "value": 2}


def test_gain_shaper_scales_joint_increments_and_passes_grippers():
    hold = np.zeros(14, np.float32)
    shaper = GainShaper(2.0, hold)
    command = np.full(14, 0.1, np.float32)
    applied = shaper.step(command)
    assert applied[0] == pytest.approx(0.2)
    assert applied[6] == pytest.approx(0.1)
    assert applied[13] == pytest.approx(0.1)
    # A second identical command advances by a zero increment.
    again = shaper.step(command)
    assert again[0] == pytest.approx(0.2)
    identity = GainShaper(1.0, hold)
    np.testing.assert_allclose(identity.step(command), command, atol=1e-7)


def test_make_perturber_matches_phase_kind():
    hold = np.zeros(14, np.float32)
    assert make_perturber({"kind": "none", "value": None}, hold) is None
    assert isinstance(make_perturber({"kind": "delay", "value": 1}, hold), DelayLine)
    assert isinstance(make_perturber({"kind": "gain", "value": 0.5}, hold), GainShaper)
