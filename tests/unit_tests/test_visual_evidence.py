# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from toolkits.rlt.visual_evidence import ImageRegion, diagnose_regions


def test_visual_evidence_separates_output_paths_and_preserves_inputs():
    images = {"main": torch.ones(2, 1, 4, 4), "wrist": torch.full((2, 1, 4, 4), 2.0)}
    regions = [ImageRegion("main", 0, 0, 2, 2), ImageRegion("wrist", 0, 0, 2, 2)]

    def predict(data):
        main = data["main"][..., :2, :2].mean((1, 2, 3))[:, None]
        wrist = data["wrist"][..., :2, :2].mean((1, 2, 3))[:, None]
        return {"vla_action": main, "rl_token": wrist, "actor_action": wrist * 3}

    result = diagnose_regions(
        images,
        regions,
        predict,
        replacement="zero",
        targets={"vla_action": torch.ones(2, 1)},
    )
    first, second = [row["outputs"] for row in result["regions"]]
    assert first["vla_action"]["rms_change"] == [1, 1]
    assert first["vla_action"]["target_mse_increase"] == [1, 1]
    assert first["actor_action"]["rms_change"] == [0, 0]
    assert second["actor_action"]["rms_change"] == [6, 6]
    assert result["forward_calls"] == 4
    assert images["main"].eq(1).all() and images["wrist"].eq(2).all()


def test_visual_evidence_reports_improvement_with_negative_loss_change():
    result = diagnose_regions(
        {"main": torch.ones(1, 1, 2, 2)},
        [ImageRegion("main", 0, 0, 2, 2)],
        lambda data: {"action": data["main"].flatten(1)},
        replacement="zero",
        targets={"action": torch.zeros(1, 4)},
    )
    assert result["regions"][0]["outputs"]["action"]["target_mse_increase"] == [-1]


def test_visual_evidence_rejects_stochastic_predictor():
    with pytest.raises(ValueError, match="deterministic"):
        diagnose_regions(
            {"main": torch.ones(1, 1, 2, 2)},
            [ImageRegion("main", 0, 0, 1, 1)],
            lambda data: {"action": torch.rand(1, 7)},
        )


@pytest.mark.parametrize(
    "region",
    [
        ImageRegion("missing", 0, 0, 1, 1),
        ImageRegion("main", 0, 0, 0, 1),
        ImageRegion("main", 0, 0, 3, 1),
        ImageRegion("main", -1, 0, 1, 1),
    ],
)
def test_visual_evidence_validates_regions_before_inference(region):
    def unexpected(data):
        raise AssertionError("Invalid request must not call the model")

    with pytest.raises(ValueError):
        diagnose_regions({"main": torch.ones(1, 1, 2, 2)}, [region], unexpected)


def test_visual_evidence_mean_fill_is_noop_on_constant_image():
    result = diagnose_regions(
        {"main": torch.ones(1, 1, 2, 2)},
        [ImageRegion("main", 0, 0, 1, 1)],
        lambda data: {"action": data["main"].flatten(1)},
    )
    assert result["regions"][0]["outputs"]["action"]["rms_change"] == [0]
