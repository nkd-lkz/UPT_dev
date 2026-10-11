# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""RoboDojo native observation/action and environment lifecycle contracts."""

import numpy as np
import pytest

from toolkits.rlt.robodojo_boundary import (
    ACTION_SEMANTICS,
    CAMERAS,
    FIELDS,
    RoboDojoBoundary,
    decode_actions,
    encode_observation,
)


def observation(index):
    return {
        "env_idx": index,
        "instruction": f"task {index}",
        "state": {k: np.full(n, index, dtype=np.float32) for k, n in FIELDS},
        "vision": {
            name: {"color": np.full((4, 5, 3), [1, 2, 3], dtype=np.uint8)}
            for name, _ in CAMERAS
        },
        "privileged_success": True,
    }


def test_rgb_channels_state_order_and_privileged_info_filtering():
    obs = observation(1)
    result = encode_observation(obs)
    assert set(result) == {"state", "images", "prompt"}
    assert result["state"].shape == (14,)
    np.testing.assert_array_equal(result["images"]["cam_high"][:, 0, 0], [1, 2, 3])
    obs["vision"]["cam_head"]["color"][:] = 9
    assert result["images"]["cam_high"][0, 0, 0] == 1
    obs["state"]["right_arm_joint_state"] = np.zeros(7)
    with pytest.raises(ValueError):
        encode_observation(obs)


def test_native_commands_preserve_joint_radians_and_clamp_only_grippers():
    kwargs = {"semantics": ACTION_SEMANTICS, "horizon": 2}
    commands = np.zeros((3, 14), dtype=np.float32)
    commands[:, 0] = 2.5  # Joint radians are not normalized [-1, 1] deltas.
    commands[:, 6] = -0.1
    commands[:, 7] = -2.5
    commands[:, 13] = 1.1
    original = commands.copy()
    result = decode_actions(commands, **kwargs)
    assert len(result) == 2 and result[0]["left_arm_joint_state"][0] == 2.5
    assert result[0]["right_arm_joint_state"][0] == -2.5
    assert result[0]["left_ee_joint_state"][0] == 0.0
    assert result[0]["right_ee_joint_state"][0] == 1.0
    np.testing.assert_array_equal(commands, original)
    with pytest.raises(ValueError, match="LIBERO"):
        decode_actions(np.zeros((10, 7)), **kwargs)
    with pytest.raises(ValueError, match="deltas"):
        decode_actions(np.zeros((3, 14)), horizon=2, semantics="delta_ee")


@pytest.mark.parametrize("column", [0, 6, 7, 13])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_actions_rejected_even_in_unexecuted_suffix(column, value):
    commands = np.zeros((3, 14), dtype=np.float32)
    commands[-1, column] = value
    with pytest.raises(ValueError, match="finite"):
        decode_actions(commands, semantics=ACTION_SEMANTICS, horizon=2)


def test_batch_reordering_shrinking_reset_and_checkpoint_guard():
    contract = {
        "embodiment": "arx_x5",
        "action_dim": 14,
        "action_semantics": ACTION_SEMANTICS,
        "normalization_id": "test-only",
    }
    seen = []

    def predict(obs):
        seen.append(obs["prompt"])
        return np.zeros((2, 14))

    boundary = RoboDojoBoundary(predict, model_contract=contract, horizon=2)
    with pytest.raises(ValueError, match="LIBERO"):
        RoboDojoBoundary(predict, model_contract=contract | {"action_dim": 7})
    boundary.update_obs_batch([observation(8), observation(2)])
    boundary.get_action_batch([2, 8])
    assert seen == ["task 2", "task 8"]
    with pytest.raises(ValueError, match="Update"):
        boundary.get_action_batch([2])
    boundary.update_obs_batch([observation(8)])
    boundary.reset()
    boundary.reset()
    with pytest.raises(ValueError, match="Update"):
        boundary.get_action_batch([8])
