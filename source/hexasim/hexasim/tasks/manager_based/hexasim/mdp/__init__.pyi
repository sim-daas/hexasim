# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "joint_pos_target_l2",
    "forward_distance_x",
    "forward_velocity",
    "lateral_velocity_penalty",
    "ang_vel_xy_penalty",
    "joint_torque_threshold_penalty",
    "base_height_penalty",
    "stand_still_penalty",
    "foot_gait_reward",
    "joint_motion_reward",
    "base_orientation_penalty",
    "non_foot_body_ground_penalty",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .rewards import (  # noqa: F401
    ang_vel_xy_penalty,
    base_height_penalty,
    base_orientation_penalty,
    foot_gait_reward,
    forward_distance_x,
    forward_velocity,
    joint_motion_reward,
    joint_pos_target_l2,
    joint_torque_threshold_penalty,
    lateral_velocity_penalty,
    non_foot_body_ground_penalty,
    stand_still_penalty,
)
