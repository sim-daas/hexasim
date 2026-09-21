# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP functions for the hexapod environment."""

from isaaclab.utils.module import lazy_export

lazy_export()

# Explicit re-exports so that `from . import mdp as hex_mdp` and
# `hex_mdp.<func>` work without relying on lazy_export auto-discovery.
from .observations import s0a_policy_obs, s0a_privileged_obs  # noqa: E402, F401
from .rewards import (  # noqa: E402, F401
    LinearAnnealRewardWeight,
    ang_vel_xy_penalty,
    base_height_penalty,
    base_orientation_penalty,
    cpg_clock,
    cpg_coxa_contact_reward,
    cpg_force_target_penalty,
    cpg_stance_rigidity_penalty,
    cpg_z_target_kernel_reward,
    cpg_z_target_l1_penalty,
    foot_clearance_reward,
    forward_velocity,
    joint_motion_reward,
    joint_pos_target_l2,
    joint_torque_threshold_penalty,
    lateral_velocity_penalty,
    non_foot_body_ground_penalty,
    stand_still_penalty,
)
