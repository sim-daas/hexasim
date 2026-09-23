# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP functions for the hexapod environment."""

from isaaclab.utils.module import lazy_export

lazy_export()

# Explicit re-exports so that `from . import mdp as hex_mdp` and
# `hex_mdp.<func>` work without relying on lazy_export auto-discovery.
from .commands import HexapodVelocityCommandCfg, pin_command  # noqa: E402, F401
from .events import foot_materials  # noqa: E402, F401
from .observations import s0a_policy_obs, s0a_privileged_obs, s0b_dr_obs, s0b_policy_obs  # noqa: E402, F401
from .rewards import (  # noqa: E402, F401
    LinearAnnealRewardWeight,
    action_beyond_clip_penalty,
    ang_vel_xy_penalty,
    base_height_penalty,
    base_orientation_penalty,
    coxa_leg_geometry,
    coxa_stroke_weight,
    cpg_clock,
    cpg_coxa_contact_reward,
    cpg_force_target_penalty,
    cpg_stance_rigidity_penalty,
    cpg_z_target_kernel_reward,
    cpg_z_target_l1_penalty,
    foot_clearance_reward,
    foot_forces_nt,
    forward_velocity,
    joint_motion_reward,
    joint_pos_target_l2,
    joint_torque_threshold_penalty,
    lateral_velocity_penalty,
    non_foot_body_ground_penalty,
    root_height_above_terrain_below,
    stance_rigidity_contact_penalty,
    stand_still_penalty,
    standing_contact_reward,
    standing_joint_speed_penalty,
    standing_load_share_penalty,
    standing_pose_penalty,
    swing_clearance,
    track_ang_vel_z,
    track_lin_vel_xy,
)
from .terrain import S0B_TERRAIN_CFG, HeightMap, pin_terrain_level, terrain_levels_cmd  # noqa: E402, F401
