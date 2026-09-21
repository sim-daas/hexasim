# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""twist_cmd distribution (paradigm spec §1): v_x ~ U[0.05, 0.20] m/s; a fraction of segments also carries small
v_y / yaw-rate commands, the rest have v_y = yaw = 0; ``rel_standing_envs`` of segments are all-zero (standing)."""

from __future__ import annotations

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass


@configclass
class HexapodVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: str = "{DIR}.velocity_command:HexapodVelocityCommand"
    rel_side_envs: float = 0.3  # fraction of segments that keep their sampled v_y and yaw rate


def pin_command(cfg, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
    """Evaluation: one fixed twist_cmd for every env and the whole episode; all-zero = standing mode."""
    c = cfg.commands.base_velocity
    c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
    c.rel_standing_envs = 1.0 if vx == vy == wz == 0.0 else 0.0
    c.rel_side_envs = 1.0
    c.resampling_time_range = (1e9, 1e9)
