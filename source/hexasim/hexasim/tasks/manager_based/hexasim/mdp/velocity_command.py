# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""twist_cmd term (loaded lazily through ``HexapodVelocityCommandCfg.class_type``: the Isaac Lab command class pulls
in USD, which must not load before the simulation app starts)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


class HexapodVelocityCommand(UniformVelocityCommand):
    """Segments: ``rel_turn_envs`` turn on the spot (v_x = v_y = 0, |yaw| ~ U[turn_yaw_range], random sign),
    ``rel_side_envs`` keep the sampled v_y and yaw, the rest walk straight (v_y = yaw = 0). Standing
    (``rel_standing_envs``, all-zero) is drawn independently by the base class and overrides the rest."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device)
        r = torch.rand(len(ids), device=self.device)
        turn = ids[r < self.cfg.rel_turn_envs]
        straight = ids[r >= self.cfg.rel_turn_envs + self.cfg.rel_side_envs]
        self.vel_command_b[straight, 1:] = 0.0
        n = len(turn)
        yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.turn_yaw_range)
        self.vel_command_b[turn, :2] = 0.0
        self.vel_command_b[turn, 2] = yaw * (torch.randint(0, 2, (n,), device=self.device) * 2 - 1)
