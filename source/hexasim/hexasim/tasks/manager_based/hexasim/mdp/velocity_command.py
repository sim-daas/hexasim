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
    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device)
        forward_only = ids[torch.rand(len(ids), device=self.device) >= self.cfg.rel_side_envs]
        self.vel_command_b[forward_only, 1:] = 0.0
