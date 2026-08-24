# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Sequence
import torch

from isaaclab.envs import ManagerBasedRLEnv

class HexapodEnv(ManagerBasedRLEnv):
    """Custom environment for the Hexapod that maintains a Central Pattern Generator (CPG) clock."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize CPG phase tensor for all environments: (num_envs, 6)
        # Duty factor beta = 5/6. Wave gait phase offsets:
        # Leg indices based on assets.robot._ALL_JOINTS:
        # 0: right_front
        # 1: right_middle
        # 2: right_back
        # 3: left_front
        # 4: left_middle
        # 5: left_back
        
        self.cpg_phase = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        
        self._cpg_base_offsets = torch.tensor([
            4.0 * math.pi / 6.0,            # 0: RF
            2.0 * math.pi / 6.0,            # 1: RM
            0.0,                            # 2: RB
            math.pi + 4.0 * math.pi / 6.0,  # 3: LF
            math.pi + 2.0 * math.pi / 6.0,  # 4: LM
            math.pi,                        # 5: LB
        ], device=self.device, dtype=torch.float32)

        # Set initial phases
        self.cpg_phase[:] = self._cpg_base_offsets.clone()

    def _step_impl(self, action: torch.Tensor):
        # Update phases continuously
        # Frequency f = 1.0 Hz (1 full cycle per second)
        f = 1.0
        delta_phase = 2.0 * math.pi * f * self.step_dt
        self.cpg_phase = (self.cpg_phase + delta_phase) % (2.0 * math.pi)
        
        # Call the parent step implementation
        return super()._step_impl(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        
        if env_ids is None or len(env_ids) == 0:
            return
            
        # Reset the phase for fallen environments to their initial offsets
        self.cpg_phase[env_ids] = self._cpg_base_offsets.clone()
