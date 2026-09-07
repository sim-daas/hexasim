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
        # Duty factor beta = 5/6. ALTERNATING TRIPOD gait phase offsets.
        # Convention: swing occupies phase [0, (1-beta)*2pi) = [0, pi/3),
        # so a leg's offset is the global time at which it starts swinging.
        #
        # Leg indices based on assets.robot._ALL_JOINTS (and ContactSensor
        # body order, verified empirically):
        # 0: right_front   -> Tripod A
        # 1: right_middle  -> Tripod B
        # 2: right_back    -> Tripod A
        # 3: left_front    -> Tripod B
        # 4: left_middle   -> Tripod A
        # 5: left_back     -> Tripod B
        #
        # Tripod A = {RF, LM, RB} swings together (offset 0);
        # Tripod B = {LF, RM, LB} swings half a cycle later (offset pi).

        self.cpg_phase = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        
        self._cpg_base_offsets = torch.tensor([
            0.0,      # 0: RF  — Tripod A
            math.pi,  # 1: RM  — Tripod B
            0.0,      # 2: RB  — Tripod A
            math.pi,  # 3: LF  — Tripod B
            0.0,      # 4: LM  — Tripod A
            math.pi,  # 5: LB  — Tripod B
        ], device=self.device, dtype=torch.float32)

        # Set initial phases
        self.cpg_phase[:] = self._cpg_base_offsets.clone()
        # Per-leg ground-contact Z baseline for cpg_z_target_reward.
        # EMA-updated each step wherever the tibia registers >1 N contact;
        # tracks torso sag so swing targets stay reachable. Init = measured
        # rest-pose contact offset.
        self._cpg_baseline_init = 0.0892
        self.cpg_foot_baseline = torch.full(
            (self.num_envs, 6), self._cpg_baseline_init,
            device=self.device, dtype=torch.float32,
        )
        # Resolve tibia body indices once (order matches cpg_phase: RF,RM,RB,LF,LM,LB)
        art = self.scene["robot"]
        self._tibia_body_ids = [i for i, n in enumerate(art.data.body_names) if "tibia" in n]

    def step(self, action: torch.Tensor):
        # Dynamic CPG frequency: f = v_x / stride_length, clamped [0.2, 2.5]
        # stride_length = 0.20 m matches the training-time empirical phase rate
        # (observed Δφ = 0.265 rad/step ≡ 2.11 Hz ≈ 0.447/0.20; stride=0.15
        # would produce 0.325 rad/step / 2.8 Hz — 33% too fast → overstepping).
        with torch.no_grad():
            vx = self.scene["robot"].data.root_lin_vel_b[:, 0]
            vx_positive = torch.clamp(vx, min=0.0)
            f = torch.clamp(vx_positive / 0.20, min=0.2, max=2.5)
        
        delta_phase = 2.0 * math.pi * f.unsqueeze(1) * self.step_dt
        self.cpg_phase = (self.cpg_phase + delta_phase) % (2.0 * math.pi)
        
        # Call the parent step implementation
        ret = super().step(action)

        # Update the adaptive foot baseline AFTER physics (sensors are fresh).
        # We only update legs that are solidly touching the ground.
        if hasattr(self, "cpg_foot_baseline"):
            sensor = self.scene["contact_forces"]
            art = self.scene["robot"]
            
            # (num_envs, 6, 3) -> L2 norm (num_envs, 6)
            forces = torch.norm(sensor.data.net_forces_w, dim=-1)
            is_contact = forces > 1.0
            
            # Current absolute Z positions of the tibias in the world frame.
            tibia_z_w = art.data.body_pos_w[:, self._tibia_body_ids, 2]
            
            # EMA parameters
            alpha = 0.2
            self.cpg_foot_baseline = torch.where(
                is_contact,
                (1.0 - alpha) * self.cpg_foot_baseline + alpha * tibia_z_w,
                self.cpg_foot_baseline,
            )
            
        return ret

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        
        if env_ids is None or len(env_ids) == 0:
            return
            
        # Reset the phase for fallen environments to their initial offsets
        self.cpg_phase[env_ids] = self._cpg_base_offsets.clone()
        self.cpg_foot_baseline[env_ids] = self._cpg_baseline_init
