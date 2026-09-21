# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S0a observation groups (paradigm spec §4.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..hexapod_env import HexapodS0aEnv


def s0a_policy_obs(env: HexapodS0aEnv) -> torch.Tensor:
    """Actor input, 114-D: o_p (111) ⊕ v_in (3).

    o_p = [q − q_default (18), q̇ (18), q_cmd − q (18), last action (18), IMU accel (3), IMU gyro (3),
           gravity (3), CPG clock (12), read-validity mask (18)]; v_in = (1−α)·v_true + α·V_0.
    """
    env.refresh_proprio()
    return torch.cat([env.o_p, env.v_in], dim=-1)


def s0a_privileged_obs(env: HexapodS0aEnv) -> torch.Tensor:
    """Critic-only extras, 22-D (S0a subset of contract P, spec §3.5).

    True base velocity (3), base height (1), per-foot normal and tangential force (6 × 2), contact state (6).
    ponytail: normal = world z and height = root z hold on flat ground only; use terrain normals/heights at S0b.
    """
    robot = env.scene["robot"]
    f = env.scene["contact_forces"].data.net_forces_w.torch
    force_nt = torch.stack((f[..., 2], torch.norm(f[..., :2], dim=-1)), dim=-1).flatten(1)
    contact = (torch.norm(f, dim=-1) > 1.0).float()
    return torch.cat([robot.data.root_lin_vel_b.torch, robot.data.root_pos_w.torch[:, 2:3], force_nt, contact], dim=-1)
