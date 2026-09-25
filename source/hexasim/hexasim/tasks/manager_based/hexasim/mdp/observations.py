# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S0a / S0b observation groups (paradigm spec §4.2)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import foot_forces_nt

if TYPE_CHECKING:
    from ..hexapod_env import HexapodS0aEnv, HexapodS0bEnv


def s0a_policy_obs(env: HexapodS0aEnv) -> torch.Tensor:
    """Actor input, 114-D: o_p (111) ⊕ v_in (3).

    o_p = [q − q_default (18), q̇ (18), q_cmd − q (18), last action (18), IMU accel (3), IMU gyro (3),
           gravity (3), CPG clock (12), read-validity mask (18)]; v_in = (1−α)·v_true + α·V_0.
    """
    env.refresh_proprio()
    return torch.cat([env.o_p, env.v_in], dim=-1)


def s0a_privileged_obs(env: HexapodS0aEnv) -> torch.Tensor:
    """Critic-only extras, 22-D (S0a subset of contract P, spec §3.5).

    True base velocity (3), base height above the terrain (1), per-foot force along / across the terrain normal
    (6 × 2), contact state (6). On flat ground (S0a) the normal is world z and the terrain height 0.
    """
    robot = env.scene["robot"]
    sensor = SceneEntityCfg("contact_forces")
    fn, ft = foot_forces_nt(env, sensor)
    contact = (torch.norm(env.scene["contact_forces"].data.net_forces_w.torch, dim=-1) > 1.0).float()
    root = robot.data.root_pos_w.torch
    height = (root[:, 2] - env.terrain_z(root[:, :2])).unsqueeze(1)
    force_nt = torch.stack((fn, ft), dim=-1).flatten(1)
    return torch.cat([robot.data.root_lin_vel_b.torch, height, force_nt, contact], dim=-1)


def s0b_policy_obs(env: HexapodS0bEnv) -> torch.Tensor:
    """Actor input, 117-D: o_p (111) ⊕ v_in (3, V_0) ⊕ twist_cmd (3)."""
    env.refresh_proprio()
    return torch.cat([env.o_p, env.v_in, env.command_manager.get_command("base_velocity")], dim=-1)


def s0b_dr_obs(env: HexapodS0bEnv) -> torch.Tensor:
    """Critic-only dynamics-randomisation parameters (part of contract P's physics block), 12-D.

    Foot material (μ_s, μ_d, restitution, log10(1 + k/1 kN·m⁻¹), log10(1 + c/1 N·s·m⁻¹)), robot mass (kg), base CoM
    offset x/y (m), servo stall-torque scale, compliance slope/32, backlash (deg), Coulomb friction (N·m).
    """
    robot = env.scene["robot"]
    servo = robot.actuators["ax12a"]
    base = robot.find_bodies(".*base_link.*")[0][0]
    return torch.cat(
        [
            env.foot_material,
            robot.data.body_mass.torch.sum(dim=1, keepdim=True).to(env.device),
            robot.data.body_com_pos_b.torch[:, base, :2].to(env.device),
            servo.stall / servo.cfg.stall_torque,
            servo.slope / 32.0,
            servo.backlash * (180.0 / math.pi),
            servo.coulomb,
        ],
        dim=-1,
    )


def s1_policy_obs(env) -> torch.Tensor:
    """S1 actor input, 1,125-D union layout (``mdp.layout.ACTOR_BLOCKS``): o_p | V_0 × per-episode keep | twist_cmd |
    z_p | ê × gate | zero-held arm slots (M*, z_s/z_F, tokens, v_ℓ⊙m_ℓ, m_ℓ, σ̂_ℓ, rem_k)."""
    env.refresh_proprio()
    return env.s1_actor_obs


def s1_mirror_latents(env) -> torch.Tensor:
    """z_p and ê (gated) recomputed on the mirrored history — the exact mirror of the learned blocks (``symmetry``)."""
    env.refresh_proprio()
    return env.s1_mirror_latents


def contract_p(env) -> torch.Tensor:
    """Critic contract P, 683-D (paradigm spec §3.5; ``mdp.layout.CONTRACT_P_BLOCKS``), computed at step t and
    independent of the action."""
    env.refresh_proprio()
    return env.s1_contract_p
