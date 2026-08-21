# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the hexapod locomotion environment.

Functions
---------
joint_pos_target_l2         : penalise joint deviation from a target value.
forward_velocity            : reward forward body-frame velocity.
lateral_velocity_penalty    : penalise sideways drift.
stand_still_penalty         : penalise slow forward motion (cancels alive bonus).
ang_vel_xy_penalty          : penalise roll/pitch rates.
joint_torque_threshold_penalty : penalise torques above threshold.
base_height_penalty         : penalise root height below minimum.
foot_clearance_reward       : reward feet for being off the ground (anti-slide).
joint_motion_reward         : reward joint velocity while penalising acceleration.
base_orientation_penalty    : two-zone roll/pitch angle penalty (projected gravity).
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Keep existing reward from template
# ---------------------------------------------------------------------------

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


# ---------------------------------------------------------------------------
# Locomotion rewards
# ---------------------------------------------------------------------------

def forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward (body-frame x-axis) linear velocity.

    Returns the raw clipped forward velocity so the reward scales linearly
    with speed — the faster the robot moves forward the more it earns.
    Negative forward velocity (moving backward) gives zero reward.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # root_lin_vel_b: body-frame linear velocity (N, 3)
    forward_vel = asset.data.root_lin_vel_b[:, 0]   # x-component (forward)
    return torch.clamp(forward_vel, min=0.0)


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral (body-frame y-axis) drift.

    Returns |v_y| so the weight should be negative in the reward config.
    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lateral_vel = asset.data.root_lin_vel_b[:, 1]   # y-component (sideways)
    return torch.abs(lateral_vel)


def stand_still_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize forward velocity if it drops below a threshold.

    Returns (threshold - v_x) if v_x < threshold, else 0.
    This discourages the robot from standing still and farming the alive bonus.
    Weight should be negative.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    v_x = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(threshold - v_x, min=0.0)


# ---------------------------------------------------------------------------
# Stability reward
# ---------------------------------------------------------------------------

def ang_vel_xy_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body roll and pitch rates (body-frame ω_x and ω_y).

    Discourages the robot from rocking / tipping. Returns the L2 norm of
    the two angular velocity components. Weight should be negative.
    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]   # (N, 2) — roll, pitch rates
    return torch.sum(torch.square(ang_vel_xy), dim=1)


# ---------------------------------------------------------------------------
# Torque threshold penalty
# ---------------------------------------------------------------------------

def joint_torque_threshold_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques that exceed a threshold (soft limit).

    Only the excess above the threshold is penalised, so normal-effort
    motion is free — only truly over-stressed joints are discouraged.

    Returns sum of excess torque magnitudes per env (weight should be negative).
    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # applied_torques: (N, num_joints)
    torques = asset.data.applied_torque
    if asset_cfg.joint_ids is not None:
        torques = torques[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(torques) - threshold, min=0.0)
    return excess.sum(dim=-1)


# ---------------------------------------------------------------------------
# Base height penalty
# ---------------------------------------------------------------------------

def base_height_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height if it drops below a specified minimum height.

    Returns (minimum_height - current_height) clipped to >= 0.
    Weight should be negative.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return torch.clamp(minimum_height - height, min=0.0)


# ---------------------------------------------------------------------------
# Foot clearance reward  (Option A — anti-slide)
# ---------------------------------------------------------------------------

def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_clearance: float = 0.02,
) -> torch.Tensor:
    """Reward each foot for being above the ground plane.

    This directly counters the sliding exploit: a sliding robot keeps all
    six feet on the ground, earning zero from this term.  A robot with a
    proper swing phase earns a positive signal for every foot that is
    airborne.

    Implemented via ``body_pos_w`` (world-frame body positions) so **no
    ContactSensor is required** — avoids the USD segfault issue.

    The reward per foot is ``clamp(foot_z - ground_z, 0, target_clearance)``
    normalised by ``target_clearance`` so the maximum contribution is 1.0 per
    foot, giving a total max of 6.0 for a perfect swing.

    Args:
        env:              The RL environment instance.
        asset_cfg:        SceneEntityCfg selecting the robot *and* its foot
                          body names (use ``body_names=[".*_foot_link"]``).
        target_clearance: Z-height (m) at which the reward saturates per foot.
                          Default 0.02 m (2 cm) — a gentle lift is enough.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # body_pos_w: (N, num_bodies, 3) — world-frame positions of all links.
    # asset_cfg.body_ids is resolved by Isaac Lab from the body_names regex.
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, 6, 3)
    foot_z = foot_pos_w[:, :, 2]                                   # (N, 6)

    # Clamp so each foot contributes in [0, target_clearance]
    clearance = torch.clamp(foot_z, min=0.0, max=target_clearance)  # (N, 6)
    # Normalise by target_clearance → max 1.0 per foot, 6.0 total
    return clearance.sum(dim=-1) / target_clearance                  # (N,)


# ---------------------------------------------------------------------------
# Joint motion reward with acceleration penalty
# ---------------------------------------------------------------------------

def joint_motion_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    accel_penalty_scale: float = 0.1,
) -> torch.Tensor:
    """Reward aggregate joint speed while penalising high joint acceleration.

    Combines two objectives into a single signed scalar:

    * **Motion incentive** ``+sum(|ω_j|)`` — the more total joint velocity the
      better, strongly discouraging all-joints-frozen sliding.
    * **Smoothness penalty** ``-accel_penalty_scale × sum(|α_j|)`` — penalises
      large joint accelerations (finite-differenced from velocity) to prevent
      the agent from learning chaotic vibration as a substitute for gait.

    The net value is positive when smooth motion is happening, and near zero
    (or negative) when joints are either static *or* jerking violently.

    Args:
        env:                The RL environment instance.
        asset_cfg:          SceneEntityCfg selecting the robot and joints.
        accel_penalty_scale: Relative weight of the acceleration penalty
                             vs. the velocity reward.  Default 0.1 means
                             acceleration is penalised at 10 % of the velocity
                             reward magnitude.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # --- velocity term ---
    joint_vel = asset.data.joint_vel  # (N, num_joints)
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    vel_reward = torch.sum(torch.abs(joint_vel), dim=-1)  # (N,)

    # --- acceleration penalty (finite-difference via joint_acc) ---
    # Isaac Lab exposes joint_acc on ArticulationData when the articulation
    # has retain_accelerations=True or when the solver computes it.
    # If unavailable, gracefully fall back to zero penalty.
    try:
        joint_acc = asset.data.joint_acc  # (N, num_joints)
        if asset_cfg.joint_ids is not None:
            joint_acc = joint_acc[:, asset_cfg.joint_ids]
        accel_penalty = torch.sum(torch.abs(joint_acc), dim=-1)  # (N,)
    except AttributeError:
        accel_penalty = torch.zeros_like(vel_reward)

    return vel_reward - accel_penalty_scale * accel_penalty  # (N,)


# ---------------------------------------------------------------------------
# Base orientation penalty  (two-zone roll / pitch)
# ---------------------------------------------------------------------------

def base_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_threshold_deg: float = 8.0,
    hard_multiplier: float = 10.0,
) -> torch.Tensor:
    """Two-zone penalty on torso pitch and roll using projected gravity.

    ``projected_gravity_b`` is the unit gravity vector expressed in the
    robot's body frame.  For a perfectly level robot it is ``[0, 0, -1]``;
    the x/y components become non-zero as the robot tilts.

    Since ``|proj_grav_xy|² = sin²(tilt_angle)`` the penalty is naturally
    proportional to the *angle*, not the rotation rate — unlike
    ``ang_vel_xy`` which only fires while the robot is actively rotating.

    Zone design
    ~~~~~~~~~~~
    * **Soft zone** ``[0°, soft_threshold_deg)``:  gentle quadratic
      ``sin²(θ)`` — small transient tilts during the step cycle cost very
      little, so normal gait dynamics are not over-penalised.
    * **Hard zone** ``[soft_threshold_deg, ∞)``:  soft penalty *plus*
      ``hard_multiplier × clamp(sin²(θ) − sin²(threshold), 0)`` — a steep
      linear-in-sin² ramp that makes sustained lean economically ruinous.

    At weight ``-5.0`` the magnitudes are:

    +--------+------------+---------+------------------+
    | Tilt   | soft       | hard    | total × -5       |
    +========+============+=========+==================+
    |  5°    | 0.008      | 0       | **-0.04 / step** |
    |  8°    | 0.019      | 0       | **-0.10 / step** |
    | 15°    | 0.067      | 0.48    | **-2.74 / step** |
    | 25°    | 0.179      | 1.60    | **-8.9 / step**  |
    +--------+------------+---------+------------------+

    At 15° the penalty exceeds the ``+2.0`` forward velocity reward,
    making a sustained backward pitch economically irrational.

    Args:
        env:                The RL environment instance.
        asset_cfg:          SceneEntityCfg selecting the robot.
        soft_threshold_deg: Tilt angle (°) below which only the gentle
                            quadratic applies.  Default ``8°``.
        hard_multiplier:    Extra scale in the hard zone.  Default ``10``
                            makes sustained lean > threshold ~10× more
                            expensive than the soft zone alone.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # projected_gravity_b: (N, 3) — gravity in body frame, unit vector.
    # x = pitch component, y = roll component.
    proj_grav_xy = asset.data.projected_gravity_b[:, :2]   # (N, 2)
    tilt_sq = proj_grav_xy.pow(2).sum(dim=-1)               # sin²(θ), (N,)

    # Soft zone: always-active light quadratic
    soft_penalty = tilt_sq

    # Hard zone: linear-in-sin² excess above the threshold
    threshold_sin_sq = math.sin(math.radians(soft_threshold_deg)) ** 2
    hard_penalty = hard_multiplier * torch.clamp(tilt_sq - threshold_sin_sq, min=0.0)

    return soft_penalty + hard_penalty  # (N,), weight must be negative
