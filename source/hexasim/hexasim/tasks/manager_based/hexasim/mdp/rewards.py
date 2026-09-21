# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the hexapod locomotion environment.

Functions
---------
joint_pos_target_l2            : penalise joint deviation from a target value.
forward_velocity               : reward forward body-frame velocity.
lateral_velocity_penalty       : penalise sideways drift.
stand_still_penalty            : penalise slow forward motion (cancels alive bonus).
ang_vel_xy_penalty             : penalise roll/pitch rates.
joint_torque_threshold_penalty : penalise torques above threshold.
base_height_penalty            : penalise root height below minimum.
foot_clearance_reward          : reward feet for being off the ground (anti-slide).
joint_motion_reward            : reward joint velocity while penalising acceleration.
base_orientation_penalty       : two-zone roll/pitch angle penalty (projected gravity).
non_foot_body_ground_penalty   : penalise femur/coxa bodies approaching the ground.
cpg_*                          : CPG-phase shaping terms (clock, swing height, stance force, coxa sweep, rigidity).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

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
    forward_vel = asset.data.root_lin_vel_b[:, 0]  # x-component (forward)
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
    lateral_vel = asset.data.root_lin_vel_b[:, 1]  # y-component (sideways)
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
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]  # (N, 2) — roll, pitch rates
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
    """Anti-slide: sum over feet of toe height clamped to [0, target_clearance], normalised (max 6.0).

    ``asset_cfg.body_names`` selects the six tibia links; the toe point is derived from them
    (the URDF foot_link is merged into the tibia at import).
    ponytail: height above world z = 0, flat ground only; use terrain height at S0b.
    """
    toe_z = toe_pos_w(env.scene[asset_cfg.name], asset_cfg.body_ids)[..., 2]
    return torch.clamp(toe_z, min=0.0, max=target_clearance).sum(dim=-1) / target_clearance


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
    proj_grav_xy = asset.data.projected_gravity_b[:, :2]  # (N, 2)
    tilt_sq = proj_grav_xy.pow(2).sum(dim=-1)  # sin²(θ), (N,)

    # Soft zone: always-active light quadratic
    soft_penalty = tilt_sq

    # Hard zone: linear-in-sin² excess above the threshold
    threshold_sin_sq = math.sin(math.radians(soft_threshold_deg)) ** 2
    hard_penalty = hard_multiplier * torch.clamp(tilt_sq - threshold_sin_sq, min=0.0)

    return soft_penalty + hard_penalty  # (N,), weight must be negative


# ---------------------------------------------------------------------------
# Non-foot body ground proximity penalty
# ---------------------------------------------------------------------------


def non_foot_body_ground_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_height: float = 0.05,
) -> torch.Tensor:
    """Penalise femur or coxa links when they approach the ground plane.

    Targets the **bent-knee pathology**: if the femur (second-to-last leg
    segment) hyper-flexes and drags near the ground, it is penalised just
    like the base-height penalty — no ContactSensor required.

    Select the bodies to monitor via ``asset_cfg.body_names``, e.g.
    ``body_names=[".*femur.*"]`` for all six femur links, or include
    ``".*coxa.*"`` if coxa drag is also observed.

    Returns ``sum(clamp(minimum_height - body_z, 0))`` over selected bodies.
    Weight must be negative.

    Args:
        env:            The RL environment instance.
        asset_cfg:      SceneEntityCfg with body_names selecting the links
                        to monitor (femur, coxa, etc.).
        minimum_height: Z threshold (m) below which the penalty fires.
                        Default 0.05 m — just above the episode termination
                        floor, so it only fires on severe drooping.

    Shape: (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # body_pos_w: (N, num_bodies, 3) — same API used by foot_clearance_reward
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, B, 3)
    body_z = body_pos_w[:, :, 2]  # (N, B)
    return torch.clamp(minimum_height - body_z, min=0.0).sum(dim=-1)  # (N,)


# ---------------------------------------------------------------------------
# Leg geometry helpers
# ---------------------------------------------------------------------------

# Leg order shared by cpg_phase, tibia bodies, contact-sensor bodies and the
# per-type joint groups (asserted in HexapodEnv.__init__).
LEG_NAMES = ("right_front", "right_middle", "right_back", "left_front", "left_middle", "left_back")

# Toe point in the tibia-link frame (URDF foot_link, merged at USD import).
# Probe (scripts/tests/probe_layout.py): toe z = 0.2 mm on flat ground at rest.
TOE_OFFSET_TIBIA = (0.095, 0.0, -0.09)

# Sign that turns +coxa joint velocity into a backward toe sweep, per leg.
# Probe: +coxa moves right toes forward (+x body), left toes backward.
COXA_BACKWARD_SIGN = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


def toe_pos_w(robot: Articulation, tibia_ids) -> torch.Tensor:
    """World-frame toe positions (N, 6, 3) for the given tibia body ids."""
    pos = robot.data.body_pos_w.torch[:, tibia_ids]
    quat = robot.data.body_quat_w.torch[:, tibia_ids]
    off = torch.tensor(TOE_OFFSET_TIBIA, device=pos.device).expand_as(pos)
    return pos + quat_apply(quat, off)


def joint_leg_index(robot: Articulation, joint_ids) -> torch.Tensor:
    """Leg index (0..5, LEG_NAMES order) of every selected joint, from joint names."""
    names = [robot.joint_names[i] for i in joint_ids]
    return torch.tensor(
        [next(k for k, leg in enumerate(LEG_NAMES) if n.startswith(leg)) for n in names], device=robot.device
    )


def _swing_end(env: ManagerBasedRLEnv) -> float:
    """Phase at which swing ends; the single duty-factor source is ``env.cfg.duty_factor``."""
    return (1.0 - env.cfg.duty_factor) * 2.0 * math.pi


# ---------------------------------------------------------------------------
# CPG terms
# ---------------------------------------------------------------------------


def cpg_clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """12-D CPG phase clock, (sin, cos) interleaved per leg."""
    phases = env.cpg_phase
    return torch.stack((torch.sin(phases), torch.cos(phases)), dim=-1).view(env.num_envs, 12)


def cpg_z_target_l1_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_height: float = 0.05,
) -> torch.Tensor:
    """L1 swing-only penalty: toe height vs a half-sine of ``step_height`` above the adaptive contact baseline.

    Swing occupies phase [0, swing_end); stance is masked out (ground contact is the force term's job).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    phases = env.cpg_phase
    swing_end = _swing_end(env)
    is_swing = phases < swing_end
    target_z = step_height * torch.sin(phases / swing_end * math.pi) * is_swing
    toe_z = toe_pos_w(robot, asset_cfg.body_ids)[..., 2]
    error = torch.abs(toe_z - (target_z + env.cpg_foot_baseline)) / step_height
    return (error * is_swing).sum(dim=1)


def cpg_z_target_kernel_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_height: float = 0.038,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Bounded swing-only reward: mean over swinging legs of exp(−(e/σ)²), e = toe height − half-sine target.

    S0 replacement for the L1 form (phase2-progress §2.0.11): same target, but bounded in [0, 1] and positive, so
    a gait that tracks the toe target earns up to 1 per step and a mismatch cannot outweigh survival. Stance legs
    earn nothing; a leg dragged through its swing window earns ≈ 0.33 at step height 0.038 m and σ 0.02 m.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    phases = env.cpg_phase
    swing_end = _swing_end(env)
    is_swing = phases < swing_end
    target_z = step_height * torch.sin(phases / swing_end * math.pi) * is_swing
    toe_z = toe_pos_w(robot, asset_cfg.body_ids)[..., 2]
    k = torch.exp(-torch.square((toe_z - (target_z + env.cpg_foot_baseline)) / sigma)) * is_swing
    return k.sum(dim=1) / is_swing.sum(dim=1).clamp(min=1)


def cpg_force_target_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 error between vertical foot force and a stance half-sine peaking at weight/3 (tripod support).

    L1, not MSE: PhysX produces 20-40 N single-step contact spikes (Phase 10 §7).
    ponytail: world-Z == surface normal on flat ground only; project onto the terrain normal at S0b.
    """
    force_z = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, :, 2]
    phases = env.cpg_phase
    swing_end = _swing_end(env)
    is_stance = phases >= swing_end
    stance_phase = (phases - swing_end) / (2.0 * math.pi - swing_end) * math.pi
    target = env.tripod_leg_load * torch.sin(stance_phase) * is_stance
    return torch.abs(force_z - target).mean(dim=1)


def cpg_coxa_contact_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    max_rate: float = math.pi,
) -> torch.Tensor:
    """Signed backward coxa sweep on planted stance legs, clipped to ±max_rate per leg.

    Signed, not abs: over a stance the integral is the net backward stroke, so buzzing the coxa
    back and forth earns ~0 instead of farming |q̇| in both directions (v7 §2.6).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    sign = torch.tensor(COXA_BACKWARD_SIGN, device=env.device)[joint_leg_index(robot, asset_cfg.joint_ids)]
    planted = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
    is_stance = env.cpg_phase >= _swing_end(env)
    rate = torch.clamp(sign * joint_vel, -max_rate, max_rate)
    return (rate * (is_stance & planted)).sum(dim=1)


def cpg_stance_rigidity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """|q̇| of every selected joint whose leg is in CPG stance (leg of each joint resolved from its name)."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    is_stance = (env.cpg_phase >= _swing_end(env))[:, joint_leg_index(robot, asset_cfg.joint_ids)]
    return (torch.abs(joint_vel) * is_stance).sum(dim=1)


class LinearAnnealRewardWeight(ManagerTermBase):
    """Curriculum term: linearly interpolate a reward weight over an env-step window."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        term_name: str,
        initial_weight: float,
        target_weight: float,
        start_step: int,
        end_step: int,
    ) -> dict:
        progress = min(max((env.common_step_counter - start_step) / float(end_step - start_step), 0.0), 1.0)
        self._term_cfg.weight = initial_weight + progress * (target_weight - initial_weight)
        return {term_name: self._term_cfg.weight}
