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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi

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
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]  # (N, B, 3)
    body_z = body_pos_w[:, :, 2] - env.terrain_z(body_pos_w[..., :2])  # height above terrain (N, B)
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


# Link capsules (start, end, radius) in each link's own frame, fitted to the right-front USD meshes (all legs share
# them; phase2-progress §2.1.38). Tibia: knee-servo top → toe, radius covering 99 % of mesh points (bounding 3.6 cm).
TIBIA_CAPSULE = ((-0.01, 0.0, 0.035), TOE_OFFSET_TIBIA, 0.030)
FEMUR_CAPSULE = ((-0.01, 0.0, 0.0), (0.075, 0.0, 0.0), 0.040)
# Same-side neighbours in LEG_NAMES order: RF-RM, RM-RB, LF-LM, LM-LB.
ADJACENT_LEGS = ((0, 1), (1, 2), (3, 4), (4, 5))


def toe_pos_w(robot: Articulation, tibia_ids) -> torch.Tensor:
    """World-frame toe positions (N, 6, 3) for the given tibia body ids."""
    pos = robot.data.body_pos_w.torch[:, tibia_ids]
    quat = robot.data.body_quat_w.torch[:, tibia_ids]
    off = torch.tensor(TOE_OFFSET_TIBIA, device=pos.device).expand_as(pos)
    return pos + quat_apply(quat, off)


def segment_distance(p1, q1, p2, q2) -> torch.Tensor:
    """Closest distance between segments p1–q1 and p2–q2, batched over leading dims (Ericson, RTCD §5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, b = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d1 * d2).sum(-1)
    c, f = (d1 * r).sum(-1), (d2 * r).sum(-1)
    s = ((b * f - c * e) / (a * e - b * b).clamp(min=1e-12)).clamp(0.0, 1.0)
    t = (b * s + f) / e
    s = torch.where(t < 0.0, (-c / a).clamp(0.0, 1.0), torch.where(t > 1.0, ((b - c) / a).clamp(0.0, 1.0), s))
    t = t.clamp(0.0, 1.0)
    return ((p1 + d1 * s.unsqueeze(-1)) - (p2 + d2 * t.unsqueeze(-1))).norm(dim=-1)


def adjacent_leg_gaps(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Surface gap (m) between same-side neighbouring legs, min over their tibia/femur capsule pairs: (N, 4)."""
    robot = env.scene["robot"]
    caps = []
    for ids, (p, q, radius) in ((env._tibia_body_ids, TIBIA_CAPSULE), (env._femur_body_ids, FEMUR_CAPSULE)):
        pos, quat = robot.data.body_pos_w.torch[:, ids], robot.data.body_quat_w.torch[:, ids]
        ends = [pos + quat_apply(quat, torch.tensor(x, device=pos.device).expand_as(pos)) for x in (p, q)]
        caps.append((*ends, radius))
    a, b = (list(x) for x in zip(*ADJACENT_LEGS))
    gaps = [
        segment_distance(p1[:, a], q1[:, a], p2[:, b], q2[:, b]) - r1 - r2 for p1, q1, r1 in caps for p2, q2, r2 in caps
    ]
    return torch.stack(gaps).min(dim=0).values


def joint_leg_index(robot: Articulation, joint_ids) -> torch.Tensor:
    """Leg index (0..5, LEG_NAMES order) of every selected joint, from joint names."""
    names = [robot.joint_names[i] for i in joint_ids]
    return torch.tensor(
        [next(k for k, leg in enumerate(LEG_NAMES) if n.startswith(leg)) for n in names], device=robot.device
    )


def _swing_end(env: ManagerBasedRLEnv) -> float:
    """Phase at which swing ends; the single duty-factor source is ``env.cfg.duty_factor``."""
    return (1.0 - env.cfg.duty_factor) * 2.0 * math.pi


def foot_forces_nt(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-foot (N, 6) force along the terrain normal at the toe, and the tangential magnitude.

    Sensor bodies share the tibia order (asserted in ``HexapodEnv.load_managers``); the normal is the height map's
    at the toe xy — the contact point on a tibia whose toe is planted.
    """
    f = env.scene[sensor_cfg.name].data.net_forces_w.torch
    toe = toe_pos_w(env.scene["robot"], env._tibia_body_ids)
    n = env.terrain_normal(toe[..., :2])
    fn = (f * n).sum(dim=-1)
    return fn, torch.norm(f - fn.unsqueeze(-1) * n, dim=-1)


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
    return (error * is_swing).sum(dim=1) * env.cpg_active


def cpg_z_target_kernel_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_height: float = 0.038,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Bounded swing-only reward: mean over swinging legs of exp(−(e/σ)²), e = toe height − half-sine target.

    S0 replacement for the L1 form (phase2-progress §2.0.11): same target, but bounded in [0, 1] and positive, so
    a gait that tracks the toe target earns up to 1 per step and a mismatch cannot outweigh survival. Stance legs
    earn nothing; a leg dragged through its swing window earns ≈ 0.33 at step height 0.038 m and σ 0.02 m,
    less at σ 0.01 m (the S0a setting).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    phases = env.cpg_phase
    swing_end = _swing_end(env)
    is_swing = phases < swing_end
    target_z = step_height * torch.sin(phases / swing_end * math.pi) * is_swing
    toe_z = toe_pos_w(robot, asset_cfg.body_ids)[..., 2]
    k = torch.exp(-torch.square((toe_z - (target_z + env.cpg_foot_baseline)) / sigma)) * is_swing
    return k.sum(dim=1) / is_swing.sum(dim=1).clamp(min=1) * env.cpg_active


def cpg_force_target_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 error between the surface-normal foot force and a stance half-sine peaking at weight/3 (tripod support).

    L1, not MSE: PhysX produces 20-40 N single-step contact spikes (Phase 10 §7).
    """
    force_z = foot_forces_nt(env, sensor_cfg)[0]
    phases = env.cpg_phase
    swing_end = _swing_end(env)
    is_stance = phases >= swing_end
    stance_phase = (phases - swing_end) / (2.0 * math.pi - swing_end) * math.pi
    target = env.tripod_leg_load * torch.sin(stance_phase) * is_stance
    return torch.abs(force_z - target).mean(dim=1) * env.cpg_active


def coxa_leg_geometry(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Body-frame toe positions from the base, p_toe, and from each coxa joint, r (both (N, 6, 3))."""
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, 6, -1)
    toe = toe_pos_w(robot, env._tibia_body_ids)
    p_toe = quat_apply_inverse(quat, toe - robot.data.root_pos_w.torch.unsqueeze(1))
    r = quat_apply_inverse(quat, toe - robot.data.body_pos_w.torch[:, env._coxa_body_ids])
    return p_toe, r


def coxa_stroke_weight(env: ManagerBasedRLEnv, twist: torch.Tensor) -> torch.Tensor:
    """Per leg (N, 6): sign × reach — the coxa rate multiplier that pays the toe stroke the body twist needs.

    A planted toe must move at v_toe = −(v + ω ẑ × p_toe) in the body frame; +coxa moves it along
    t⁺ = ẑ × r, r = p_toe − p_coxa. Sign of cos(t⁺, v_toe), 0 where |cos| < 0.2 (sweep nearly orthogonal); for a
    pure forward twist it equals ``COXA_BACKWARD_SIGN`` (asserted in ``HexapodEnv.load_managers``). The sign flips,
    correctly, when a toe sits inside its coxa relative to the base, and there a tiny toe motion needs a large coxa
    rate — so the rate is weighted by the horizontal reach |r_xy| relative to that leg's rest-pose reach: the term
    pays toe stroke speed, not coxa spin (phase2-progress §2.1.26: tucked toes spun at 2 rad/s for reward).
    """
    p_toe, r = coxa_leg_geometry(env)
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, 6, -1)
    toe = toe_pos_w(robot, env._tibia_body_ids)
    p_toe = quat_apply_inverse(quat, toe - robot.data.root_pos_w.torch.unsqueeze(1))
    r = quat_apply_inverse(quat, toe - robot.data.body_pos_w.torch[:, env._coxa_body_ids])
    t = torch.stack((-r[..., 1], r[..., 0]), dim=-1)
    v_toe = -(twist[:, None, :2] + twist[:, None, 2:3] * torch.stack((-p_toe[..., 1], p_toe[..., 0]), dim=-1))
    cos = (t * v_toe).sum(-1) / (t.norm(dim=-1) * v_toe.norm(dim=-1)).clamp(min=1e-6)
    return torch.sign(cos) * (cos.abs() >= 0.2) * r[..., :2].norm(dim=-1) / env.coxa_rest_reach


def cpg_coxa_contact_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    max_rate: float = math.pi,
) -> torch.Tensor:
    """Signed coxa sweep in the direction the commanded motion needs (v7 §2.6: "projected on commanded heading"), on
    planted stance legs, clipped to ±max_rate per leg.

    Signed, not abs: over a stance the integral is the net stroke, so buzzing the coxa back and forth earns ~0.
    ``env.coxa_stroke_weight()``: the fixed backward sign in Flat/S0a; in S0b ``coxa_stroke_weight`` of the command
    (reach-weighted), so turning
    pays the two sides in opposite directions (phase2-progress §2.1.21) — the fixed sign fought every turn.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    sign = env.coxa_stroke_weight()[:, joint_leg_index(robot, asset_cfg.joint_ids)]
    planted = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
    is_stance = env.cpg_phase >= _swing_end(env)
    rate = torch.clamp(sign * joint_vel, -max_rate, max_rate)
    return (rate * (is_stance & planted)).sum(dim=1) * env.cpg_active


def cpg_stance_rigidity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """|q̇| of every selected joint whose leg is in CPG stance (leg of each joint resolved from its name)."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    is_stance = (env.cpg_phase >= _swing_end(env))[:, joint_leg_index(robot, asset_cfg.joint_ids)]
    return (torch.abs(joint_vel) * is_stance).sum(dim=1) * env.cpg_active


# ---------------------------------------------------------------------------
# S0b: command tracking, phase-free floors (paradigm spec §5), standing, terrain-aware termination
# ---------------------------------------------------------------------------


def track_lin_vel_xy(env: ManagerBasedRLEnv, command_name: str, std: float, l1_scale: float) -> torch.Tensor:
    """Command tracking of the body xy velocity: exp(−|e|²/σ²) − l1_scale·|e|₁, e = v − v_cmd (per step).

    The L1 part keeps a slope far from the command, where the kernel is flat. A gait-period-averaged v was tried
    and made tracking worse (phase2-progress §2.1.8).
    """
    e = env.scene["robot"].data.root_lin_vel_b.torch[:, :2] - env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-e.square().sum(dim=1) / std**2) - l1_scale * e.abs().sum(dim=1)


def track_ang_vel_z(env: ManagerBasedRLEnv, command_name: str, std: float, l1_scale: float) -> torch.Tensor:
    """Yaw-rate tracking: exp(−e²/σ²) − l1_scale·|e|, e = ω_z − ω_z,cmd (body frame, per step)."""
    e = env.scene["robot"].data.root_ang_vel_b.torch[:, 2] - env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-e.square() / std**2) - l1_scale * e.abs()


class swing_clearance(ManagerTermBase):
    """Phase-free swing floor: per foot, peak toe height above the terrain over an airborne interval, scored once at
    touchdown as min(peak/h_c, 1). Contact from the sim sensor (reward only, never an input). Standing envs earn 0.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peak = torch.zeros(env.num_envs, 6, device=env.device)

    def reset(self, env_ids=None):
        self.peak[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, height: float) -> torch.Tensor:
        toe = toe_pos_w(env.scene["robot"], env._tibia_body_ids)
        h = toe[..., 2] - env.terrain_z(toe[..., :2])
        contact = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
        score = (torch.clamp(self.peak / height, max=1.0) * (contact & (self.peak > 0.0))).sum(dim=1)
        self.peak = torch.where(contact, torch.zeros_like(self.peak), torch.maximum(self.peak, h))
        return score * env.cpg_active


def stance_rigidity_contact_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, joint_scale: float
) -> torch.Tensor:
    """Phase-free stance floor (spec §5): on legs in measured contact, toe tangential speed (m/s) plus
    ``joint_scale`` × Σ|q̇| (rad/s) of that leg's joints, coxa included. L1. ``asset_cfg`` selects the joints.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ids = env._tibia_body_ids
    contact = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
    quat = robot.data.body_quat_w.torch[:, ids]
    arm = quat_apply(quat, torch.tensor(TOE_OFFSET_TIBIA, device=env.device).expand(*quat.shape[:-1], 3))
    v_toe = robot.data.body_link_lin_vel_w.torch[:, ids] + torch.cross(
        robot.data.body_link_ang_vel_w.torch[:, ids], arm, dim=-1
    )
    toe = robot.data.body_pos_w.torch[:, ids] + arm
    n = env.terrain_normal(toe[..., :2])
    v_t = torch.norm(v_toe - (v_toe * n).sum(-1, keepdim=True) * n, dim=-1)
    qd = torch.abs(robot.data.joint_vel.torch[:, asset_cfg.joint_ids])
    leg_contact = contact[:, joint_leg_index(robot, asset_cfg.joint_ids)]
    return (v_t * contact).sum(dim=1) + joint_scale * (qd * leg_contact).sum(dim=1)


def standing_contact_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Standing hold phase only (after the settle window): fraction of feet in contact."""
    contact = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
    return contact.float().mean(dim=1) * env.stand_hold


def standing_joint_speed_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Standing hold phase only: Σ|q̇| over all joints (rad/s) — the stance floor misses legs held in the air."""
    return env.scene["robot"].data.joint_vel.torch.abs().sum(dim=1) * env.stand_hold


def standing_load_share_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Standing hold phase only: Σ_feet |F_n − W/6| / W — 0 when all six feet carry an equal share, ≈ 1 on a tripod.

    L1 on the terrain-normal force (PhysX contact spikes, rule 3); smooth where the 1 N contact test flickers.
    W = robot weight per env (``tripod_leg_load`` = W/3).
    """
    fn = foot_forces_nt(env, sensor_cfg)[0]
    w = 3.0 * env.tripod_leg_load  # (N, 1)
    return (fn - w / 6.0).abs().sum(dim=1) / w.squeeze(1) * env.stand_hold


def standing_pose_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Standing mode only: Σ|q − q_rest| (rad). A dense pull home for every joint — the policy otherwise kept the
    right-back leg tucked (coxa at its stop, femur up) although standing evenly paid more (phase2-progress §2.1.14).
    """
    robot = env.scene["robot"]
    return (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs().sum(dim=1) * ~env.cpg_active


def stance_footprint_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Σ over legs in contact of | |r_xy| − rest |r_xy| | (m): horizontal toe distance from its coxa axis vs the rest
    pose. The distance is invariant to the coxa sweep, so the term keeps legs extended without touching the stride.

    Without it the four middle/back toes migrated under their hips (|r_xy| ≈ 1 cm vs ≈ 6 cm at rest): the cheapest
    holding-torque posture, 25 % narrower support polygon, inward shins, ±48° coxa arcs (phase2-progress §2.1.34).
    """
    _, r = coxa_leg_geometry(env)
    contact = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
    return ((r[..., :2].norm(dim=-1) - env.coxa_rest_reach).abs() * contact).sum(dim=1)


class swing_radial_motion(ManagerTermBase):
    """Σ over airborne legs of |Δ|r_xy|| / dt (m/s): in/out toe motion relative to the coxa axis during swing.

    Coxa sweep (constant |r_xy|) and a vertical lift are free, so only a coordinated femur + tibia lift avoids it;
    π_S0 folded the tibia (toe 5 cm in, back, in again) and π_S0a lifted with the femur (toe 7 cm out)
    (phase2-progress §2.1.36–37). Standing envs score 0.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev = torch.zeros(env.num_envs, 6, device=env.device)
        self.valid = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        reach = coxa_leg_geometry(env)[1][..., :2].norm(dim=-1)
        air = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) <= 1.0
        rate = ((reach - self.prev).abs() * air * self.valid).sum(dim=1) / env.step_dt
        self.prev, self.valid[:] = reach, True
        return rate * env.cpg_active


class touchdown_reach(ManagerTermBase):
    """Σ over legs touching down this step of | |r_xy| − rest |r_xy| | (m): the landing distance from the coxa axis.

    Replaces the stance footprint, which π_S0 met by bending the tibia further (§2.1.35–37). Standing envs score 0.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev = torch.ones(env.num_envs, 6, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.prev[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        reach = coxa_leg_geometry(env)[1][..., :2].norm(dim=-1)
        contact = torch.norm(env.scene[sensor_cfg.name].data.net_forces_w.torch, dim=-1) > 1.0
        landing = contact & ~self.prev
        self.prev = contact
        return ((reach - env.coxa_rest_reach).abs() * landing).sum(dim=1) * env.cpg_active


def leg_proximity_penalty(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    """Σ over same-side neighbouring leg pairs of max(0, margin − surface gap) (m). Self-collision is off in sim, so
    without this the legs pass through each other; the margin covers real-robot backlash and tracking error
    (phase2-progress §2.1.38). Zero while the legs keep the margin, so stride and turning are free when there is room.
    """
    return (margin - adjacent_leg_gaps(env)).clamp(min=0.0).sum(dim=1)


def action_beyond_clip_penalty(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Σ |requested joint target − clipped target| (rad): 0 while every target is inside the servo limits.

    Past the clip the reward cannot depend on the action, so a joint parked beyond its limit gets no gradient back
    (phase2-progress §2.1.12: right-back coxa held at its −55° stop). This gives it one.
    """
    term = env.action_manager.get_term(action_name)
    requested = term.raw_actions * term._scale + term._offset
    return (requested - term.processed_actions).abs().sum(dim=1)


def root_height_above_terrain_below(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """Termination: root height above the terrain under it below ``minimum_height``."""
    root = env.scene["robot"].data.root_pos_w.torch
    return root[:, 2] - env.terrain_z(root[:, :2]) < minimum_height


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
