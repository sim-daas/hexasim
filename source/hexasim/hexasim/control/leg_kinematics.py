# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic forward/inverse kinematics for one PhantomX Mark IV leg (coxa about local +Z, femur and tibia about
local +Y), body frame throughout.

Constants were measured directly from the simulated robot (not hand-read from the USD/URDF): perturb one joint
at a time through `env.step()` on the `Hexapod-S2-v0` task (18 parallel envs, one perturbed joint per env, PD
settled via the real actuator model — not `Articulation.set_joint_position_target`, which is inert here because
S2 uses the explicit AX-12A actuator model with PhysX joint stiffness/damping = 0), then read back
`body_pos_w`/`body_quat_w`. Isaac Lab's `root_quat_w`/`body_quat_w` and `isaaclab.utils.math` are **XYZW**, not
WXYZ (see `CLAUDE.md`'s `_yaw_offsets_w` postmortem) — `quat_mul`/`quat_apply`/`quat_apply_inverse` all assume it.
Validated: `forward_kinematics()` reproduces the simulator's own `toe_pos_w` (mdp.rewards) to < 0.01 mm over 64
envs x 6 legs x random joint angles spanning ~85% of each joint's configured range.

LEG_NAMES order throughout (matches hexasim...mdp.rewards.LEG_NAMES and the per-leg grouping of
robot.joint_names): right_front, right_middle, right_back, left_front, left_middle, left_back.
"""

from __future__ import annotations

import math

import torch

from hexasim.tasks.manager_based.hexasim.mdp.rewards import TOE_OFFSET_TIBIA

# Coxa joint position, body frame, metres (x forward, y left, z up).
MOUNT_POS = torch.tensor(
    [
        [0.125, -0.065, 0.0],  # right_front
        [0.0, -0.100, 0.0],  # right_middle
        [-0.125, -0.065, 0.0],  # right_back
        [0.125, 0.065, 0.0],  # left_front
        [0.0, 0.100, 0.0],  # left_middle
        [-0.125, 0.065, 0.0],  # left_back
    ]
)

# Coxa zero-angle orientation about body +Z, radians. q_coxa = 0 points the leg along this body-frame azimuth.
MOUNT_YAW = torch.tensor([math.radians(a) for a in (-45.0, -90.0, -135.0, 45.0, 90.0, 135.0)])

# Coxa -> femur joint offset, in the coxa body's own (rotating) local frame; identical for all 6 legs.
COXA_OFF = torch.tensor([0.05291, 0.0, 0.0])
# Femur -> tibia joint offset, in the femur body's own local frame; identical for all 6 legs.
FEMUR_OFF = torch.tensor([0.06, 0.0, -0.015])
# Tibia -> toe offset: reuse the project's own constant, do not redefine it.
TIBIA_OFF = torch.tensor(TOE_OFFSET_TIBIA)

_AXIS_Y = torch.tensor([0.0, 1.0, 0.0])
_AXIS_Z = torch.tensor([0.0, 0.0, 1.0])


def _rot_z(a: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices about +Z, angle a with shape (...)."""
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def _rot_y(a: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices about +Y, angle a with shape (...)."""
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 18) joint positions, `robot.joint_names` order (coxa x6, femur x6, tibia x6), radians.
    Returns toe positions in the body frame, (..., 6, 3), metres."""
    device = q.device
    mount_pos, mount_yaw = MOUNT_POS.to(device), MOUNT_YAW.to(device)
    coxa_off, femur_off, tibia_off = COXA_OFF.to(device), FEMUR_OFF.to(device), TIBIA_OFF.to(device)
    q_coxa, q_femur, q_tibia = q[..., 0:6], q[..., 6:12], q[..., 12:18]
    r_coxa = _rot_z(mount_yaw + q_coxa)
    p_femur = mount_pos + (r_coxa @ coxa_off)
    r_femur = r_coxa @ _rot_y(q_femur)
    p_tibia = p_femur + (r_femur @ femur_off)
    r_tibia = r_femur @ _rot_y(q_tibia)
    return p_tibia + (r_tibia @ tibia_off)


def inverse_kinematics(
    target: torch.Tensor,
    q_init: torch.Tensor,
    joint_limits: torch.Tensor,
    iters: int = 6,
    damping: float = 1e-3,
    max_step: float = 0.3,
) -> torch.Tensor:
    """Damped Newton IK (Levenberg-Marquardt) with per-leg 3x3 geometric Jacobians.

    target: (..., 6, 3) desired toe positions, body frame.
    q_init: (..., 18) warm start, `robot.joint_names` order. NEVER pass zeros — validated round-trip error from a
        zero cold start is tens of mm (sometimes 100+ mm) even at 10 iterations; from the robot's actual current
        `robot.data.joint_pos` (near the rest pose) it is a few mm at 5 iterations, and from a *warm* start
        (previous call's output, as consecutive 20 ms control steps give) it is sub-millimetre at 3 iterations.
        Callers MUST warm-start from the previous call's return value every step after the first.
    joint_limits: (18, 2) [low, high] radians — pass `robot.data.joint_pos_limits.torch[0]` (identical across
        envs). Every iterate is clamped into this range, so the result is always sim-executable; this is also
        numerically the same range as this task's `cfg.actions.joint_pos.clip`, so no separate action-side clip
        is needed.
    iters, damping, max_step: tuned against the validated FK above (iters=6, damping=1e-3, max_step=0.3 rad);
        do not change without re-running the round-trip check in Verification.
    """
    device = target.device
    joint_limits = joint_limits.to(device)
    lo, hi = joint_limits[:, 0], joint_limits[:, 1]
    mount_pos, mount_yaw = MOUNT_POS.to(device), MOUNT_YAW.to(device)
    coxa_off, femur_off, tibia_off = COXA_OFF.to(device), FEMUR_OFF.to(device), TIBIA_OFF.to(device)
    axis_y, axis_z = _AXIS_Y.to(device), _AXIS_Z.to(device)
    eye3 = torch.eye(3, device=device)

    q = q_init.clone()
    for _ in range(iters):
        q_coxa, q_femur, q_tibia = q[..., 0:6], q[..., 6:12], q[..., 12:18]
        r_coxa = _rot_z(mount_yaw + q_coxa)
        p_coxa = mount_pos
        p_femur = p_coxa + (r_coxa @ coxa_off)
        r_femur = r_coxa @ _rot_y(q_femur)
        p_tibia = p_femur + (r_femur @ femur_off)
        r_tibia = r_femur @ _rot_y(q_tibia)
        p_toe = p_tibia + (r_tibia @ tibia_off)

        ax_coxa = axis_z.broadcast_to(p_toe.shape)
        ax_femur = r_coxa @ axis_y
        ax_tibia = r_femur @ axis_y
        j0 = torch.cross(ax_coxa, p_toe - p_coxa, dim=-1)
        j1 = torch.cross(ax_femur, p_toe - p_femur, dim=-1)
        j2 = torch.cross(ax_tibia, p_toe - p_tibia, dim=-1)
        jac = torch.stack([j0, j1, j2], dim=-1)  # (..., 6, 3, 3)

        dp = (target - p_toe).unsqueeze(-1)
        jt = jac.transpose(-1, -2)
        dq = torch.linalg.solve(jt @ jac + damping * eye3, jt @ dp).squeeze(-1).clamp(-max_step, max_step)

        q_coxa = (q_coxa + dq[..., 0]).clamp(lo[0:6], hi[0:6])
        q_femur = (q_femur + dq[..., 1]).clamp(lo[6:12], hi[6:12])
        q_tibia = (q_tibia + dq[..., 2]).clamp(lo[12:18], hi[12:18])
        q = torch.cat([q_coxa, q_femur, q_tibia], dim=-1)
    return q
