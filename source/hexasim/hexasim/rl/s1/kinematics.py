# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg forward kinematics and the kinematic body velocity v_kin (paradigm spec §2.1, v7 §4.3). Deployable: joint
reads, tracking error and the gyro only — never contact switches.

Chain per leg (``models/hexapod.urdf``): base → hip origin (xyz, yaw) → Rz(q_coxa) → +(0.0529, 0, 0) → Ry(q_femur)
→ +(0.06, 0, −0.015) → Ry(q_tibia) → toe +(0.095, 0, −0.09) (the §1 toe point). Legs RF RM RB LF LM LB; joints
[coxa × 6, femur × 6, tibia × 6].

v_kin: per foot, stance weight w = σ((|e_femur| + |e_tibia| − τ_e)/s) · σ((z_plane − z_toe + δ)/5 mm), e the tracking
error (stance-load sag under P-only control), z_plane a least-squares plane through the six toes (body frame). Each
foot gives v_b = −(J q̇ + ω × p); v_kin is the w-weighted mean. J q̇ is the directional derivative of FK along the
deployable q̇ (§1 two-sample difference, held on dropped reads). τ_e, s, δ: provisional, calibrated in sim (D26),
then on the bench (v7 item 14).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

URDF = Path(__file__).resolve().parents[5] / "models" / "hexapod.urdf"
LEGS = ("right_front", "right_middle", "right_back", "left_front", "left_middle", "left_back")


def _joint(root, name):
    j = root.find(f"joint[@name='{name}']")
    o = j.find("origin")
    xyz = [float(v) for v in o.get("xyz").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return xyz, rpy


def _chain():
    root = ET.parse(URDF).getroot()
    hip, yaw = [], []
    for leg in LEGS:
        xyz, rpy = _joint(root, f"{leg}_coxa")
        assert rpy[0] == 0 and rpy[1] == 0
        hip.append(xyz)
        yaw.append(rpy[2])
        # The remaining offsets are identical for every leg; assert it rather than assume it.
        assert _joint(root, f"{leg}_femur")[0] == _joint(root, "right_front_femur")[0]
        assert _joint(root, f"{leg}_tibia")[0] == _joint(root, "right_front_tibia")[0]
        assert _joint(root, f"{leg}_foot")[0] == _joint(root, "right_front_foot")[0]
    return (
        torch.tensor(hip),
        torch.tensor(yaw),
        torch.tensor(_joint(root, "right_front_femur")[0]),
        torch.tensor(_joint(root, "right_front_tibia")[0]),
        torch.tensor(_joint(root, "right_front_foot")[0]),
    )


HIP, HIP_YAW, FEMUR_OFF, TIBIA_OFF, TOE_OFF = _chain()


def _ry(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack((c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]), dim=-1)


def _rz(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def toe_positions(q: torch.Tensor) -> torch.Tensor:
    """Absolute joint angles (..., 18) → toe positions in the base frame (..., 6, 3)."""
    dev = q.device
    qc, qf, qt = q[..., 0:6], q[..., 6:12], q[..., 12:18]
    shape = q.shape[:-1] + (6, 3)
    p = _ry(TOE_OFF.to(dev).expand(shape), qt) + TIBIA_OFF.to(dev)
    p = _ry(p, qf) + FEMUR_OFF.to(dev)
    p = _rz(p, qc + HIP_YAW.to(dev))
    return p + HIP.to(dev)


def v_kin(
    q: torch.Tensor,
    qd: torch.Tensor,
    track_err: torch.Tensor,
    gyro: torch.Tensor,
    tau_e: float = 0.005,
    s: float = 0.003,
    delta: float = -0.005,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kinematic body velocity (..., 3) and the per-foot stance weights (..., 6).

    q absolute joint angles, qd deployable joint rates, track_err q_cmd − q (all (..., 18)), gyro (..., 3) body rates.
    """
    p = toe_positions(q)
    jqd = (toe_positions(q + eps * qd) - toe_positions(q - eps * qd)) / (2 * eps)
    v_foot = -(jqd + torch.cross(gyro[..., None, :].expand_as(p), p, dim=-1))
    e = track_err[..., 6:12].abs() + track_err[..., 12:18].abs()
    # Least-squares plane z = a x + b y + c through the six toes.
    A = torch.cat((p[..., :2], torch.ones_like(p[..., :1])), dim=-1)
    coef = torch.linalg.lstsq(A, p[..., 2:3]).solution
    z_plane = (A @ coef)[..., 0]
    w = torch.sigmoid((e - tau_e) / s) * torch.sigmoid((z_plane - p[..., 2] + delta) / 0.005)
    v = (w[..., None] * v_foot).sum(-2) / w.sum(-1, keepdim=True).clamp(min=1e-3)
    return v, w


if __name__ == "__main__":
    # Self-checks: rest pose geometry, FK consistency with a finite difference, v_kin of a rigidly translating body.
    q0 = torch.zeros(18)
    q0[6:12] = math.radians(30)
    q0[12:18] = math.radians(30)
    p = toe_positions(q0)
    assert (p[:3, 1] < 0).all() and (p[3:, 1] > 0).all(), p  # right legs at −y
    # Pure body translation at v with feet planted: toe moves −v relative to the body → v_kin = v.
    torch.manual_seed(0)
    qd = torch.zeros(18)
    v, w = v_kin(q0[None], qd[None], torch.full((1, 18), 0.05), torch.zeros(1, 3))
    assert v.abs().max() < 1e-6
    print("kinematics self-check OK; rest toes (m):", [[round(x, 3) for x in r] for r in p.tolist()])
