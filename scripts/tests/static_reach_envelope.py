# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static reach and torque envelope (phase2-progress §2.3.21): tallest step and widest gap the robot can clear.

Pure kinematics on the sim-validated leg model (``hexasim.control.leg_kinematics``) — no Kit. A toe pose carries
load if, with the tripod leg load (weight / 3) pushing straight up on the toe, the femur and tibia holding torques
(lever arm × load, as in ``static_torque_envelope.py``) stay within ``--frac`` of AX-12A stall and the toe is
outside the femur joint (no foot under the body). Level body, quasi-static, vertical load only.

Usage: python scripts/tests/static_reach_envelope.py [--frac 0.6]
"""

import argparse
import math

import torch
from hexasim.control.leg_kinematics import COXA_OFF, FEMUR_OFF, MOUNT_POS, MOUNT_YAW, forward_kinematics
from hexasim.tasks.manager_based.hexasim.mdp.rewards import LEG_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("--mass", type=float, default=3.37, help="robot 3.22 kg (URDF) + 0.15 kg camera mast")
parser.add_argument("--stall", type=float, default=1.5, help="AX-12A stall torque, N·m (datasheet, 12 V)")
parser.add_argument("--frac", type=float, default=0.6, help="allowed share of stall (spec §4.2 gate: 0.6)")
parser.add_argument("--belly", type=float, default=0.033, help="base bottom plate below the root, m (§2.1.19)")
parser.add_argument("--margin", type=float, default=0.01, help="clearance kept to an edge / step top, m")
args = parser.parse_args()

LIMITS = ((-55.0, 55.0), (-98.0, 90.0), (-45.0, 125.0))  # coxa, femur, tibia, degrees (assets/robot.py = action clip)
load = args.mass * 9.81 / 3.0
tau_max = args.frac * args.stall


def envelope(q3: torch.Tensor):
    """q3 (N, 3) one leg's coxa / femur / tibia → toe (N, 6, 3) body frame, load-bearing mask (N, 6), femur torque."""
    toe = forward_kinematics(q3.repeat_interleave(6, dim=1))
    az = (MOUNT_YAW + q3[:, :1]).unsqueeze(-1)
    radial = torch.cat([torch.cos(az), torch.sin(az)], -1)
    r = ((toe - MOUNT_POS)[..., :2] * radial).sum(-1)  # signed toe distance from the coxa axis, along the leg
    r_femur = COXA_OFF[0]
    r_tibia = r_femur + (FEMUR_OFF[0] * torch.cos(q3[:, 1:2]) - FEMUR_OFF[2] * -torch.sin(q3[:, 1:2]))
    tau_f, tau_t = load * (r - r_femur).abs(), load * (r - r_tibia).abs()
    return toe, (tau_f <= tau_max) & (tau_t <= tau_max) & (r >= r_femur), tau_f


# Self-check against the simulator: the S0a rest pose (femur +30°, tibia +30°) holds tripod stance at 12 % of stall
# (lever-arm torque, static_torque_envelope.py, assets/robot.py) and stands ≈ 0.17 m high (phase2-progress §2.1).
rest = torch.tensor([[0.0, math.radians(30.0), math.radians(30.0)]])
toe0, ok0, tau0 = envelope(rest)
H0 = -toe0[0, 0, 2].item()
assert ok0.all() and abs(tau0[0, 0].item() / args.stall - 0.12) < 0.05 and 0.15 < H0 < 0.19, (tau0, H0)

# ponytail: 1.5° joint grid (≈ 3 mm toe resolution), finer grid if a limit lands within a few mm of a decision
axes = [torch.deg2rad(torch.arange(lo, hi + 0.1, 1.5)) for lo, hi in LIMITS]
toe, ok, _ = envelope(torch.cartesian_prod(*axes))
z_load = toe[..., 2][ok]
H_max = -z_load.min().item()
print(f"leg load {load:.1f} N, torque limit {tau_max:.2f} N·m ({args.frac:.0%} of stall)")
print(f"toe depth below the root: rest pose {H0 * 100:.1f} cm, deepest load-bearing {H_max * 100:.1f} cm")
print(f"highest unloaded toe (swing): {toe[..., 2].max() * 100:+.1f} cm relative to the root")


def stroke(H: float) -> list[float]:
    """Per leg: longest fore-aft toe travel at depth H under load along one line of constant lateral offset, m."""
    out = []
    for leg in range(6):
        m = ok[:, leg] & ((toe[:, leg, 2] + H).abs() < 0.003)
        x, ybin = toe[m, leg, 0], (toe[m, leg, 1] / 0.005).round().long()
        # ponytail: x-range per 5 mm lateral bin, assumes the slice is one interval (convex-ish workspace)
        out.append(max((x[ybin == b].max() - x[ybin == b].min()).item() for b in ybin.unique()) if m.any() else 0.0)
    return out


print(f"\n{'body height':>12s} {'step (level body)':>18s} {'stroke min / per leg, cm':>40s} {'gap':>6s}")
for H in [0.10, 0.12, 0.14, 0.16, round(H0, 3), 0.18, 0.19]:
    # Step: low feet at depth H, high feet on the step at depth H − h; the belly must clear the step top.
    step = H - args.belly - args.margin
    s = stroke(H)
    # Gap: a foot is carried one stroke per step, and needs a margin on both edges.
    gap = min(s) - 2 * args.margin
    legs = " ".join(f"{v * 100:4.1f}" for v in s)
    print(f"{H * 100:10.1f} cm {step * 100:15.1f} cm {min(s) * 100:9.1f} / {legs} {max(gap, 0) * 100:5.1f}")
print("leg order:", " ".join(LEG_NAMES))
