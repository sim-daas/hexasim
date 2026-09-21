# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static torque envelope at the rest pose (paradigm spec §4.2 prerequisite, v7 §5.2).

Settles the robot (with camera mast) on six legs, then lifts tripod A (RF, RB, LM) and reports the
holding torque of the tripod-B stance joints against the AX-12A stall torque. Gate: femur stance
torque <= 60 % of stall, else redesign the rest pose before S0a.

Usage: python scripts/tests/static_torque_envelope.py --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Flat-v0")
parser.add_argument("--femur_deg", type=float, default=30.0, help="rest-pose femur angle to test")
parser.add_argument("--tibia_deg", type=float, default=-30.0, help="rest-pose tibia angle to test")
parser.add_argument("--payload", type=float, default=0.0, help="extra base payload, kg")
parser.add_argument("--stall", type=float, default=1.5, help="AX-12A stall torque, N·m (datasheet)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, num_envs=1)
cfg.events.reset_robot_base.params["pose_range"] = {}
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
cfg.curriculum = None
m = 0.15 + args.payload  # mast + payload on the base
cfg.events.camera_mast_mass.params["mass_distribution_params"] = (m, m)
cfg.scene.robot.init_state.joint_pos.update(
    {".*_femur": math.radians(args.femur_deg), ".*_tibia": math.radians(args.tibia_deg)}
)
env = gym.make(args.task, cfg=cfg).unwrapped
robot, sensor = env.scene["robot"], env.scene["contact_forces"]
names = robot.joint_names
tripod_a = (0, 2, 4)  # RF, RB, LM in leg order
femur_a = [names.index(n) for n in names if n.endswith("_femur")]
femur_a = [femur_a[k] for k in tripod_a]


body = robot.data.body_names
tibia_ids = [i for i, n in enumerate(body) if "tibia" in n]
femur_ids = [i for i, n in enumerate(body) if "femur" in n]


def lever_torques():
    """Per-leg femur/tibia holding torque from contact force and geometry: ((toe - joint) × F)·axis_y.

    Independent of the actuator's own torque estimate. Joint frames = child link origins, pitch axis = link y.
    """
    toe = toe_pos_w(robot, tibia_ids)[0]
    f = sensor.data.net_forces_w.torch[0]
    out = []
    for ids in (femur_ids, tibia_ids):
        p = robot.data.body_pos_w.torch[0, ids]
        axis = quat_apply(
            robot.data.body_quat_w.torch[0, ids], torch.tensor([[0.0, 1.0, 0.0]], device=env.device).expand(6, 3)
        )
        out.append((torch.cross(toe - p, f, dim=-1) * axis).sum(-1).abs())
    return out


def hold(action, steps=100):
    for _ in range(steps):
        env.step(action)
    return robot.data.applied_torque.torch[0].clone(), sensor.data.net_forces_w.torch[0, :, 2].clone()


print(f"total mass (incl. mast): {robot.data.body_mass.torch[0].sum().item():.3f} kg")
zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
env.reset()
tau6, f6 = hold(zero)
st6 = lever_torques()
print(
    "six-leg stance:  Fz", [round(v, 2) for v in f6.tolist()], f"root z {robot.data.root_pos_w.torch[0, 2].item():.3f}"
)

best = None
for sign in (1.0, -1.0):
    env.reset()
    act = zero.clone()
    act[0, femur_a] = sign * 1.0  # 0.5 rad femur offset on tripod A
    tau, f = hold(act)
    lifted = f[list(tripod_a)].abs().max().item()
    print(
        f"femur sign {sign:+.0f}: tripod-A max Fz {lifted:.2f} N, root z {robot.data.root_pos_w.torch[0, 2].item():.3f}"
    )
    if best is None or lifted < best[0]:
        best = (lifted, tau, f, lever_torques())
lifted, tau3, f3, st3 = best
assert lifted < 0.5, "could not unload tripod A"
print("tripod stance:   Fz", [round(v, 2) for v in f3.tolist()])

tripod_b = (1, 3, 5)
worst = {}
for jt in ("_coxa", "_femur", "_tibia"):
    ids = [i for i, n in enumerate(names) if n.endswith(jt)]
    six = tau6[ids].abs().max().item()
    three = tau3[[ids[k] for k in tripod_b]].abs().max().item()
    worst[jt] = three
    print(
        f"{jt[1:]:>6}: six-leg max {six:.3f} N·m ({six / args.stall:.0%}),"
        f" tripod max {three:.3f} N·m ({three / args.stall:.0%} of stall)"
    )
for k, jt in enumerate(("femur", "tibia")):
    six, three = st6[k].max().item(), st3[k][list(tripod_b)].max().item()
    print(
        f"lever-arm {jt}: six-leg max {six:.3f} N·m ({six / args.stall:.0%}),"
        f" tripod max {three:.3f} N·m ({three / args.stall:.0%})"
    )
worst["_femur"] = st3[0][list(tripod_b)].max().item()  # gate on lever-arm torques, not the actuator's PD estimate
ratio = worst["_femur"] / args.stall
print(
    f"pose femur {args.femur_deg:+.0f} tibia {args.tibia_deg:+.0f} payload {args.payload:.1f} kg:"
    f" tripod root z {robot.data.root_pos_w.torch[0, 2].item():.3f} m"
)
print("TORQUE_ENVELOPE", "PASS" if ratio <= 0.6 else "FAIL", f"femur tripod stance = {ratio:.0%} of stall")
env.close()
app.close()
