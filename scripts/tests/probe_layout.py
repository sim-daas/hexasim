# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the robot's runtime layout: joint/body/sensor order, limits, masses, toe geometry, coxa sign.

Usage: python scripts/tests/probe_layout.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Hexapod-Flat-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, num_envs=1)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]
sensor = env.scene["contact_forces"]
from pxr import Usd  # noqa: E402

stage = env.sim.stage
print(
    "base prims:",
    [
        str(p.GetPath())
        for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
        if "base_link" in str(p.GetPath())
    ][:6],
)
print("joint_names:", robot.joint_names)
print("body_names:", robot.data.body_names)
print("sensor bodies:", sensor.body_names)
print("joint limits:", robot.data.joint_pos_limits.torch[0].tolist())
print("default q:", robot.data.default_joint_pos.torch[0].tolist())
print("masses:", dict(zip(robot.data.body_names, robot.data.body_mass.torch[0].tolist())))

# settle at rest with zero action
zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for _ in range(100):
    env.step(zero)
tib = [i for i, n in enumerate(robot.data.body_names) if "tibia" in n]
d = robot.data
root_p, root_q = d.root_pos_w.torch, d.root_quat_w.torch
off = torch.tensor([0.095, 0.0, -0.09], device=env.device).expand(len(tib), 3)
toe_w = d.body_pos_w.torch[0, tib] + quat_apply(d.body_quat_w.torch[0, tib], off)
print("root z:", root_p[0, 2].item())
print("tibia origin z:", d.body_pos_w.torch[0, tib, 2].tolist())
print("toe z:", toe_w[:, 2].tolist())
toe_b = quat_apply_inverse(root_q.expand(len(tib), 4), toe_w - root_p)
print("toe body-frame (nominal foothold):", toe_b.tolist())
print("contact force z:", sensor.data.net_forces_w.torch[0, :, 2].tolist())
print("applied torque:", dict(zip(robot.joint_names, d.applied_torque.torch[0].tolist())))

# coxa sign: +action on every coxa, see which way each toe moves in body x
before = toe_b.clone()
act = zero.clone()
coxa = [i for i, n in enumerate(robot.joint_names) if "coxa" in n]
act[:, coxa] = 0.4
for _ in range(30):
    env.step(act)
toe_w = d.body_pos_w.torch[0, tib] + quat_apply(d.body_quat_w.torch[0, tib], off)
after = quat_apply_inverse(d.root_quat_w.torch.expand(len(tib), 4), toe_w - d.root_pos_w.torch)
print("toe dx body for +coxa:", dict(zip([robot.data.body_names[i] for i in tib], (after - before)[:, 0].tolist())))
env.close()
app.close()
