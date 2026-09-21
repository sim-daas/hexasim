# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AX-12A servo-model sanity check on Hexapod-S0a-v0: hold a fixed pose for 3 s with six legs down.

Poses: the S0a rest pose (action 0) and π_base's old rest pose (tibia −30°, commanded through the action).
Reports root height, mean |tracking error|, fraction of torque-saturated joints, read-drop rate and joint
velocity RMS (oscillation check), per joint type.

Usage: python scripts/tests/ax12a_stand_test.py --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Hexapod-S0a-v0", num_envs=16)
cfg.events.reset_robot_base.params["pose_range"] = {}
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
cfg.terminations.base_contact = None
env = gym.make("Hexapod-S0a-v0", cfg=cfg).unwrapped
robot = env.scene["robot"]
act_model = robot.actuators["ax12a"]
names = robot.joint_names
stall = act_model.cfg.stall_torque

for label, tibia_action in (("S0a rest (tibia +30°)", 0.0), ("π_base pose (tibia −30°)", -math.radians(60.0) / 0.5)):
    env.reset()
    act = torch.zeros(env.num_envs, 18, device=env.device)
    act[:, [i for i, n in enumerate(names) if n.endswith("_tibia")]] = tibia_action
    err, sat, valid, qd2 = [], [], [], []
    for t in range(150):
        env.step(act)
        if t >= 50:  # after settling
            q_cmd = env.action_manager.get_term("joint_pos").processed_actions
            err.append((q_cmd - robot.data.joint_pos.torch).abs())
            sat.append((act_model.applied_effort.abs() >= 0.99 * stall).float())
            valid.append(env.proprio.mask)
            qd2.append(robot.data.joint_vel.torch.pow(2))
    err, sat, valid, qd = (torch.stack(x).mean((0, 1)) for x in (err, sat, valid, qd2))
    print(
        f"--- {label}: root z {robot.data.root_pos_w.torch[:, 2].mean().item():.3f} m,"
        f" drop rate {1 - valid.mean().item():.3f}"
    )
    for jt in ("_coxa", "_femur", "_tibia"):
        ids = [i for i, n in enumerate(names) if n.endswith(jt)]
        print(
            f"  {jt[1:]:>5}: |err| {err[ids].mean().item():.3f} rad, saturated {sat[ids].mean().item():.0%},"
            f" q̇ rms {qd[ids].mean().sqrt().item():.3f} rad/s"
        )
env.close()
app.close()
