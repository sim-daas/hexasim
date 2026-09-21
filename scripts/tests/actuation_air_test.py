# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit PhysX drive vs explicit IdealPD at identical gains, robot floating in zero gravity (no contact).

Every joint tracks q_rest + 0.25·sin(2π·1 Hz·t) (action 0.5·sin). Reports tracking error and joint speed per joint
type, so actuation can be compared without ground contact.

Usage: python scripts/tests/actuation_air_test.py --headless --mode {implicit,explicit}
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["implicit", "explicit"], required=True)
parser.add_argument("--armature", type=float, default=0.0)
parser.add_argument("--physics_hz", type=int, default=200)
parser.add_argument("--ext_every_iter", action="store_true", help="PhysX enable_external_forces_every_iteration")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Hexapod-Flat-v0", num_envs=4)
cfg.curriculum = None
cfg.terminations.base_contact = None
cfg.sim.gravity = (0.0, 0.0, 0.0)  # floating, no contact (fix_root_link clashes with this USD's root layout)
cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
cfg.sim.physics = PhysxCfg(enable_external_forces_every_iteration=args.ext_every_iter)
cfg.sim.dt = 1.0 / args.physics_hz
cfg.decimation = args.physics_hz // 50
cfg.sim.render_interval = cfg.decimation
cfg.events.reset_robot_base.params["pose_range"] = {}
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
if args.mode == "explicit":
    cfg.scene.robot.actuators = {
        "all": IdealPDActuatorCfg(
            joint_names_expr=[".*"], stiffness=20.0, damping=0.5, effort_limit=3.0, armature=args.armature
        )
    }
env = gym.make("Hexapod-Flat-v0", cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]
names = robot.joint_names
types = {t: [i for i, n in enumerate(names) if n.endswith(t)] for t in ("_coxa", "_femur", "_tibia")}
err = {t: 0.0 for t in types}
vel = {t: 0.0 for t in types}
n = 0
for k in range(200):
    act = torch.full((env.num_envs, 18), 0.5 * math.sin(2 * math.pi * 1.0 * k * env.step_dt), device=env.device)
    env.step(act)
    if k >= 50:
        q_t = env.action_manager.get_term("joint_pos").processed_actions
        for t, ids in types.items():
            err[t] += (q_t[:, ids] - robot.data.joint_pos.torch[:, ids]).abs().mean().item()
            vel[t] += robot.data.joint_vel.torch[:, ids].pow(2).mean().item()
        n += 1
print(f"MODE {args.mode} armature {args.armature} physics {args.physics_hz} Hz ext_every_iter {args.ext_every_iter}")
d = robot.data
for attr in ("computed_torque", "applied_torque", "joint_effort_target", "joint_armature", "joint_vel_limits"):
    if hasattr(d, attr):
        print(f"  {attr}: {[round(x, 4) for x in getattr(d, attr).torch[0, [0, 6, 12]].tolist()]}")
for t in types:
    print(f"  {t[1:]:>5}: |q_target − q| {err[t] / n:.3f} rad, q̇ rms {(vel[t] / n) ** 0.5:.2f} rad/s (ideal 1.11)")
env.close()
app.close()
