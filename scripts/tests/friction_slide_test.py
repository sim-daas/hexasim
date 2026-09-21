# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Friction combine-mode check (paradigm spec §1): effective foot μ must equal the ground μ.

Ground μ_g = 0.4, foot μ = 1.0, combine = multiply -> μ_eff = 0.4, slip onset atan(0.4) = 21.8°.
Gravity is tilted by θ (an incline without terrain). The standing robot must hold at 18° and slide
at 26°; PhysX's default average combine (μ_eff 0.7, onset 35°) would hold at both.

Usage: python scripts/tests/friction_slide_test.py --headless [--combine average]
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--combine", default="multiply", help="negative control: --combine average must FAIL")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

MU_G = 0.4
drift = {}
for theta_deg in (18.0, 26.0):
    cfg = parse_env_cfg("Hexapod-Flat-v0", num_envs=4)
    cfg.scene.ground.spawn.physics_material.static_friction = MU_G
    cfg.scene.ground.spawn.physics_material.dynamic_friction = MU_G
    cfg.scene.ground.spawn.physics_material.friction_combine_mode = args.combine
    cfg.events.reset_robot_base.params["pose_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    th = math.radians(theta_deg)
    cfg.sim.gravity = (9.81 * math.sin(th), 0.0, -9.81 * math.cos(th))
    cfg.curriculum = None
    env = gym.make("Hexapod-Flat-v0", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(25):  # settle
        env.step(zero)
    x0 = robot.data.root_pos_w.torch[:, 0].clone()
    for _ in range(100):  # 2 s
        env.step(zero)
    drift[theta_deg] = (robot.data.root_pos_w.torch[:, 0] - x0).mean().item()
    print(f"theta {theta_deg:.0f} deg: mean root x drift over 2 s = {drift[theta_deg]:.4f} m")
    env.close()

ok = abs(drift[18.0]) < 0.01 and abs(drift[26.0]) > 0.10
print("FRICTION_TEST", "PASS" if ok else "FAIL", drift)
app.close()
