# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the left–right mirror of ``mdp.symmetry`` against the simulator (Hexapod-S0b-v0).

Gravity off, robots floating 0.3 m up. Pairs of envs: even env holds a random joint pose q (targets = q), odd env
holds mirror(q). After settling, body-frame toe positions must mirror (legs swapped, y negated) and the policy's
joint-position block must mirror (``mirror_policy_obs`` on [0:18]). Servo randomisation off; tolerance 5 mm / 0.02 rad
(backlash, quantisation). A wrong joint sign shows up as centimetres / tenths of a radian.

Usage: python scripts/tests/mirror_test.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import coxa_leg_geometry  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.symmetry import LEG_SWAP, mirror_actions, mirror_policy_obs  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

N = 32
cfg = parse_env_cfg("Hexapod-S0b-v0", num_envs=N)
cfg.curriculum = None
cfg.sim.gravity = (0.0, 0.0, 0.0)
cfg.events.reset_robot_base.params["pose_range"]["z"] = (0.3, 0.3)
cfg.drop_p0 = cfg.drop_k = 0.0
# Identical servos in every env (no per-env stall/slope/backlash/friction randomisation): pairs differ only by the
# mirror.
servo = cfg.scene.robot.actuators["ax12a"]
servo.stall_scale_range, servo.compliance_slope_choices, servo.backlash_range, servo.coulomb_range = (
    (1.0, 1.0),
    None,
    None,
    (0.0, 0.0),
)
pin_command(cfg, 0.0)
env = gym.make("Hexapod-S0b-v0", cfg=cfg).unwrapped
pin_terrain_level(env, 0)
torch.manual_seed(0)
a = (torch.rand(N // 2, 18, device=env.device) * 2 - 1) * 0.6  # raw actions: targets rest ± 0.3 rad
actions = torch.zeros(N, 18, device=env.device)
actions[0::2] = a
actions[1::2] = mirror_actions(a)
with torch.inference_mode():
    obs, _ = env.reset()
    for _ in range(50):
        obs, *_ = env.step(actions)
    p_toe = coxa_leg_geometry(env)[0]
    toe_even, toe_odd = p_toe[0::2], p_toe[1::2]
    expected = toe_even[:, LEG_SWAP] * torch.tensor([1.0, -1.0, 1.0], device=env.device)
    toe_errs = (toe_odd - expected).norm(dim=-1)
    toe_err = toe_errs.max().item()
    print(f"MIRROR toe median error {toe_errs.median().item() * 1000:.2f} mm")
    o = obs["policy"]
    q_err = (o[1::2, 0:18] - mirror_policy_obs(o[0::2])[:, 0:18]).abs().max().item()
print(f"MIRROR toe max error {toe_err * 1000:.2f} mm, joint-obs max error {q_err:.4f} rad")
ok = toe_err < 0.005 and q_err < 0.02
print("MIRROR_TEST", "PASS" if ok else "FAIL")
env.close()
app.close()
