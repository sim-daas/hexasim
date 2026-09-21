# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env foot material check for Hexapod-S0b-v0 (paradigm spec §1 friction invariant, v7 §5.6).

Flat terrain (μ 1, combine multiply), zero actions, 256 envs with their own sampled foot material.
1. Friction: gravity tilted by θ. An env must slide (> 5 cm in 2 s) if its μ_s < tan θ − 0.05 and hold (< 2 cm:
   servo sag creeps a little at 30°) if μ_d > tan θ + 0.15 — the robot lands from its spawn height with some
   tangential speed, and once sliding only μ_d stops it; within 0.12 of tan 30° a few envs still slip 6–11 cm
   (landing transient, per-leg load shift). I.e. μ_eff equals the sampled foot μ (the terrain's 1.0 multiplies
   away). An unbound foot material (PhysX default μ 0.5) fails the 30° hold side; an average combine fails the
   30° slide side.
2. Compliance: θ = 0, settled toe sink below the terrain surface; compliant envs must sink more than rigid ones
   (by > 0.2 mm on average), and the sink must fall with stiffness k (Spearman < −0.5). Prints the effective
   per-foot stiffness relative to the material's nominal k (PhysX applies k per contact point).

Usage: python scripts/tests/foot_material_test.py --headless
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def make(theta_deg):
    cfg = parse_env_cfg("Hexapod-S0b-v0", num_envs=args.num_envs)
    cfg.seed = 0
    cfg.scene.terrain.terrain_generator.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
    cfg.curriculum = None
    cfg.events.reset_robot_base.params["pose_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    th = math.radians(theta_deg)
    cfg.sim.gravity = (9.81 * math.sin(th), 0.0, -9.81 * math.cos(th))
    env = gym.make("Hexapod-S0b-v0", cfg=cfg).unwrapped
    env.reset()
    return env


def run(env, settle, measure):
    robot = env.scene["robot"]
    zero = torch.zeros(env.num_envs, 18, device=env.device)
    with torch.inference_mode():
        for _ in range(settle):
            env.step(zero)
        x0 = robot.data.root_pos_w.torch[:, 0].clone()
        for _ in range(measure):
            env.step(zero)
    return (robot.data.root_pos_w.torch[:, 0] - x0).abs()


ok = True
for theta in (20.0, 30.0):
    env = make(theta)
    drift = run(env, 25, 100)
    fm = env.foot_material
    t = math.tan(math.radians(theta))
    slide, hold = fm[:, 0] < t - 0.05, fm[:, 1] > t + 0.15
    bad_slide = (slide & (drift < 0.05)).sum().item()
    bad_hold = (hold & (drift > 0.02)).sum().item()
    print(
        f"theta {theta:.0f} (tan {t:.3f}): should slide {slide.sum().item()}, not sliding {bad_slide};"
        f" should hold {hold.sum().item()}, not holding {bad_hold}"
    )
    for name, m in (("slide", slide & (drift < 0.05)), ("hold", hold & (drift > 0.02))):
        for i in m.nonzero().flatten()[:6].tolist():
            print(
                f"   failed {name}: mu_s {fm[i, 0]:.3f} mu_d {fm[i, 1]:.3f} log_k {fm[i, 3]:.2f} drift {drift[i]:.4f}"
            )
    hc, hr = hold & (fm[:, 3] > 0), hold & (fm[:, 3] == 0)
    print(
        f"   hold drift: rigid median {drift[hr].median():.4f} max {drift[hr].max():.4f};"
        f" compliant median {drift[hc].median():.4f} max {drift[hc].max():.4f}"
    )
    ok &= bad_slide == 0 and bad_hold == 0
    env.close()

env = make(0.0)
run(env, 100, 1)
robot = env.scene["robot"]
toe = toe_pos_w(robot, env._tibia_body_ids)
sink = -(toe[..., 2] - env.terrain_z(toe[..., :2])).mean(dim=1)  # m below the surface
log_k = env.foot_material[:, 3]  # log10(1 + k / 1 kN·m⁻¹), 0 = rigid
rigid = log_k == 0
rank = lambda x: x.argsort().argsort().float()  # noqa: E731
rho = torch.corrcoef(torch.stack([rank(log_k[~rigid]), rank(sink[~rigid])]))[0, 1].item()
print(
    f"sink rigid {sink[rigid].mean() * 1e3:.2f} mm, compliant {sink[~rigid].mean() * 1e3:.2f} mm"
    f" (max {sink[~rigid].max() * 1e3:.2f} mm); Spearman(k, sink) = {rho:.2f}"
)
# Effective per-foot stiffness F_n / (sink − rigid sink) vs the material's nominal k.
fn = torch.norm(env.scene["contact_forces"].data.net_forces_w.torch, dim=-1).mean(dim=1)
k_nom = (10**log_k - 1) * 1000.0
k_eff = fn / (sink - sink[rigid].mean()).clamp(min=1e-5)
print(f"foot load {fn.mean():.2f} N; k_eff / k_nominal median {(k_eff[~rigid] / k_nom[~rigid]).median():.2f}")
ok &= sink[~rigid].mean() > sink[rigid].mean() + 2e-4 and rho < -0.5
servo = robot.actuators["ax12a"]
print("servo slope choices seen", torch.unique(servo.slope, return_counts=True))
print("FOOT_MATERIAL_TEST", "PASS" if ok else "FAIL")
env.close()
app.close()
