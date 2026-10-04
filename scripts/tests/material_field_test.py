# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spatial terrain materials check for Hexapod-S1-v0 (phase2-progress §2.2; ``mdp.materials``).

Flat tiles (geometry off; the material field is generated as in training), zero actions, standing command.
1. Build: the terrain mesh partition assigned every triangle (asserted inside ``build_terrain_materials``).
2. Read-back: for every foot in contact, the PhysX shape material equals the value in effect (μ_s, μ_d, −k), and
   that value is within one write quantum of the field target under the toe.
3. Friction from the field: gravity tilted 20° (tan 0.364). Envs whose six feet all stand on μ_s < 0.30 must slide
   (> 5 cm in 2 s); envs whose six feet all stand on μ_d > 0.55 must hold (< 2 cm). Envs on the soft-ground entries
   (``--task Hexapod-S2-v0``, k < 2e3) are reported separately: drift in two consecutive 2 s windows.
4. Stability: no NaN, max base speed on θ = 0 < 0.05 m/s after settling (rigid entries use k = 1e6).

Usage: python scripts/tests/material_field_test.py --headless [--num_envs 512]
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", default="Hexapod-S1-v0", help="Hexapod-S2-v0 adds the soft-ground entries")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
import warp as wp  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def make(theta_deg):
    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    cfg.seed = 0
    cfg.scene.terrain.terrain_generator.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
    cfg.curriculum = None
    cfg.events.reset_robot_base.params["pose_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}  # spread over the field
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    th = math.radians(theta_deg)
    cfg.sim.gravity = (9.81 * math.sin(th), 0.0, -9.81 * math.cos(th))
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    return env


def run(env, n):
    zero = torch.zeros(env.num_envs, 18, device=env.device)
    with torch.inference_mode():
        for _ in range(n):
            env.step(zero)


ok = True
env = make(0.0)
run(env, 100)
robot = env.scene["robot"]
fw = env.foot_writer
contact = torch.norm(env.scene["contact_forces"].data.net_forces_w.torch, dim=-1) > 1.0
mat = wp.to_torch(fw.view.get_material_properties())
cols = [c[0] for c in fw.cols]
phys = torch.stack((mat[:, cols, 0], mat[:, cols, 1], torch.log10(-mat[:, cols, 2])), dim=-1).to(env.device)
err_rb = (phys - fw.current).abs()[contact].max().item()
toe = toe_pos_w(robot, env._tibia_body_ids)
target = env.palette.at(env.material_field, toe[..., :2])
d = (target - fw.current).abs()[contact]
err_tg = max(d[:, 0].max().item(), d[:, 1].max().item())
err_k = d[:, 2].max().item()
t2 = err_rb < 1e-3 and err_tg <= fw.dmu + 1e-4 and err_k <= fw.dlogk + 1e-4
print(
    f"2. read-back max |Δ| {err_rb:.2e}; target vs in-effect μ {err_tg:.4f} (quantum {fw.dmu}), log k {err_k:.4f}"
    f" -> {'PASS' if t2 else 'FAIL'}"
)
v = robot.data.root_lin_vel_w.torch.norm(dim=-1)
t4 = bool(torch.isfinite(v).all() and v.max() < 0.05)
print(f"4. θ=0 max base speed {v.max():.4f} m/s -> {'PASS' if t4 else 'FAIL'}")
mu = fw.current[..., 0]
print(
    f"   μ_s under feet: min {mu.min():.2f} max {mu.max():.2f};"
    f" rigid feet {(fw.current[..., 2] > 5.9).float().mean():.2f}"
)
ok &= t2 and t4
env.close()

env = make(20.0)
run(env, 25)
x0 = env.scene["robot"].data.root_pos_w.torch[:, 0].clone()
cur0 = env.foot_writer.current.clone()
run(env, 100)
drift = (env.scene["robot"].data.root_pos_w.torch[:, 0] - x0).abs()
cur = env.foot_writer.current
# Footing at the start and the end of the measurement (an env sliding onto a grippy patch is not a failed hold).
slide = (cur0[..., 0] < 0.30).all(dim=1)
hold = (cur0[..., 1] > 0.55).all(dim=1) & (cur[..., 1] > 0.55).all(dim=1)
# Soft ground (k below the S1 library's 2e3, D39) is reported, not gated: its contact springs shear downhill.
soft = hold & (cur[..., 2] < 3.3).any(dim=1)
hold &= ~soft
t3 = bool((drift[slide] > 0.05).all() and (drift[hold] < 0.02).all()) and slide.sum() > 0 and hold.sum() > 0
print(
    f"3. tan20 0.364: all-feet μ_s<0.30 envs {slide.sum().item()}, sliding {(drift[slide] > 0.05).sum().item()};"
    f" all-feet μ_d>0.55 envs {hold.sum().item()}, holding {(drift[hold] < 0.02).sum().item()}"
    f" -> {'PASS' if t3 else 'FAIL'}"
)
for i in (hold & (drift > 0.02)).nonzero().flatten()[:6].tolist():
    print(
        f"   failed hold env {i}: drift {drift[i]:.3f}, μ_d start {cur0[i, :, 1].tolist()} end {cur[i, :, 1].tolist()},"
        f" log k {cur[i, :, 2].tolist()}"
    )
if soft.any():
    x1 = env.scene["robot"].data.root_pos_w.torch[:, 0].clone()
    run(env, 100)
    drift2 = (env.scene["robot"].data.root_pos_w.torch[:, 0] - x1).abs()
    print(
        f"   soft-footing envs (μ_d > 0.55, k < 2e3) {soft.sum().item()}: drift 0.5–2.5 s median"
        f" {drift[soft].median():.3f} max {drift[soft].max():.3f} m; 2.5–4.5 s median {drift2[soft].median():.3f} max"
        f" {drift2[soft].max():.3f} m (≈ 0 in the second window = elastic settling, not creep)"
    )
mixed = ~slide & ~hold & ~soft
print(f"   mixed-footing envs {mixed.sum().item()}: drift median {drift[mixed].median():.3f} m")
ok &= t3
env.close()
print("MATERIAL_FIELD_TEST", "PASS" if ok else "FAIL")
app.close()
