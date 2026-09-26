# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait metrics split by ground friction (phase2-progress §2.2.6: swing-lift drift and leg clearance in the S1 pilot).

Hexapod-S1-v0, flat tiles, fixed walking command, deterministic actions. Every env's palette is overwritten after the
reset: even envs all-slippery (μ_s 0.25, μ_d 0.22), odd envs all-grippy (μ_s 0.9, μ_d 0.85), all rigid (k 1e6), so the
only difference between the groups is friction. Per policy and group, over T steps after a settle:
  lift      mean peak toe height above the terrain per swing (≥ 5 airborne steps), front / middle / back legs, cm
  coxa      coxa angle range (max − min over the window), front / middle / back, deg
  gap       worst and 1st-percentile same-side leg capsule surface gap, and share of steps < 2 cm
  clip      Σ |requested − clipped joint target| (rad, the action_beyond_clip term), mean per step
  slip      mean horizontal toe speed of feet in contact, m/s
  v_x       mean base forward speed, m/s

Usage: python scripts/tests/gait_by_material.py --headless --cmd 0.15 pi_s0=<ckpt> s1=<ckpt> [--num_envs 512]
"""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("policies", nargs="+", help="tag=checkpoint")
parser.add_argument("--cmd", type=float, default=0.15)
parser.add_argument("--num_envs", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state  # noqa: E402
from hexasim.rl.s0_ppo import widen_actor_state  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import (  # noqa: E402
    action_beyond_clip_penalty,
    adjacent_leg_gaps,
    coxa_leg_geometry,
    toe_pos_w,
)
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

N, SETTLE, T = args.num_envs, 100, 400
GROUPS = {"slippery": (0.25, 0.22), "grippy": (0.9, 0.85)}
LEG_TYPES = {"front": [0, 3], "middle": [1, 4], "back": [2, 5]}


def run(tag, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = parse_env_cfg("Hexapod-S1-v0", num_envs=N)
    configure(cfg, ck)
    eval_physx_buffers(cfg)
    cfg.curriculum = None
    pin_command(cfg, args.cmd)
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        num_rows=1, num_cols=2, difficulty_range=(0.0, 0.0), sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)}
    )
    env = gym.make("Hexapod-S1-v0", cfg=cfg).unwrapped
    pin_terrain_level(env, 0)
    load_env_state(env, ck)
    obs, _ = env.reset()
    even = torch.arange(N, device=env.device) % 2 == 0
    for name, (mu_s, mu_d) in GROUPS.items():
        m = even if name == "slippery" else ~even
        env.palette.v[m] = torch.tensor([mu_s, mu_d, 6.0], device=env.device)
    pol = MLPModel(
        TensorDict({"policy": obs["policy"]}, [N]),
        {"actor": ["policy"]},
        "actor",
        18,
        [256, 128, 64],
        "elu",
        True,
        {"class_name": "GaussianDistribution", "init_std": 1.0},
    ).to(env.device)
    state = ck["actor_state_dict"]
    if state["mlp.0.weight"].shape[1] < obs["policy"].shape[1]:
        state = widen_actor_state({k: v.cpu() for k, v in state.items()}, obs["policy"].shape[1], 0.05)
    pol.load_state_dict({k: v.to(env.device) for k, v in state.items()})
    pol.eval()
    robot, sensor = env.scene["robot"], env.scene["contact_forces"]
    coxa = [i for i, n in enumerate(robot.joint_names) if n.endswith("_coxa")]
    peak = torch.zeros(N, 6, device=env.device)
    air = torch.zeros(N, 6, device=env.device)
    lift_sum = torch.zeros(N, 6, device=env.device)
    lift_n = torch.zeros(N, 6, device=env.device)
    cmin = torch.full((N, 6), 9.0, device=env.device)
    cmax = torch.full((N, 6), -9.0, device=env.device)
    gaps, clip, slip, vx = [], [], [], []
    rm = env.reward_manager
    terms = torch.zeros(len(rm.active_terms), device=env.device)
    land_r_sum = torch.zeros(N, 6, device=env.device)
    prev_toe = None
    with torch.inference_mode():
        for t in range(SETTLE + T):
            obs, *_ = env.step(pol(TensorDict({"policy": obs["policy"]}, [N])))
            toe = toe_pos_w(robot, env._tibia_body_ids)
            h = toe[..., 2] - env.terrain_z(toe[..., :2])
            contact = sensor.data.net_forces_w.torch.norm(dim=-1) > 1.0
            if t >= SETTLE:
                landed = contact & (air >= 5)
                land_r_sum += torch.where(landed, coxa_leg_geometry(env)[1][..., :2].norm(dim=-1), 0.0)
                terms += rm._step_reward.mean(0)
                lift_sum += torch.where(landed, peak, 0.0)
                lift_n += landed.float()
                q = robot.data.joint_pos.torch[:, coxa]
                cmin, cmax = torch.minimum(cmin, q), torch.maximum(cmax, q)
                gaps.append(adjacent_leg_gaps(env))
                clip.append(action_beyond_clip_penalty(env, "joint_pos"))
                if prev_toe is not None:
                    v = ((toe - prev_toe)[..., :2].norm(dim=-1) / env.step_dt) * contact
                    slip.append(v.sum(1) / contact.sum(1).clamp(min=1))
                vx.append(robot.data.root_lin_vel_b.torch[:, 0])
            peak = torch.where(contact, 0.0, torch.maximum(peak, h))
            air = torch.where(contact, 0.0, air + 1)
            prev_toe = toe
    gaps, clip, slip, vx = torch.stack(gaps), torch.stack(clip), torch.stack(slip), torch.stack(vx)
    lift = lift_sum / lift_n.clamp(min=1)
    crange = torch.rad2deg(cmax - cmin)
    for name in GROUPS:
        m = even if name == "slippery" else ~even
        lf = " ".join(f"{lift[m][:, i].mean() * 100:4.2f}" for i in LEG_TYPES.values())
        cr = " ".join(f"{crange[m][:, i].mean():5.1f}" for i in LEG_TYPES.values())
        g = gaps[:, m]
        print(
            f"MATGAIT {tag:8s} v{args.cmd:.2f} {name:8s} lift f/m/b {lf} cm | coxa range f/m/b {cr} deg |"
            f" gap min {g.min() * 100:+.1f} p1 {g.flatten().quantile(0.01) * 100:+.1f} cm <2cm"
            f" {(g < 0.02).float().mean() * 100:.1f}% | clip {clip[:, m].mean():.4f} | slip {slip[:, m].mean():.3f}"
            f" | v_x {vx[:, m].mean():.3f}"
        )
    land_r = land_r_sum / lift_n.clamp(min=1)
    lr = " ".join(f"{land_r[:, i].mean() * 100:4.1f}" for i in LEG_TYPES.values())
    print(f"MATGAIT {tag:8s} landing |r| f/m/b {lr} cm (rest reach {env.coxa_rest_reach[0] * 100:.1f} cm)")
    print(
        f"MATGAIT {tag:8s} terms (weighted, per s): "
        + " ".join(f"{n}={v / T:.3f}" for n, v in zip(rm.active_terms, terms.tolist()) if abs(v / T) > 1e-3)
    )
    env.close()


for spec in args.policies:
    tag, path = spec.split("=")
    run(tag, path)
app.close()
