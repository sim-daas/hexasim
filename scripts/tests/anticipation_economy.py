# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does the reward pay for anticipation? (phase2-progress §2.3, external S1 review item 3.)

Transition course (``mdp.materials.strip_course``): flat tiles (or --terrain at difficulty 1), spawn at the tile
centre with yaw 0, a low-μ strip (μ_s --mu, μ_d 0.88 μ_s,
rigid) 0.5 m long starting --start m ahead of spawn, μ 0.9 / 0.85 elsewhere. Command v_x = --cmd for the whole episode,
deterministic actor, one 10 s episode per env. Behaviours:
  walk      the policy as is: walks onto the strip at full speed
  cautious  scripted stride shortening: the actor sees v_x = --cmd × --scale (the clock and the rewards keep --cmd,
            so the stride shortens at the commanded cadence) while its base is within --lead m before the strip
            and on it
Per behaviour: falls, slip on the strip, v_x on the strip, and per-term reward summed over the episode (dt-scaled, as
PPO sees it) for the S1 reward and for the S2 reward (the same minus the four clock-phase terms). Anticipation is
learnable only if cautious ≥ walk under the reward the arms train on.

Usage: python scripts/tests/anticipation_economy.py --headless --ckpt <S1 checkpoint> [--cmd 0.15 --scale 0.73]
"""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--cmd", type=float, default=0.15)
parser.add_argument("--scale", type=float, default=0.73, help="cautious: commanded speed seen by the actor, × cmd")
parser.add_argument("--lead", type=float, default=0.3, help="cautious: m before the strip where shortening starts")
parser.add_argument("--mu", type=float, default=0.25, help="strip μ_s (μ_d = 0.88 μ_s), rigid")
parser.add_argument("--terrain", default="flat", help="flat or a mdp.terrain sub-terrain (difficulty 1)")
parser.add_argument("--start", type=float, default=0.5, help="strip start, m ahead of spawn (slopes begin at 0.5 m)")
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import layout, pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.materials import strip_course  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.terrain import EVAL_TERRAIN_TYPES  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

START, LENGTH = args.start, 0.5
CLOCK_TERMS = ("cpg_z_target", "cpg_stance_rigidity", "cpg_force_target", "cpg_coxa_contact")
N = args.num_envs
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = parse_env_cfg("Hexapod-S1-v0", num_envs=N)
configure(cfg, ck)
eval_physx_buffers(cfg)
cfg.seed = 0
cfg.curriculum = None
pin_command(cfg, args.cmd)
cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
    num_rows=1,
    num_cols=2,
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        args.terrain: MeshPlaneTerrainCfg(proportion=1.0)
        if args.terrain == "flat"
        else EVAL_TERRAIN_TYPES[args.terrain].replace(proportion=1.0)
    },
)
cfg.events.reset_robot_base.params["pose_range"].update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
env = gym.make("Hexapod-S1-v0", cfg=cfg).unwrapped
pin_terrain_level(env, 0)
load_env_state(env, ck)
strip_course(env, START, LENGTH, low=(args.mu, 0.88 * args.mu, 6.0))
dim = env.observation_manager.group_obs_dim["policy"][0]
pol = MLPModel(
    TensorDict({"policy": torch.zeros(1, dim)}, [1]),
    {"actor": ["policy"]},
    "actor",
    18,
    [256, 128, 64],
    "elu",
    True,
    {"class_name": "GaussianDistribution", "init_std": 1.0},
).to(env.device)
pol.load_state_dict({k: v.to(env.device) for k, v in ck["actor_state_dict"].items()})
pol.eval()
robot, sensor, rm = env.scene["robot"], env.scene["contact_forces"], env.reward_manager
names = rm.active_terms
clock_idx = [names.index(n) for n in CLOCK_TERMS if n in names]
T, cmd_col = int(env.max_episode_length), layout.ACTOR["twist_cmd"].start


def run(behaviour):
    with torch.inference_mode():
        obs, _ = env.reset()
        alive = torch.ones(N, dtype=torch.bool, device=env.device)
        fell = torch.zeros_like(alive)
        per_term = torch.zeros(len(names), device=env.device)
        slip, vx_on, n_on = 0.0, 0.0, 0.0
        prev_toe = None
        for _ in range(T):
            o = obs["policy"].clone()
            s = robot.data.root_pos_w.torch[:, 0] - env.spawn_xy[:, 0]
            on = (s >= START) & (s < START + LENGTH)
            if behaviour == "cautious":
                near = (s >= START - args.lead) & (s < START + LENGTH)
                o[near, cmd_col] = args.cmd * args.scale
            obs, _, term, trunc, _ = env.step(pol(TensorDict({"policy": o}, [N])))
            live = alive & ~term & ~trunc
            per_term += (rm._step_reward * env.step_dt * live.unsqueeze(1)).sum(0)
            toe = toe_pos_w(robot, env._tibia_body_ids)
            contact = sensor.data.net_forces_w.torch.norm(dim=-1) > 1.0
            if prev_toe is not None:
                v = ((toe - prev_toe)[..., :2].norm(dim=-1) / env.step_dt * contact).sum(1)
                v = v / contact.sum(1).clamp(min=1)
                slip += (v * on * live).sum().item()
            vx_on += (robot.data.root_lin_vel_b.torch[:, 0] * on * live).sum().item()
            n_on += (on & live).sum().item()
            prev_toe = toe
            fell |= alive & term
            alive &= ~(term | trunc)
    per_term = per_term / N
    s1 = per_term.sum().item()
    s2 = s1 - per_term[clock_idx].sum().item()
    print(
        f"ANTICIP {behaviour:8s} {args.terrain} mu {args.mu:.2f} cmd {args.cmd:.2f} scale {args.scale:.2f}:"
        f" fall {fell.float().mean():.3f} |"
        f" on strip v_x {vx_on / max(n_on, 1):.3f} slip {slip / max(n_on, 1):.3f} m/s | episode reward S1 {s1:+.3f}"
        f" S2 {s2:+.3f}"
    )
    print("    ", {n: round(v, 4) for n, v in zip(names, per_term.tolist()) if abs(v) > 1e-4})
    return per_term


w, c = run("walk"), run("cautious")
d = c - w
print("ANTICIP cautious − walk per term:", {n: round(v, 4) for n, v in zip(names, d.tolist()) if abs(v) > 1e-4})
print(f"ANTICIP cautious − walk: S1 {d.sum():+.4f}  S2 {d.sum() - d[clock_idx].sum():+.4f}  (> 0 ⇒ reward pays for it)")
env.close()
app.close()
