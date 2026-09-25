# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg-to-leg clearance per command (phase2-progress §2.1.38): capsule surface gap between same-side neighbouring
legs (`mdp.rewards.adjacent_leg_gaps`, tibia + femur capsules), min / p1 / median over envs × steps, the fraction of
steps below the 2 cm margin, and the coxa angles at the closest approach. Flat ground, S0b policies.

Usage: python scripts/tests/leg_gap_probe.py --headless tag=<ckpt> [tag=<ckpt> ...]
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("policies", nargs="+", help="tag=checkpoint pairs")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import ADJACENT_LEGS, adjacent_leg_gaps  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

N, T, SETTLE, MARGIN = args.num_envs, 400, 100, 0.02
PAIRS = ("RF-RM", "RM-RB", "LF-LM", "LM-LB")
CMDS = {
    "fwd0.10": (0.10, 0.0, 0.0),
    "fwd0.15": (0.15, 0.0, 0.0),
    "fwd0.20": (0.20, 0.0, 0.0),
    "turn+0.3": (0.0, 0.0, 0.3),
    "turn-0.3": (0.0, 0.0, -0.3),
    "walkturn": (0.10, 0.0, 0.2),
    "side+0.05": (0.10, 0.05, 0.0),
    "stand": (0.0, 0.0, 0.0),
}

for spec in args.policies:
    tag, path = spec.split("=")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    worst = 9.0
    for cname, cmd in CMDS.items():
        task = task_for(ck)
        cfg = parse_env_cfg(task, num_envs=N)
        configure(cfg, ck)
        eval_physx_buffers(cfg)
        cfg.curriculum = None
        cfg.v0_ramp_steps = (0, 1)
        pin_command(cfg, *cmd)
        cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
            num_rows=1,
            num_cols=2,
            difficulty_range=(0.0, 0.0),
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        env = gym.make(task, cfg=cfg).unwrapped
        pin_terrain_level(env, 0)
        load_env_state(env, ck)
        with torch.inference_mode():
            obs, _ = env.reset()
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
        pol.load_state_dict({k: v.to(env.device) for k, v in ck["actor_state_dict"].items()})
        pol.eval()
        robot = env.scene["robot"]
        coxa = [i for i, n in enumerate(robot.joint_names) if n.endswith("_coxa")]
        G, Q = [], []
        with torch.inference_mode():
            for t in range(T):
                obs, *_ = env.step(pol(TensorDict({"policy": obs["policy"]}, [N])))
                if t >= SETTLE:
                    G.append(adjacent_leg_gaps(env).cpu())
                    Q.append(robot.data.joint_pos.torch[:, coxa].cpu())
        G, Q = torch.stack(G), torch.stack(Q)  # (T', N, 4), (T', N, 6)
        cols = []
        for k, (a, b) in enumerate(ADJACENT_LEGS):
            g = G[..., k].flatten()
            i = g.argmin()
            qa, qb = (math.degrees(Q.flatten(0, 1)[i, j]) for j in (a, b))
            cols.append(
                f"{PAIRS[k]} min {g.min() * 100:+5.1f} p1 {g.quantile(0.01) * 100:+5.1f} med {g.median() * 100:5.1f} "
                f"<2cm {(g < MARGIN).float().mean() * 100:4.1f}% coxa {qa:+5.1f}/{qb:+5.1f}"
            )
        worst = min(worst, G.min().item())
        print(f"GAP {tag} {cname:9s} cm | " + " | ".join(cols), flush=True)
        env.close()
    print(f"GAP {tag} worst surface gap {worst * 100:+.1f} cm", flush=True)
app.close()
