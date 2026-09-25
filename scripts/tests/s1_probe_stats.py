# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed probe-target standardisation and E_p frame normaliser for S1+ (paradigm spec §3.3: "targets standardised
with fixed statistics from the S0 rollout"; phase2-progress §2.2.5).

Rolls π_S0 (widened, deterministic; ê gate shut, V_0 in) on Hexapod-S1-v0 with every env pinned to a uniformly drawn
training terrain level 0–14 (curriculum off) and the training command mix, fills the aux sequence buffer, computes
the statistics over ``--samples`` random valid rows and writes them to ``hexasim/rl/s1/probe_stats_pi_s0.pt`` (the
default ``AuxCfg.stats_path``) — one file for every S1 seed and arm.

Usage: python scripts/tests/s1_probe_stats.py --headless --ckpt <π_S0.pt> [--num_envs 2048] [--steps 300]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--samples", type=int, default=262144)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import load_env_state  # noqa: E402
from hexasim.rl.s0_ppo import widen_actor_state  # noqa: E402
from hexasim.rl.s1.aux import STAT_KEYS, STATS_DEFAULT  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.terrain import LEVELS_DESIGNED  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Hexapod-S1-v0", num_envs=args.num_envs)
cfg.seed = 7
cfg.curriculum = None
cfg.aux.buffer_len = args.steps + 8
cfg.aux.stats_path = ""  # compute, do not load
env = gym.make("Hexapod-S1-v0", cfg=cfg).unwrapped
terrain = env.scene.terrain
levels = torch.randint(0, LEVELS_DESIGNED, (env.num_envs,), device=env.device)
terrain.terrain_levels[:] = levels
terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
load_env_state(env, ck)
obs, _ = env.reset()
pol = MLPModel(
    TensorDict({"policy": obs["policy"]}, [env.num_envs]),
    {"actor": ["policy"]},
    "actor",
    18,
    [256, 128, 64],
    "elu",
    True,
    {"class_name": "GaussianDistribution", "init_std": 1.0},
).to(env.device)
dim = obs["policy"].shape[1]
pol.load_state_dict({k: v.to(env.device) for k, v in widen_actor_state(ck["actor_state_dict"], dim, 0.05).items()})
pol.eval()
with torch.inference_mode():
    for _ in range(args.steps):
        obs, *_ = env.step(pol(TensorDict({"policy": obs["policy"]}, [env.num_envs])))
env.aux.compute_stats(args.samples)
out = {k: getattr(env.aux, k).cpu().clone() for k in STAT_KEYS}
out["meta"] = {
    "ckpt": args.ckpt,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "samples": args.samples,
    "levels": f"uniform 0–{LEVELS_DESIGNED - 1}",
    "seed": cfg.seed,
}
torch.save(out, STATS_DEFAULT)
print(f"PROBE_STATS written to {STATS_DEFAULT}; level mean {levels.float().mean():.2f}")
for k in STAT_KEYS:
    v = out[k]
    print(f"PROBE_STATS {k:10s} min {v.min():.4f} max {v.max():.4f}")
env.close()
app.close()
