# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward economics of Hexapod-S0a-v0 (CLAUDE.md rule 2): is walking strictly better than standing and than falling?

Reference behaviours, deterministic, 256 envs × one 10 s episode:
  walk   the warm-started actor from --ckpt (frozen π_base while iteration < 300), ground-truth v_in (α = 0)
  stand  zero action (rest pose)
Falling is scored analytically: the return of an episode that terminates on its first step.

Per behaviour: per-term reward per step (dt-scaled, as PPO sees it) over live steps, fall rate, episode length,
toe swing apex, and the discounted return from reset (γ = 0.99). Candidate reward settings are tried with
--set, e.g. --set rewards.cpg_z_target.params.step_height=0.06 rewards.terminating.weight=-50

Usage: python scripts/tests/reward_economy.py --headless --ckpt <S0a checkpoint> [--set k=v ...]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--set", nargs="*", default=[], help="cfg overrides, dotted path=value")
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

GAMMA = 0.99
cfg = parse_env_cfg("Hexapod-S0a-v0", num_envs=args.num_envs)
cfg.seed = 0
cfg.v0_ramp_steps = (10**9, 10**9 + 1)  # α = 0, as at the start of S0a
for kv in args.set:
    path, val = kv.split("=")
    *parents, leaf = path.split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p) if not isinstance(obj, dict) else obj[p]
    if isinstance(obj, dict):
        obj[leaf] = float(val)
    else:
        setattr(obj, leaf, float(val))
env = gym.make("Hexapod-S0a-v0", cfg=cfg).unwrapped
env.v0_train = False
ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
td = TensorDict({"policy": torch.zeros(1, 114)}, [1])
actor = MLPModel(td, {"actor": ["policy"]}, "actor", 18, [256, 128, 64], "elu", True,
                 {"class_name": "GaussianDistribution", "init_std": 1.0}).to(env.device)
actor.load_state_dict(ck["actor_state_dict"])
actor.eval()
robot, sensor, rm = env.scene["robot"], env.scene["contact_forces"], env.reward_manager
tibia = [i for i, n in enumerate(robot.data.body_names) if "tibia" in n]
N, T, dt = env.num_envs, int(env.max_episode_length), env.step_dt
names = rm.active_terms


def run(behaviour):
    with torch.inference_mode():
        obs, _ = env.reset()
    alive = torch.ones(N, dtype=torch.bool, device=env.device)
    fell = torch.zeros_like(alive)
    per_term = torch.zeros(len(names), device=env.device)
    live_steps = 0.0
    ret = torch.zeros(N, device=env.device)
    peak = torch.zeros(N, 6, device=env.device)
    apex = []
    with torch.inference_mode():
        for t in range(T):
            a = actor(TensorDict({"policy": obs["policy"]}, [N])) if behaviour == "walk" else torch.zeros(N, 18, device=env.device)
            obs, rew, term, trunc, _ = env.step(a)
            ret += (GAMMA**t) * rew * alive  # rew is already dt-scaled; includes the terminating term
            live = alive & ~term & ~trunc
            per_term += (rm._step_reward * dt * live.unsqueeze(1)).sum(0)
            live_steps += live.sum().item()
            contact = torch.norm(sensor.data.net_forces_w.torch, dim=-1) > 1.0
            toe_z = toe_pos_w(robot, tibia)[..., 2]
            td_mask = contact & (peak > 0.005) & live.unsqueeze(1)
            apex.append(peak[td_mask])
            peak = torch.where(contact, torch.zeros_like(peak), torch.maximum(peak, toe_z))
            fell |= alive & term
            alive &= ~(term | trunc)
    apex = torch.cat(apex)
    return {
        "fall_rate": fell.float().mean().item(),
        "live_steps_per_env": live_steps / N,
        "reward_per_live_step": (per_term.sum() / live_steps).item(),
        "per_term_per_step": {n: round((v / live_steps).item(), 5) for n, v in zip(names, per_term)},
        "discounted_return": ret.mean().item(),
        "toe_apex_m": [round(apex.quantile(q).item(), 4) for q in (0.25, 0.5, 0.75)] if len(apex) else None,
    }


w_term = cfg.rewards.terminating.weight
print("SETTINGS", args.set)
print(f"FALL (terminate on step 0): discounted return ≈ {w_term * dt:.4f}")
for b in ("walk", "stand"):
    r = run(b)
    print(f"{b.upper()}: fall {r['fall_rate']:.3f}, live steps {r['live_steps_per_env']:.0f}, "
          f"reward/step {r['reward_per_live_step']:+.4f}, discounted return {r['discounted_return']:+.3f}, "
          f"toe apex q25/50/75 {r['toe_apex_m']}")
    print("   ", {k: v for k, v in r["per_term_per_step"].items() if abs(v) > 1e-5})
env.close()
app.close()
