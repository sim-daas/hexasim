# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward economics (CLAUDE.md rule 2): is walking strictly better than standing and than falling?

Reference behaviours, deterministic, 256 envs × one 10 s episode:
  walk   the actor from --ckpt (S0a: ground-truth v_in, α = 0; S0b: V_0, actor widened by zero columns if it
         predates the task's layout, so π_S0a walks at its own speed whatever the command)
  stand  zero action (rest pose)
Falling is scored analytically: the return of an episode that terminates on its first step.
S0b: --cmd pins twist_cmd (all-zero = standing mode), --terrain_level pins the terrain level.

Per behaviour: per-term reward per step (dt-scaled, as PPO sees it) over live steps, fall rate, episode length,
toe swing apex, and the discounted return from reset (γ = 0.99). Candidate reward settings are tried with
--set, e.g. --set rewards.cpg_z_target.params.step_height=0.06 rewards.terminating.weight=-50

Usage: python scripts/tests/reward_economy.py --headless --ckpt <S0a checkpoint> [--set k=v ...]
"""

import argparse
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--task", default="Hexapod-S0a-v0")
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
parser.add_argument("--terrain_level", type=int, default=None)
parser.add_argument(
    "--pin_joints", default=None, help="diagnosis: extra behaviour 'walk_pinned' = actor, these joints (regex) at rest"
)
parser.add_argument(
    "--obs_cmd",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="S0b diagnosis: the actor sees this twist_cmd while rewards and clock use --cmd (drives it at other speeds)",
)
parser.add_argument("--set", nargs="*", default=[], help="cfg overrides, dotted path=value")
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.s0_ppo import widen_actor_state  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

GAMMA = 0.99
cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
cfg.seed = 0
s0b = args.task.startswith(("Hexapod-S0b", "Hexapod-S1", "Hexapod-S2"))  # command tasks (S0b and later)
if s0b:
    cfg.curriculum = None
    if args.cmd is not None:
        pin_command(cfg, *args.cmd)
else:
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
env = gym.make(args.task, cfg=cfg).unwrapped
if args.terrain_level is not None:
    pin_terrain_level(env, args.terrain_level)
env.v0_train = False
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
dim = env.observation_manager.group_obs_dim["policy"][0]
state = ck["actor_state_dict"]
if state["mlp.0.weight"].shape[1] < dim:
    state = widen_actor_state(state, dim, 0.05)
td = TensorDict({"policy": torch.zeros(1, dim)}, [1])
actor = MLPModel(
    td,
    {"actor": ["policy"]},
    "actor",
    18,
    [256, 128, 64],
    "elu",
    True,
    {"class_name": "GaussianDistribution", "init_std": 1.0},
).to(env.device)
actor.load_state_dict({k: v.to(env.device) for k, v in state.items()})
if s0b:  # V_0 and, from S1, the aux learner, ê gate and dropout state (the trunk does not walk without them)
    from hexasim.rl.load_policy import load_env_state

    load_env_state(env, ck)
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
    vx = torch.zeros(N, device=env.device)
    with torch.inference_mode():
        for t in range(T):
            o = obs["policy"].clone()
            if args.obs_cmd is not None:
                o[:, -3:] = torch.tensor(args.obs_cmd, device=env.device)
            a = (
                actor(TensorDict({"policy": o}, [N]))
                if behaviour.startswith("walk")
                else torch.zeros(N, 18, device=env.device)
            )
            if behaviour == "walk_pinned":
                a[:, pinned] = 0.0
            obs, rew, term, trunc, _ = env.step(a)
            ret += (GAMMA**t) * rew * alive  # rew is already dt-scaled; includes the terminating term
            live = alive & ~term & ~trunc
            per_term += (rm._step_reward * dt * live.unsqueeze(1)).sum(0)
            live_steps += live.sum().item()
            vx += robot.data.root_lin_vel_b.torch[:, 0] * live
            contact = torch.norm(sensor.data.net_forces_w.torch, dim=-1) > 1.0
            toe = toe_pos_w(robot, tibia)
            toe_z = toe[..., 2] - env.terrain_z(toe[..., :2])
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
        "vx": (vx.sum() / live_steps).item(),
        "toe_apex_m": [round(apex.quantile(q).item(), 4) for q in (0.25, 0.5, 0.75)] if len(apex) else None,
    }


w_term = cfg.rewards.terminating.weight
print("SETTINGS", args.task, "cmd", args.cmd, "obs_cmd", args.obs_cmd, "terrain_level", args.terrain_level, args.set)
print(f"FALL (terminate on step 0): discounted return ≈ {w_term * dt:.4f}")
pinned = [i for i, n in enumerate(robot.joint_names) if args.pin_joints and re.match(args.pin_joints, n)]
behaviours = ("walk",) if args.obs_cmd is not None else ("walk", "stand")
for b in behaviours + (("walk_pinned",) if pinned else ()):
    r = run(b)
    print(
        f"{b.upper()}: fall {r['fall_rate']:.3f}, live steps {r['live_steps_per_env']:.0f}, vx {r['vx']:.3f}, "
        f"reward/step {r['reward_per_live_step']:+.4f}, discounted return {r['discounted_return']:+.3f}, "
        f"toe apex q25/50/75 {r['toe_apex_m']}"
    )
    print("   ", {k: v for k, v in r["per_term_per_step"].items() if abs(v) > 1e-5})
env.close()
app.close()
