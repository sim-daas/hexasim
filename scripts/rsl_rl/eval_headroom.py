# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One condition of the O vs B0 headroom evaluation (phase2-progress §2.3.7; primary metrics D38).

The task, the arm and the clock dropout follow the checkpoint (``hexasim.rl.load_policy``). Deterministic actor, one
episode per env, fixed command, fixed seed (terrain geometry and spawns are then the same for every arm).

Two kinds of course:
  material  ``--strip MU_S MU_D LOG10K``: 15° ``--terrain`` slope_up / slope_down (or flat), spawn at the tile centre
            with yaw 0, the whole ground ``--base`` except one strip across the path from ``--start`` m ahead of the
            spawn, ``--length`` m long (``mdp.materials.strip_course``). Window = from 0.5 m before the strip to its
            end. ``--start -100 --length 1000`` makes the whole ground the strip material (soft-ground V tests).
  shape     no ``--strip``: ``--terrain`` bumps / stairs_up / stairs_down / rough at ``--level`` (0–14 training, 15–20
            held out), training spawns, material field with ``--layout_seed``. Window = the whole episode.

Per-episode metrics (JSON lists, one value per env, for the episode bootstrap; NaN = no sample):
  fall        terminated
  vx_err      |mean v_x in the window − command|, m/s
  slip        mean horizontal toe speed of feet in contact in the window, m/s
  impact      mean over touchdowns of the peak |F| in the first 40 ms of contact ÷ the static tripod leg load (L1 on
              the force norm, never squared; no world-Z / surface-normal question on slopes)
  td_speed    mean toe speed at touchdown, m/s
  stumble     share of touchdowns whose swing lasted < 60 % of the run's median swing duration
  sink        mean depth of the toes in contact below the terrain surface in the window, m
  height      mean base height above the terrain under it in the window, m
  pitch       mean base pitch in the window, rad (gravity x in the body frame)
  stride      descriptive: mean v_x ÷ per-leg touchdown rate in the window, m
plus ``vtest_*`` (v7 §4.3 V tests on this rollout).

Usage: python scripts/rsl_rl/eval_headroom.py --headless --ckpt <ckpt> --terrain slope_down --strip 0.25 0.22 6 \
           --out logs/eval/<tag>.json
"""

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--cmd", type=float, default=0.15)
parser.add_argument("--terrain", default="flat", help="flat or a mdp.terrain sub-terrain")
parser.add_argument("--level", type=int, default=9, help="shape course: terrain level")
parser.add_argument("--strip", type=float, nargs=3, default=None, metavar=("MU_S", "MU_D", "LOG10K"))
parser.add_argument("--base", type=float, nargs=3, default=(0.9, 0.85, 6.0), metavar=("MU_S", "MU_D", "LOG10K"))
parser.add_argument("--start", type=float, default=0.5, help="strip start, m ahead of spawn (slopes begin at 0.5 m)")
parser.add_argument("--length", type=float, default=0.5)
parser.add_argument("--layout_seed", type=int, default=1, help="material-field layout (training = 0)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument(
    "--ablate_oracle",
    action="store_true",
    help="ablation: the oracle block (M* / H*) reads its normaliser mean (z = 0: no information, in distribution)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.materials import strip_course  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.terrain import EVAL_TERRAIN_TYPES, single_type_generator  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

N = args.num_envs
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
task = task_for(ck)
cfg = parse_env_cfg(task, num_envs=N)
configure(cfg, ck)
eval_physx_buffers(cfg)
cfg.seed = args.seed
cfg.curriculum = None
cfg.events.terrain_materials.params["seed"] = args.layout_seed
pin_command(cfg, args.cmd)
if args.strip is not None:
    sub = MeshPlaneTerrainCfg() if args.terrain == "flat" else EVAL_TERRAIN_TYPES[args.terrain]
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        num_rows=1, num_cols=2, difficulty_range=(1.0, 1.0), sub_terrains={args.terrain: sub.replace(proportion=1.0)}
    )
    cfg.events.reset_robot_base.params["pose_range"].update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
else:
    cfg.scene.terrain.terrain_generator = single_type_generator(args.terrain)
env = gym.make(task, cfg=cfg).unwrapped
pin_terrain_level(env, 0 if args.strip is not None else args.level)
load_env_state(env, ck)
if args.strip is not None:
    strip_course(env, args.start, args.length, low=tuple(args.strip), high=tuple(args.base))
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
oracle = None
if args.ablate_oracle:  # zero would be far out of distribution for M* (μ ≈ 0.7, log k ≈ 5.4): use the mean instead
    from hexasim.tasks.manager_based.hexasim.mdp import layout  # noqa: PLC0415

    oracle = layout.S2_ACTOR[layout.ARM_BLOCK[cfg.arm]]
    oracle_mean = pol.obs_normalizer._mean[0, oracle].clone()
robot, sensor = env.scene["robot"], env.scene["contact_forces"]
dev, dt, T = env.device, env.step_dt, int(env.max_episode_length)
zeros = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
acc = {k: zeros(N) for k in ("n", "vx", "slip", "sink", "n_contact", "height", "pitch", "imp", "td", "td_speed")}
alive = torch.ones(N, dtype=torch.bool, device=dev)
fell = torch.zeros_like(alive)
air = zeros(N, 6)
pending = torch.zeros(N, 6, dtype=torch.bool, device=dev)  # touched down last step: this step ends the 40 ms window
peak = zeros(N, 6)
swings = []  # (N, 6) swing duration in steps at each touchdown, 0 elsewhere
prev_toe, prev_contact = None, torch.ones(N, 6, dtype=torch.bool, device=dev)
with torch.inference_mode():
    obs, _ = env.reset()
    for _ in range(T):
        o = obs["policy"]
        if oracle is not None:
            o = o.clone()
            o[:, oracle] = oracle_mean
        obs, _, term, trunc, _ = env.step(pol(TensorDict({"policy": o}, [N])))
        live = alive & ~term & ~trunc
        root = robot.data.root_pos_w.torch
        s = root[:, 0] - env.spawn_xy[:, 0]
        win = live & ((s >= args.start - 0.5) & (s < args.start + args.length) if args.strip is not None else True)
        toe = toe_pos_w(robot, env._tibia_body_ids)
        force = sensor.data.net_forces_w.torch.norm(dim=-1)
        contact = force > 1.0
        h = toe[..., 2] - env.terrain_z(toe[..., :2])
        toe_v = zeros(N, 6, 3) if prev_toe is None else (toe - prev_toe) / dt
        w = win.float()
        acc["n"] += w
        acc["vx"] += robot.data.root_lin_vel_b.torch[:, 0] * w
        acc["slip"] += (toe_v[..., :2].norm(dim=-1) * contact).sum(1) * w
        acc["sink"] += (-h * contact).sum(1) * w
        acc["n_contact"] += contact.sum(1) * w
        acc["height"] += (root[:, 2] - env.terrain_z(root[:, :2])) * w
        acc["pitch"] += torch.asin(robot.data.projected_gravity_b.torch[:, 0].clamp(-1, 1)) * w
        # Touchdowns (after ≥ 2 airborne steps: shorter gaps are contact chatter), first-contact impact over 2 steps.
        done = pending & win[:, None]
        acc["imp"] += (torch.maximum(peak, force) * done).sum(1) / env.tripod_leg_load[:, 0]
        acc["td"] += done.sum(1)
        td = contact & ~prev_contact & (air >= 2) & win[:, None]
        acc["td_speed"] += (toe_v.norm(dim=-1) * td).sum(1)
        swings.append(air * td)
        peak, pending = force * td, td
        air = torch.where(contact, 0.0, air + 1)
        prev_toe, prev_contact = toe, contact
        fell |= alive & term
        alive &= ~(term | trunc)
    vtests = env.aux.v_tests() if env.aux.stats_frozen else {}

swings = torch.stack(swings)  # (T, N, 6)
real = swings[swings >= 5]
median = real.median().item() if len(real) else float("nan")
stumbles = ((swings > 0) & (swings < 0.6 * median)).sum((0, 2))
nan = float("nan")
div = lambda a, b: torch.where(b > 0, a / b.clamp(min=1), nan)  # noqa: E731
vx = div(acc["vx"], acc["n"])
rate = acc["td"] / 6 / (acc["n"] * dt).clamp(min=dt)  # touchdowns per leg per s
ep = {
    "fall": fell.float(),
    "vx_err": (vx - args.cmd).abs(),
    "slip": div(acc["slip"], acc["n_contact"]),
    "impact": div(acc["imp"], acc["td"]),
    "td_speed": div(acc["td_speed"], acc["td"]),
    "stumble": div(stumbles.float(), acc["td"]),
    "sink": div(acc["sink"], acc["n_contact"]),
    "height": div(acc["height"], acc["n"]),
    "pitch": div(acc["pitch"], acc["n"]),
    "stride": div(vx, rate),
}
m = {
    "ckpt": args.ckpt,
    "task": task,
    "arm": cfg.arm,
    "ablate_oracle": args.ablate_oracle,
    "clock_drop_p": env.clock_drop_p,
    "cmd": args.cmd,
    "terrain": args.terrain,
    "level": args.level,
    "strip": args.strip,
    "base": list(args.base),
    "start": args.start,
    "length": args.length,
    "layout_seed": args.layout_seed,
    "seed": args.seed,
    "num_envs": N,
    "median_swing_s": median * dt,
}
m.update({f"vtest_{k}": v for k, v in vtests.items()})
m.update({k: v.nanmean().item() for k, v in ep.items()})
m["episodes"] = {k: v.tolist() for k, v in ep.items()}
print("HEADROOM", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "episodes"})
with open(args.out, "w") as f:
    json.dump(m, f)
env.close()
app.close()
