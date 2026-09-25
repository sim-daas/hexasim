# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decompose each swing's toe motion relative to its hip (phase2-progress §2.1.36).

For every swing (foot off the ground ≥ 5 control steps) of every leg, in the body frame relative to that leg's coxa
axis: radial distance |r_xy| (in/out from the hip), sweep (tangential arc |r|·Δθ), lift (toe height above the
terrain) and the tibia joint angle. Prints per leg the radial excursion and path, lift and sweep per swing, and for
the front pair and the middle+back four the mean profile over 10 normalised swing-phase bins (lift-off → touchdown)
of Δ|r_xy| and Δtibia relative to lift-off. An in-then-out radial profile tied to the tibia is the "shin slants more,
then less just before landing" the user sees.

Flat ground, forward command 0.15 m/s (S0b) — S0a policies have no command and walk at their own speed.

Usage: python scripts/tests/swing_radial_probe.py --headless tag=<ckpt> [tag=<ckpt> ...]
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("policies", nargs="+", help="tag=checkpoint pairs")
parser.add_argument("--num_envs", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import coxa_leg_geometry, toe_pos_w  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

N, T, SETTLE, BINS, MIN_AIR = args.num_envs, 400, 100, 10, 5
LEGS = ("RF", "RM", "RB", "LF", "LM", "LB")

for spec in args.policies:
    tag, path = spec.split("=")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    task = task_for(ck)
    cmd_task = task != "Hexapod-S0a-v0"
    cfg = parse_env_cfg(task, num_envs=N)
    configure(cfg, ck)
    eval_physx_buffers(cfg)
    cfg.curriculum = None
    cfg.v0_ramp_steps = (0, 1)
    if cmd_task:
        pin_command(cfg, 0.15)
        cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
            num_rows=1,
            num_cols=2,
            difficulty_range=(0.0, 0.0),
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
    env = gym.make(task, cfg=cfg).unwrapped
    if cmd_task:
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
    robot, sensor = env.scene["robot"], env.scene["contact_forces"]
    tib = [i for i, n in enumerate(robot.joint_names) if n.endswith("_tibia")]
    R, A, Z, Q, C = [], [], [], [], []
    with torch.inference_mode():
        for _ in range(T):
            obs, *_ = env.step(pol(TensorDict({"policy": obs["policy"]}, [N])))
            _, r = coxa_leg_geometry(env)
            toe = toe_pos_w(robot, env._tibia_body_ids)
            R.append(r[..., :2].norm(dim=-1).cpu())
            A.append(torch.atan2(r[..., 1], r[..., 0]).cpu())
            Z.append((toe[..., 2] - env.terrain_z(toe[..., :2])).cpu())
            Q.append(robot.data.joint_pos.torch[:, tib].cpu())
            C.append((torch.norm(sensor.data.net_forces_w.torch, dim=-1) > 1.0).cpu())
    R, A, Z, Q, C = (torch.stack(x)[SETTLE:] for x in (R, A, Z, Q, C))  # (T', N, 6)

    stats = {k: [[] for _ in range(6)] for k in ("rad_exc", "rad_path", "lift", "sweep", "tib_exc")}
    prof = {g: {"dr": torch.zeros(BINS), "dq": torch.zeros(BINS), "n": 0} for g in ("front", "mid+back")}
    for e in range(N):
        for leg in range(6):
            air = ~C[:, e, leg]
            t = 0
            while t < len(air):
                if not air[t]:
                    t += 1
                    continue
                s = t
                while t < len(air) and air[t]:
                    t += 1
                if t - s < MIN_AIR or s == 0 or t == len(air):
                    continue
                r, a, z, q = R[s:t, e, leg], A[s:t, e, leg], Z[s:t, e, leg], Q[s:t, e, leg]
                stats["rad_exc"][leg].append((r.max() - r.min()).item())
                stats["rad_path"][leg].append((r[1:] - r[:-1]).abs().sum().item())
                stats["lift"][leg].append((z.max() - z[0]).item())
                da = torch.atan2(torch.sin(a[-1] - a[0]), torch.cos(a[-1] - a[0]))
                stats["sweep"][leg].append((r.mean() * da.abs()).item())
                stats["tib_exc"][leg].append((q.max() - q.min()).item())
                idx = torch.clamp((torch.arange(BINS) * (t - s)) // BINS, max=t - s - 1)
                g = prof["front" if leg in (0, 3) else "mid+back"]
                g["dr"] += r[idx] - r[0]
                g["dq"] += q[idx] - q[0]
                g["n"] += 1

    mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
    print(f"\nRADIAL {tag} ({task}); per swing, cm / deg, legs {' '.join(LEGS)}")
    for k, unit in (("rad_exc", 100), ("rad_path", 100), ("lift", 100), ("sweep", 100), ("tib_exc", 180 / math.pi)):
        print(f"  {k:9s}", " ".join(f"{mean(stats[k][leg]) * unit:6.2f}" for leg in range(6)))
    for gname, g in prof.items():
        n = max(g["n"], 1)
        print(f"  {gname:8s} Δ|r| mm  ", " ".join(f"{x * 1000 / n:+6.1f}" for x in g["dr"]))
        print(f"  {gname:8s} Δtibia °  ", " ".join(f"{x * 180 / math.pi / n:+6.1f}" for x in g["dq"]))
    env.close()
app.close()
