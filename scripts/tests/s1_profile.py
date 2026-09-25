# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-component timing of one Hexapod-S1-v0 control step (collection cost; phase2-progress §2.2.5).

Wraps the S1 additions with CUDA-synchronised timers and reports ms per env step, next to a plain Hexapod-S0b-v0
step at the same env count. Actions are small random joint targets.

Usage: python scripts/tests/s1_profile.py --headless [--num_envs 4096] [--steps 60] [--task Hexapod-S1-v0]
"""

import argparse
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--task", default="Hexapod-S1-v0")
parser.add_argument("--no_writer", action="store_true")
parser.add_argument("--rigid_half", action="store_true", help="S1 with 50 %% rigid foot buckets (as S0b)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.tasks.manager_based.hexasim import hexapod_env as he  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import materials, observations, symmetry  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

T = defaultdict(float)
active = [True]


def timed(name, fn):
    def w(*a, **k):
        if not active[0]:
            return fn(*a, **k)
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t
        return r

    return w


he.HexapodS0bEnv.step = timed("S0b step (physics + managers)", he.HexapodS0bEnv.step)
he.HexapodS0aEnv.refresh_proprio = timed("S0a refresh_proprio (o_p, V_0)", he.HexapodS0aEnv.refresh_proprio)
if args.task == "Hexapod-S1-v0":
    he.HexapodS1Env.refresh_proprio = timed("S1 refresh_proprio total", he.HexapodS1Env.refresh_proprio)
    he.HexapodS1Env.material4 = timed("  material4 (strips + toes)", he.HexapodS1Env.material4)
    he.HexapodS1Env._local_heights = timed("  local heights", he.HexapodS1Env._local_heights)
    symmetry.mirror_frame = timed("  mirror_frame", symmetry.mirror_frame)
    observations.s0a_privileged_obs = timed("  s0a_privileged_obs", observations.s0a_privileged_obs)
    materials.FootWriter.update = timed("foot_writer.update", materials.FootWriter.update)
    if args.no_writer:
        materials.FootWriter.update = lambda *a, **k: None
cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
if args.rigid_half:
    cfg.events.foot_material.params["rigid_fraction"] = 0.5
env = gym.make(args.task, cfg=cfg).unwrapped
if args.task == "Hexapod-S1-v0":
    env.aux.latents = timed("  aux.latents (×2)", env.aux.latents)
    env.aux.buffer.add = timed("  buffer.add", env.aux.buffer.add)
    env.aux.buffer.touchdown = timed("  buffer.touchdown", env.aux.buffer.touchdown)
env.reset()
g = torch.Generator(device=env.device).manual_seed(0)
with torch.inference_mode():
    for _ in range(10):
        env.step(0.3 * torch.randn(env.num_envs, 18, device=env.device, generator=g))
    T.clear()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(0.3 * torch.randn(env.num_envs, 18, device=env.device, generator=g))
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / args.steps
print(
    f"PROFILE {args.task} no_writer={args.no_writer} rigid_half={args.rigid_half} {args.num_envs} envs:"
    f" env.step {total * 1e3:.1f} ms"
)
for k, v in T.items():
    print(f"PROFILE   {k:36s} {v / args.steps * 1e3:7.2f} ms")
if args.task == "Hexapod-S1-v0":
    active[0] = False
    b = env.aux.buffer.sample(env.aux.cfg.batch, env.step_dt)
    for _ in range(2):
        env.aux.update()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3):
        env.aux.compute_stats() if not env.aux.stats_frozen else None
        env.aux.update()
    torch.cuda.synchronize()
    print(f"PROFILE aux.update per PPO iteration {(time.perf_counter() - t) / 3 * 1e3:.1f} ms")
env.close()
app.close()
