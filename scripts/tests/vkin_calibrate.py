# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim calibration of the v_kin stance-weight constants τ_e, s, δ (paradigm spec §2.1; deviation D26).

Rolls π_S0 (widened, deterministic) on Hexapod-S1-v0 with the training command mix, collects the deployable per-step
frames and ground truth from the aux sequence buffer, and grid-searches (τ_e, s, δ) for the lowest v_kin RMSE
against the true base velocity. Reports the stance-classification quality of the chosen weights against the contact
sensor (instrumentation only — the sensor never enters v_kin).

Usage: python scripts/tests/vkin_calibrate.py --headless --ckpt <π_S0.pt> [--num_envs 512] [--steps 400]
"""

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import load_env_state  # noqa: E402
from hexasim.rl.s0_ppo import widen_actor_state  # noqa: E402
from hexasim.rl.s1.aux import CUR_PROBE, ERR, GYRO, QD, Q  # noqa: E402
from hexasim.rl.s1.kinematics import v_kin  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Hexapod-S1-v0", num_envs=args.num_envs)
cfg.seed = 3
cfg.aux.buffer_len = args.steps + 8
env = gym.make("Hexapod-S1-v0", cfg=cfg).unwrapped
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
load_env_state(env, ck)
obs, _ = env.reset()
dim = obs["policy"].shape[1]
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
pol.load_state_dict({k: v.to(env.device) for k, v in widen_actor_state(ck["actor_state_dict"], dim, 0.05).items()})
pol.eval()
with torch.inference_mode():
    for _ in range(args.steps):
        obs, *_ = env.step(pol(TensorDict({"policy": obs["policy"]}, [env.num_envs])))

buf = env.aux.buffer
rows = buf.row_step >= 0
frame = buf.frame[rows][:, :]
cur = buf.cur[rows]
q = frame[:, Q] + env.aux.q_default
v_true = cur[:, CUR_PROBE : CUR_PROBE + 3]
contact = cur[:, :6] > 0.5
walking = v_true[:, :2].norm(dim=-1) > 0.03  # calibrate on stepping, not standing
best = None
for tau_e, s, delta in itertools.product(
    (0.0, 0.002, 0.005, 0.01, 0.02), (0.001, 0.002, 0.003, 0.006), (-0.02, -0.015, -0.01, -0.005, 0.0)
):
    v, w = v_kin(q, frame[:, QD], frame[:, ERR], frame[:, GYRO], tau_e, s, delta)
    rmse = (v - v_true)[walking].pow(2).sum(-1).mean().sqrt().item()
    if best is None or rmse < best[0]:
        best = (rmse, tau_e, s, delta, w)
rmse, tau_e, s, delta, w = best
v0, w0 = v_kin(q, frame[:, QD], frame[:, ERR], frame[:, GYRO])  # current defaults
rmse0 = (v0 - v_true)[walking].pow(2).sum(-1).mean().sqrt().item()
pred = w > 0.5
acc = (pred == contact)[walking].float().mean().item()
w_st, w_sw = w[walking][contact[walking]].mean().item(), w[walking][~contact[walking]].mean().item()
print(f"VKIN rows {int(rows.sum())} (walking {int(walking.sum())}); v_true rms {v_true[walking].norm(dim=-1).pow(2).mean().sqrt():.4f}")
print(f"VKIN current defaults: RMSE {rmse0:.4f} m/s")
print(f"VKIN best tau_e {tau_e} s {s} delta {delta}: RMSE {rmse:.4f} m/s")
print(f"VKIN stance weights vs contact sensor: accuracy {acc:.3f}; mean w stance {w_st:.3f}, swing {w_sw:.3f}")
env.close()
app.close()
