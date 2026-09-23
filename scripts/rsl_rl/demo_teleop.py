# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard-driven demonstration of an S0b policy on every terrain type at one difficulty, with video recording.

The terrain is one difficulty (``--level``, 0–20 on the 15-level scale: 0–14 are the training range, 15–20 the
held-out suite) laid out as columns of terrain type — flat, rough, slope down, slope up, bumps, stairs down,
stairs up — three rows deep. Walking sideways crosses from one type to the next; each tile is 4 m.

Keys (the Isaac Sim window must have focus):
    ↑ / ↓        forward speed 0 … 0.20 m/s (backwards is not commanded: the policy never trained on it)
    ← / →        sideways ±0.05 m/s
    Z / X        turn ±0.4 rad/s
    (no key)     zero command — the policy's standing mode
    R            start / stop a video take (mp4 in --video_dir)
    N            reset the robot (fresh dynamics randomisation: friction, compliance, payload, servos)

Dynamics randomisation, sensor noise and dropped reads stay exactly as in training.

Usage:
  python scripts/rsl_rl/demo_teleop.py --ckpt <π_S0.pt> --level 9
  python scripts/rsl_rl/demo_teleop.py --ckpt <π_S0.pt> --level 18   # held-out difficulty
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="policy checkpoint (117-D S0b actor)")
parser.add_argument("--level", type=int, default=9, help="terrain difficulty level, 0–20")
parser.add_argument("--video_dir", default="logs/videos/demo", help="where R writes the mp4 takes")
parser.add_argument("--no_video", action="store_true", help="skip the recorder (viewer only)")
parser.add_argument(
    "--selftest", type=int, default=0, help="headless check: run N steps on a scripted command, no keyboard, then exit"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video  # the recorder needs a render product
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.terrain import LEVELS_DESIGNED, S0B_TERRAIN_CFG  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

V_X_MAX, V_Y_MAX, YAW_MAX = 0.20, 0.05, 0.4  # the trained command ranges (spec §1 + D19)

cfg = parse_env_cfg("Hexapod-S0b-v0", num_envs=1)
cfg.curriculum = None
# One difficulty, one column per terrain type, three rows deep.
types = list(S0B_TERRAIN_CFG.sub_terrains)
difficulty = args.level / LEVELS_DESIGNED
cfg.scene.terrain.terrain_generator = S0B_TERRAIN_CFG.replace(
    num_rows=3,
    num_cols=len(types),
    difficulty_range=(difficulty, difficulty),
    sub_terrains={k: v.replace(proportion=1.0 / len(types)) for k, v in S0B_TERRAIN_CFG.sub_terrains.items()},
)
cfg.scene.terrain.max_init_terrain_level = 0
pin_command(cfg, 0.0)  # the keyboard writes the command every step
# Chase camera on the robot.
cfg.viewer.origin_type = "asset_root"
cfg.viewer.asset_name = "robot"
cfg.viewer.eye = (1.6, 1.6, 0.8)
cfg.viewer.lookat = (0.0, 0.0, 0.15)

env = gym.make("Hexapod-S0b-v0", cfg=cfg, render_mode=None if args.no_video else "rgb_array")
recorder = None
if not args.no_video:
    recorder = gym.wrappers.RecordVideo(
        env, video_folder=args.video_dir, step_trigger=lambda _: False, video_length=0, name_prefix="s0_demo"
    )
    env = recorder
base = env.unwrapped

ckpt = torch.load(args.ckpt, map_location=base.device, weights_only=False)
with torch.inference_mode():
    obs, _ = env.reset()
base.v0.load_state_dict(ckpt["v0_state_dict"])
base.v0_train = False
policy = MLPModel(
    TensorDict({"policy": obs["policy"]}, batch_size=[1]),
    {"actor": ["policy"]},
    "actor",
    18,
    [256, 128, 64],
    "elu",
    True,
    {"class_name": "GaussianDistribution", "init_std": 1.0},
).to(base.device)
policy.load_state_dict(ckpt["actor_state_dict"])
policy.eval()

keyboard = (
    None
    if args.selftest
    else Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=V_X_MAX, v_y_sensitivity=V_Y_MAX, omega_z_sensitivity=YAW_MAX))
)
state = {"recording": False, "reset": False, "take": 0}


def toggle_record():
    if recorder is None:
        return
    if state["recording"]:
        recorder.stop_recording()
        print(f"[demo] take {state['take']} written to {args.video_dir}")
    else:
        state["take"] += 1
        recorder.start_recording(f"s0_demo_L{args.level}_take{state['take']}")
        print(f"[demo] recording take {state['take']} …")
    state["recording"] = not state["recording"]


if keyboard is not None:
    keyboard.add_callback("R", toggle_record)
    keyboard.add_callback("N", lambda: state.update(reset=True))
    keyboard.reset()
print(f"[demo] terrain types across the columns: {types} at level {args.level} (difficulty {difficulty:.2f})")
print("[demo] arrows = walk, Z/X = turn, no key = stand, R = record, N = reset")

cmd_term = base.cmd_term
step = 0
while app.is_running() and (not args.selftest or step < args.selftest):
    step += 1
    with torch.inference_mode():
        scripted = (0.10, 0.0, 0.2 if step > args.selftest // 2 else 0.0)  # selftest: walk, then walk and turn
        kb = torch.tensor(scripted if args.selftest else keyboard.advance(), dtype=torch.float32, device=base.device)
        cmd = torch.stack([kb[0].clamp(0.0, V_X_MAX), kb[1].clamp(-V_Y_MAX, V_Y_MAX), kb[2].clamp(-YAW_MAX, YAW_MAX)])
        cmd_term.vel_command_b[:] = cmd
        cmd_term.is_standing_env[:] = cmd.abs().max() < 1e-6  # zero command → standing mode, as in training
        obs, *_ = env.step(policy(TensorDict({"policy": obs["policy"]}, batch_size=[1])))
        if state["reset"]:
            obs, _ = env.reset()
            state["reset"] = False

if args.selftest:
    toggle_record()  # start a take
    for _ in range(50):
        with torch.inference_mode():
            obs, *_ = env.step(policy(TensorDict({"policy": obs["policy"]}, batch_size=[1])))
    root = base.scene["robot"].data.root_pos_w.torch
    print(
        f"SELFTEST ok: {step} steps, base height above terrain "
        f"{(root[0, 2] - base.terrain_z(root[:, :2])[0]).item():.3f} m"
    )
if state["recording"]:
    toggle_record()
env.close()
app.close()
