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

``--pretty`` switches the showcase renderer on (``demo_look``): HDRI sky, raking sun + rim light, one MDL per
terrain material and RTX quality settings. It is visual only — the policy and the physics are unchanged.

Usage:
  python scripts/rsl_rl/demo_teleop.py --ckpt <π_S0.pt> --level 9
  python scripts/rsl_rl/demo_teleop.py --ckpt <π_S0.pt> --level 18   # held-out difficulty
  python scripts/rsl_rl/demo_teleop.py --ckpt <π_S0.pt> --level 9 --pretty --res 1920x1080 --take_seconds 15
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="logs/rsl_rl/hexapod_s1/2026-09-26_02-33-23_s1_pilot2_seed42/model_8000.pt", help="policy checkpoint (infer task from env_state_dict)")
parser.add_argument("--level", type=int, default=9, help="terrain difficulty level, 0–20")
parser.add_argument("--video_dir", default="logs/videos/demo", help="where R writes the mp4 takes")
parser.add_argument("--no_video", action="store_true", help="skip the recorder (viewer only)")
parser.add_argument(
    "--selftest", type=int, default=0, help="headless check: run N steps on a scripted command, no keyboard, then exit"
)
parser.add_argument("--pretty", action="store_true", help="showcase rendering (hexasim...demo_look): sky, MDLs, RTX")
parser.add_argument("--sky", default="noon_grass", help="--pretty: HDRI key, see demo_look.SKIES")
parser.add_argument("--quality", default="high", choices=["high", "mid"], help="--pretty: RT2 path tracing, or RT1")
parser.add_argument("--res", default="1280x720", help="--pretty: recorded resolution, e.g. 1920x1080")
parser.add_argument("--fps", type=float, default=0.0, help="video frame rate; 0 = native 50. 25 = 2x slow motion")
parser.add_argument("--take_seconds", type=float, default=0.0, help="stop a take automatically after N s; 0 = manual")
parser.add_argument(
    "--robot_mdl", default="", help="--pretty: re-skin the robot (grey, light_grey, abs, …); '' = its own flat black"
)
parser.add_argument(
    "--terrain_mdl", default="granite", help="--pretty: one ground material (demo_look.GROUND_MDL), or 'multi'"
)
parser.add_argument("--sky_tilt", type=float, default=270.0, help="--pretty: sky rotation about +X; 270 = upright")
parser.add_argument("--light_gain", type=float, default=1.0, help="--pretty: scales sun, sky and rim together")
parser.add_argument(
    "--backdrop", type=float, default=400.0, help="--pretty: visual-only ground slab, metres; 0 = off"
)
parser.add_argument(
    "--backdrop_z", type=float, default=-0.05, help="--pretty: slab top height; must clear the terrain's lowest point"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video  # the recorder needs a render product
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim import demo_look  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.terrain import LEVELS_DESIGNED, S0B_TERRAIN_CFG  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

V_X_MAX, V_Y_MAX, YAW_MAX = 0.20, 0.05, 0.4  # the trained command ranges (spec §1 + D19)
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
base_task = task_for(ckpt)
play_task = base_task.replace("-v0", "-Play-v0")

cfg = parse_env_cfg(play_task, num_envs=1)
configure(cfg, ckpt)
eval_physx_buffers(cfg)
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
if args.pretty:
    w, h = (int(v) for v in args.res.lower().split("x"))
    demo_look.apply(
        cfg,
        sky=args.sky,
        sky_tilt=args.sky_tilt,
        light_gain=args.light_gain,
        quality=args.quality,
        res=(w, h),
        robot_mdl=args.robot_mdl or None,
        terrain_mdl=args.terrain_mdl,
        backdrop=args.backdrop,
        backdrop_z=args.backdrop_z,
    )
env = gym.make(play_task, cfg=cfg, render_mode=None if args.no_video else "rgb_array")
recorder = None
if not args.no_video:
    recorder = gym.wrappers.RecordVideo(
        env,
        video_folder=args.video_dir,
        step_trigger=lambda _: False,
        video_length=0,
        name_prefix="s0_demo",
        fps=args.fps or None,  # None → the env's native 1/step_dt = 50 Hz
    )
    env = recorder
base = env.unwrapped
ckpt = torch.load(args.ckpt, map_location=base.device, weights_only=False)
load_env_state(base, ckpt)
with torch.inference_mode():
    obs, _ = env.reset()

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
state = {"recording": False, "reset": False, "take": 0, "started": 0}


def toggle_record():
    if recorder is None:
        return
    if state["recording"]:
        recorder.stop_recording()
        print(f"[demo] take {state['take']} written to {args.video_dir}")
    else:
        state["take"] += 1
        state["started"] = step
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
take_steps = int(args.take_seconds / base.step_dt)
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
        if take_steps and state["recording"] and step - state["started"] >= take_steps:
            toggle_record()

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
