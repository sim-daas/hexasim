# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record one policy's gait on flat ground and print its leg geometry (phase2-progress §2.1.33).

Walks at ``--cmd`` (S0b tasks; Flat/S0a have no command and simply walk), records a side-view video, and reports,
averaged over the run and per leg (RF RM RB LF LM LB):

  coxa / femur / tibia   mean joint angle, degrees (absolute, not relative to the rest pose)
  toe x / y / z          toe position in the body frame, metres
  reach                  toe distance from its coxa joint in the horizontal plane, projected on the outward
                         direction of that hip: **positive = toe outboard of the hip, negative = tucked inboard**
  tibia lateral          lateral component of the tibia direction (tibia joint → toe), + = pointing outboard
  stance width           mean |toe y| of the left and right legs

The task follows the checkpoint's actor width: 75 → Hexapod-Flat-v0, 114 → Hexapod-S0a-v0, 117 → Hexapod-S0b-v0.

Usage:
  python scripts/rsl_rl/record_gait.py --headless --ckpt <pt> --tag s0_final [--cmd 0.15] [--seconds 10]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", required=True, help="name for the video file and the printed row")
parser.add_argument("--cmd", type=float, default=0.15, help="forward speed command, S0b tasks only")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--video_dir", default="logs/videos/gait")
parser.add_argument("--view", default="side", choices=["side", "front", "diag"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import LEG_NAMES, coxa_leg_geometry, toe_pos_w  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
dim = state["mlp.0.weight"].shape[1]
task = {75: "Hexapod-Flat-v0", 114: "Hexapod-S0a-v0", 117: "Hexapod-S0b-v0"}[dim]
cfg = parse_env_cfg(task, num_envs=1)
cfg.curriculum = None
s0b = task == "Hexapod-S0b-v0"
if s0b:
    pin_command(cfg, args.cmd)
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        num_rows=1, num_cols=2, difficulty_range=(0.0, 0.0), sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)}
    )
if hasattr(cfg, "v0_ramp_steps"):
    cfg.v0_ramp_steps = (0, 1)
eye = {"side": (0.0, 1.3, 0.35), "front": (1.4, 0.0, 0.35), "diag": (1.1, 1.1, 0.5)}[args.view]
cfg.viewer.origin_type = "asset_root"
cfg.viewer.asset_name = "robot"
cfg.viewer.eye = eye
cfg.viewer.lookat = (0.0, 0.0, 0.12)

env = gym.make(task, cfg=cfg, render_mode="rgb_array")
steps = int(args.seconds / env.unwrapped.step_dt)
env = gym.wrappers.RecordVideo(
    env, video_folder=args.video_dir, step_trigger=lambda s: s == 0, video_length=steps, name_prefix=args.tag
)
base = env.unwrapped
if s0b:
    pin_terrain_level(base, 0)
ck = torch.load(args.ckpt, map_location=base.device, weights_only=False)
with torch.inference_mode():
    obs, _ = env.reset()
if "v0_state_dict" in ck:
    base.v0.load_state_dict(ck["v0_state_dict"])
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
policy.load_state_dict(ck["actor_state_dict"])
policy.eval()

robot = base.scene["robot"]
groups = {t: [i for i, n in enumerate(robot.joint_names) if n.endswith("_" + t)] for t in ("coxa", "femur", "tibia")}
acc = {k: torch.zeros(6, device=base.device) for k in ("coxa", "femur", "tibia", "x", "y", "z", "reach", "tib_lat")}
vx = 0.0
n = 0
with torch.inference_mode():
    for t in range(steps):
        obs, *_ = env.step(policy(TensorDict({"policy": obs["policy"]}, batch_size=[1])))
        if t < steps // 5:  # let the gait settle before averaging
            continue
        n += 1
        for name, ids in groups.items():
            acc[name] += robot.data.joint_pos.torch[0, ids]
        p_toe, r = coxa_leg_geometry(base)
        coxa_dir = p_toe[0, :, :2] - r[0, :, :2]  # base → coxa joint, body frame
        outward = coxa_dir / coxa_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        acc["x"] += p_toe[0, :, 0]
        acc["y"] += p_toe[0, :, 1]
        acc["z"] += p_toe[0, :, 2]
        acc["reach"] += (r[0, :, :2] * outward).sum(-1)
        tibia_dir = toe_pos_w(robot, base._tibia_body_ids)[0] - robot.data.body_pos_w.torch[0, base._tibia_body_ids]
        quat = robot.data.root_quat_w.torch[0:1].expand(6, 4)
        from isaaclab.utils.math import quat_apply_inverse  # noqa: PLC0415

        acc["tib_lat"] += (quat_apply_inverse(quat, tibia_dir)[:, :2] * outward).sum(-1)
        vx += robot.data.root_lin_vel_b.torch[0, 0].item()

deg = 57.2958
print(f"\nGAIT {args.tag} ({task}, cmd {args.cmd if s0b else 'n/a'}): mean v_x {vx / n:.3f} m/s, {n} steps averaged")
print("leg      coxa°  femur°  tibia°   toe x   toe y   toe z   reach  tibia_lat")
for k, leg in enumerate(LEG_NAMES):
    print(
        f"{leg:9s}{acc['coxa'][k] / n * deg:6.1f}  {acc['femur'][k] / n * deg:6.1f}  {acc['tibia'][k] / n * deg:6.1f}"
        f"  {acc['x'][k] / n:6.3f}  {acc['y'][k] / n:6.3f}  {acc['z'][k] / n:6.3f}"
        f"  {acc['reach'][k] / n:6.3f}  {acc['tib_lat'][k] / n:9.3f}"
    )
width = (acc["y"][:3].abs().sum() + acc["y"][3:].abs().sum()) / n / 6
print(f"mean |toe y| (stance half-width) {width:.3f} m; video → {args.video_dir}/{args.tag}-step-0.mp4")
env.close()
app.close()
