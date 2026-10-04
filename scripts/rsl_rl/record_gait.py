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

The task follows the checkpoint's actor width (``hexasim.rl.load_policy.TASK_BY_DIM``).

``--pretty`` switches the showcase renderer on (``demo_look``): HDRI sky, raking sun + rim light, one MDL per
terrain material, RTX quality settings and a chosen resolution. It is visual only — the policy, the physics and the
printed geometry are identical with and without it.

Usage:
  python scripts/rsl_rl/record_gait.py --headless --ckpt <pt> --tag s0_final [--cmd 0.15] [--seconds 10]
  python scripts/rsl_rl/record_gait.py --headless --ckpt <pt> --tag hero --pretty --view hero --res 1920x1080
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", required=True, help="name for the video file and the printed row")
parser.add_argument("--cmd", type=float, default=0.15, help="forward speed command, S0b tasks only")
parser.add_argument("--yaw_cmd", type=float, default=0.0, help="yaw speed command, S0b tasks only")
parser.add_argument(
    "--terrain_type",
    default=None,
    help="S0b/S1: one generated terrain type (mdp.terrain.EVAL_TERRAIN_TYPES), e.g. rough; default flat",
)
parser.add_argument("--terrain_level", type=int, default=0, help="terrain level 0-20 for --terrain_type")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument(
    "--seed", type=int, default=42, help="env seed; without one the DR/noise draws differ between runs"
)
parser.add_argument("--video_dir", default="logs/videos/gait")
parser.add_argument("--view", default="side", choices=["side", "front", "diag", "hero", "orbit"])
parser.add_argument("--pretty", action="store_true", help="showcase rendering (hexasim...demo_look): sky, MDLs, RTX")
parser.add_argument("--sky", default="noon_grass", help="--pretty: HDRI key, see demo_look.SKIES")
parser.add_argument("--quality", default="high", choices=["high", "mid"], help="--pretty: RT2 path tracing, or RT1")
parser.add_argument("--res", default="1280x720", help="--pretty: recorded resolution, e.g. 1920x1080")
parser.add_argument(
    "--fps",
    type=float,
    default=0.0,
    help="video frame rate; 0 = native 50 (real time). 25 = clean 2x slow motion. Does not change the simulation.",
)
parser.add_argument("--smooth", type=float, default=0.92, help="chase-camera EMA on the body position, 0 = off")
parser.add_argument(
    "--uniform_friction",
    action="store_true",
    help="DEMO ONLY, CHANGES PHYSICS: restrict the material field to 'concrete' (mu 0.8-1.0, rigid) so the robot "
    "does not cross low-friction patches. Not the trained condition — never use such a run as evidence.",
)
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
parser.add_argument("--orbit_turns", type=float, default=0.5, help="--view orbit: revolutions over the whole take")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.load_policy import configure, eval_physx_buffers, load_env_state, task_for  # noqa: E402
from hexasim.tasks.manager_based.hexasim import demo_look  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import materials  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import LEG_NAMES, coxa_leg_geometry, toe_pos_w  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from hexasim.tasks.manager_based.hexasim.mdp.terrain import single_type_generator  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
base_task = task_for({"actor_state_dict": state})
# S2 has no Play config: the training task with one env (episodes of 10 s cover the recording).
task = base_task if base_task == "Hexapod-S2-v0" else base_task.replace("-v0", "-Play-v0")
cfg = parse_env_cfg(task, num_envs=1)
configure(cfg, torch.load(args.ckpt, map_location="cpu", weights_only=False))
eval_physx_buffers(cfg)
cfg.curriculum = None
s0b = hasattr(cfg, "commands") and cfg.commands is not None  # S0b and later
if s0b:
    pin_command(cfg, args.cmd, 0.0, args.yaw_cmd)
    if args.terrain_type is None:
        cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
            num_rows=1, num_cols=2, difficulty_range=(0.0, 0.0), sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)}
        )
    else:
        cfg.scene.terrain.terrain_generator = single_type_generator(args.terrain_type)
cfg.seed = args.seed  # the terrain layout and every DR/noise draw key off this; without it no two runs match
# S2 has no Play cfg, so episode_length_s is the training 10 s and a longer take gets cut by a time-out
# reset — the robot snapping back to its spawn mid-video. Falls still reset, which is the honest part.
cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)
if hasattr(cfg, "v0_ramp_steps"):
    cfg.v0_ramp_steps = (0, 1)
# Chase camera, aimed at the robot every step below: headless runs create no viewport camera controller, so
# ``cfg.viewer.origin_type = "asset_root"`` is ignored, and ``sim.set_camera_view`` does not move the recorder's
# camera either — only ViewportManager on the Kit perspective camera does.
_FIXED = {
    "side": ((0.0, 1.3, 0.35), (0.0, 0.0, 0.0)),
    "front": ((1.4, 0.0, 0.35), (0.0, 0.0, 0.0)),
    "diag": ((1.1, 1.1, 0.5), (0.0, 0.0, 0.0)),
    "hero": ((0.75, 0.75, 0.06), (0.0, 0.0, 0.14)),  # low and close: legs and toes against the sky
}


def camera(t: int, steps: int) -> tuple[tuple, tuple]:
    """(eye, lookat) offsets from the body, in world axes. ``orbit`` turns ``--orbit_turns`` times over the take, so
    the angular rate follows ``--seconds``: a longer take at the same turns is a slower orbit."""
    if args.view != "orbit":
        return _FIXED[args.view]
    a = 2.0 * math.pi * args.orbit_turns * t / max(steps, 1)
    return (1.2 * math.cos(a), 1.2 * math.sin(a), 0.45), (0.0, 0.0, 0.12)


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

env = gym.make(task, cfg=cfg, render_mode="rgb_array")
steps = int(args.seconds / env.unwrapped.step_dt)
env = gym.wrappers.RecordVideo(
    env,
    video_folder=args.video_dir,
    step_trigger=lambda s: s == 0,
    video_length=steps,
    name_prefix=args.tag,
    fps=args.fps or None,  # None → the env's native 1/step_dt = 50 Hz
)
base = env.unwrapped
if args.uniform_friction:
    # Friction comes from the material field (entries over the terrain) times a per-episode palette, written onto
    # the feet by FootWriter — never from the visual material. Giving every entry concrete's range makes the
    # ground uniformly grippy while leaving the field, its seed and the observation layout untouched. (Shrinking
    # the field to one entry would be the obvious route and is a trap: MaterialField._sites resamples until two
    # neighbouring sites differ, which cannot terminate with a single entry.)
    assert hasattr(base, "palette"), f"{task} has no material palette; --uniform_friction would do nothing"
    concrete = materials.LIBRARY[0]
    base.palette.lo[:, 0], base.palette.hi[:, 0] = concrete.mu_s
    base.palette.lo[:, 1] = base.palette.hi[:, 1] = math.log10(materials.RIGID_K)
    base.palette.redraw(torch.arange(base.num_envs, device=base.device))
    print(f"[record_gait] UNIFORM FRICTION: every entry = concrete mu {concrete.mu_s}, rigid."
          " NOT the trained condition.")
if s0b:
    pin_terrain_level(base, args.terrain_level)
ck = torch.load(args.ckpt, map_location=base.device, weights_only=False)
load_env_state(base, ck)
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
policy.load_state_dict(ck["actor_state_dict"])
policy.eval()

robot = base.scene["robot"]
groups = {t: [i for i, n in enumerate(robot.joint_names) if n.endswith("_" + t)] for t in ("coxa", "femur", "tibia")}
acc = {k: torch.zeros(6, device=base.device) for k in ("coxa", "femur", "tibia", "x", "y", "z", "reach", "tib_lat")}
vx = 0.0
n = 0
with torch.inference_mode():
    anchor = robot.data.root_pos_w.torch[0].cpu().clone()  # EMA of the body position: the raw root bobs and yaws
    for t in range(steps):
        root = robot.data.root_pos_w.torch[0].cpu()
        anchor = args.smooth * anchor + (1.0 - args.smooth) * root
        eye, lookat = camera(t, steps)
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=(anchor + torch.tensor(eye)).tolist(),
            target=(anchor + torch.tensor(lookat)).tolist(),
        )
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
