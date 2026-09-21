# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless gait evaluation for the S0a gates (paradigm spec §4.2): one 10 s episode per env, deterministic actions.

Metrics, per env over the steps before its first termination, then averaged:
  fall_rate        fraction of envs terminated (not timed out) within the episode
  vx_mean          body-frame forward velocity, m/s
  duty_factor      fraction of steps each foot is in contact (|F| > 1 N)
  clearance        mean over swings of the peak toe height, m (toe point, spec §1)
  coxa_hf_power    coxa joint-angle power above 5 Hz, rad² (mean over coxa joints)
  v0_rmse          V_0 vs true base velocity (S0a tasks only), m/s

Usage:
  python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-Flat-v0 --ckpt <π_base.pt> [--rest_tibia_deg 30]
  python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0a-v0 --ckpt <π_S0a.pt>
"""

import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rest_tibia_deg", type=float, default=None, help="override the rest-pose tibia angle")
parser.add_argument("--out", default=None, help="write metrics JSON here")
parser.add_argument("--stall", type=float, default=None, help="diagnosis: override the AX-12A stall torque, N·m")
parser.add_argument("--servo", nargs="*", default=[], help="diagnosis: AX-12A cfg overrides, e.g. compliance_slope=4")
parser.add_argument(
    "--ablate",
    nargs="*",
    default=[],
    choices=["mast", "friction", "jitter", "clip", "payload", "resetpose"],
    help="diagnosis: drop the mast / multiply friction / phase jitter / action clip / payload;"
    " resetpose: reset tibias around π_base's −30° instead of the S0a rest pose",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def toe_pos_w(robot, tibia_ids):
    """Toe point (spec §1). Duplicated from mdp.rewards so the script also runs on the Phase 1 code."""
    pos, quat = robot.data.body_pos_w.torch[:, tibia_ids], robot.data.body_quat_w.torch[:, tibia_ids]
    return pos + quat_apply(quat, torch.tensor([0.095, 0.0, -0.09], device=pos.device).expand_as(pos))


cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
cfg.seed = args.seed
cfg.curriculum = None
if args.rest_tibia_deg is not None:
    cfg.scene.robot.init_state.joint_pos[".*_tibia"] = math.radians(args.rest_tibia_deg)
if "mast" in args.ablate:
    cfg.events.camera_mast_mass = None
    cfg.events.camera_mast_com = None
if "friction" in args.ablate:
    cfg.events.robot_material = None
    cfg.scene.ground.spawn.physics_material.friction_combine_mode = "average"
    cfg.scene.ground.spawn.physics_material.restitution_combine_mode = "average"
if "clip" in args.ablate:
    cfg.actions.joint_pos.clip = None
if "payload" in args.ablate:
    cfg.events.camera_mast_mass.params["mass_distribution_params"] = (0.15, 0.15)
if "resetpose" in args.ablate:
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    cfg.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_coxa", ".*_femur"])
    d = -math.radians(60.0)
    cfg.events.reset_tibia_old_pose = EventTermCfg(
        func=cfg.events.reset_robot_joints.func,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_tibia"]),
            "position_range": (d - 0.25, d + 0.25),
            "velocity_range": (-0.05, 0.05),
        },
    )
if "jitter" in args.ablate:
    from hexasim.tasks.manager_based.hexasim.hexapod_env import HexapodEnv

    _reset = HexapodEnv._reset_idx

    def _reset_no_jitter(self, env_ids):
        _reset(self, env_ids)
        self.cpg_phase[env_ids] = self._cpg_base_offsets

    HexapodEnv._reset_idx = _reset_no_jitter
if args.stall is not None:
    cfg.scene.robot.actuators["ax12a"].stall_torque = args.stall
for kv in args.servo:
    k, v = kv.split("=")
    setattr(cfg.scene.robot.actuators["ax12a"], k, type(getattr(cfg.scene.robot.actuators["ax12a"], k))(float(v)))
s0a = hasattr(cfg, "v0_ramp_steps")
if s0a:
    cfg.v0_ramp_steps = (0, 1)  # evaluate on the deployable estimate only
env = gym.make(args.task, cfg=cfg).unwrapped
ckpt = torch.load(args.ckpt, map_location=env.device, weights_only=False)
obs, _ = env.reset()
if s0a:
    env.v0.load_state_dict(ckpt["v0_state_dict"])
    env.v0_train = False

td = TensorDict({"policy": obs["policy"]}, batch_size=[env.num_envs])
dist = {"class_name": "GaussianDistribution", "init_std": 1.0}
policy = MLPModel(td, {"actor": ["policy"]}, "actor", 18, [256, 128, 64], "elu", True, dist).to(env.device)
policy.load_state_dict(ckpt["actor_state_dict"])
policy.eval()

robot, sensor = env.scene["robot"], env.scene["contact_forces"]
tibia = [i for i, n in enumerate(robot.data.body_names) if "tibia" in n]
coxa = [i for i, n in enumerate(robot.joint_names) if n.endswith("_coxa")]
T, N = int(env.max_episode_length), env.num_envs
alive = torch.ones(N, dtype=torch.bool, device=env.device)
fell = torch.zeros(N, dtype=torch.bool, device=env.device)
steps = torch.zeros(N, device=env.device)
vx_sum = torch.zeros(N, device=env.device)
contact_sum = torch.zeros(N, 6, device=env.device)
swing_peak = torch.zeros(N, 6, device=env.device)
peak_sum = torch.zeros(N, 6, device=env.device)
swing_count = torch.zeros(N, 6, device=env.device)
v0_se = torch.zeros(N, device=env.device)
coxa_traj = torch.zeros(T, N, len(coxa), device=env.device)

with torch.inference_mode():
    for t in range(T):
        act = policy(TensorDict({"policy": obs["policy"]}, batch_size=[N]))
        obs, _, terminated, truncated, _ = env.step(act)
        # post-step state belongs to the episode unless it was just reset
        live = alive & ~terminated & ~truncated
        vx_sum += robot.data.root_lin_vel_b.torch[:, 0] * live
        contact = torch.norm(sensor.data.net_forces_w.torch, dim=-1) > 1.0
        contact_sum += contact * live.unsqueeze(1)
        toe_z = toe_pos_w(robot, tibia)[..., 2]
        swing_peak = torch.where(contact, swing_peak, torch.maximum(swing_peak, toe_z))
        touchdown = contact & (swing_peak > 0)
        peak_sum += swing_peak * touchdown * live.unsqueeze(1)
        swing_count += touchdown * live.unsqueeze(1)
        swing_peak = torch.where(contact, torch.zeros_like(swing_peak), swing_peak)
        if s0a:
            v0_se += (env.v_in - robot.data.root_lin_vel_b.torch).pow(2).sum(-1) * live
        coxa_traj[t] = robot.data.joint_pos.torch[:, coxa]
        steps += live
        fell |= alive & terminated
        alive &= ~(terminated | truncated)

steps = steps.clamp(min=1)
spec = torch.fft.rfft(coxa_traj - coxa_traj.mean(0), dim=0).abs().pow(2) / T
freqs = torch.fft.rfftfreq(T, d=env.step_dt).to(env.device)
m = {
    "task": args.task,
    "ckpt": args.ckpt,
    "rest_tibia_deg": args.rest_tibia_deg,
    "ablate": args.ablate,
    "stall": args.stall,
    "servo": args.servo,
    "num_envs": N,
    "seed": args.seed,
    "fall_rate": fell.float().mean().item(),
    "vx_mean": (vx_sum / steps).mean().item(),
    "duty_factor": (contact_sum / steps.unsqueeze(1)).mean().item(),
    "duty_factor_per_leg": (contact_sum / steps.unsqueeze(1)).mean(0).tolist(),
    "clearance": (peak_sum.sum() / swing_count.sum().clamp(min=1)).item(),
    "swings_per_leg_per_s": (swing_count.sum(0) / (steps.sum() * env.step_dt)).tolist(),
    "coxa_hf_power": spec[freqs > 5.0].sum(0).mean().item(),
}
if s0a:
    m["v0_rmse"] = (v0_se.sum() / steps.sum()).sqrt().item()
print("EVAL", json.dumps(m, indent=1))
if args.out:
    with open(args.out, "w") as f:
        json.dump(m, f, indent=1)
env.close()
app.close()
