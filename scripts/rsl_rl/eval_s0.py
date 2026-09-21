# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless gait evaluation for the S0a gates (paradigm spec §4.2): one 10 s episode per env, deterministic actions.

Metrics, per env over the steps before its first termination, then averaged:
  fall_rate        fraction of envs terminated (not timed out) within the episode
  vx_mean          body-frame forward velocity, m/s
  duty_factor      fraction of steps each foot is in contact (|F| > 1 N)
  clearance        mean over swings (apex > 5 mm) of the peak toe height, m (toe point, spec §1); plus quartiles
  micro_lifts      touchdowns after a lift of ≤ 5 mm (stutter), per leg per second
  coxa_hf_power    coxa joint-angle power above 5 Hz, rad² (mean over coxa joints)
  v0_rmse          V_0 vs true base velocity (S0a tasks only), m/s
S0b (fixed --cmd for the whole episode; clearance is toe height above the terrain):
  vx_rmse_step     per-step v_x − v_cmd,x RMSE (includes the within-stride velocity oscillation), m/s
  vx_rmse_cycle    RMSE of v_x averaged over gait-period windows (1/f_ref), first window (start from rest)
                   dropped — the tracking gate
  stride           per-leg touchdown-to-touchdown toe displacement (world xy), m; stride_ref = |v_cmd|/f_ref
  standing (all-zero cmd): base xy drift over the episode (m), mean Σ|q̇| (rad/s), feet-in-contact fraction

Usage:
  python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-Flat-v0 --ckpt <π_base.pt> [--rest_tibia_deg 30]
  python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0a-v0 --ckpt <π_S0a.pt>
  python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0b-v0 --ckpt <π_S0.pt> --cmd 0.10 0 0 \
      --terrain_level 0 [--no_drops]
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
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"), help="S0b: fixed twist_cmd")
parser.add_argument("--terrain_level", type=int, default=None, help="S0b: pin every env to this terrain level")
parser.add_argument("--no_drops", action="store_true", help="S0a/S0b: disable dropped reads")
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
s0b = hasattr(cfg, "commands") and cfg.commands is not None
if s0b:
    from hexasim.tasks.manager_based.hexasim.mdp import pin_command, pin_terrain_level

    assert args.cmd is not None, "S0b evaluation needs --cmd"
    pin_command(cfg, *args.cmd)
if args.no_drops:
    cfg.drop_p0 = cfg.drop_k = 0.0
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
if args.terrain_level is not None:
    pin_terrain_level(env, args.terrain_level)
ckpt = torch.load(args.ckpt, map_location=env.device, weights_only=False)
obs, _ = env.reset()
if s0a:
    env.v0.load_state_dict(ckpt["v0_state_dict"])
    env.v0_train = False

td = TensorDict({"policy": obs["policy"]}, batch_size=[env.num_envs])
dist = {"class_name": "GaussianDistribution", "init_std": 1.0}
policy = MLPModel(td, {"actor": ["policy"]}, "actor", 18, [256, 128, 64], "elu", True, dist).to(env.device)
state = ckpt["actor_state_dict"]
if state["mlp.0.weight"].shape[1] < td["policy"].shape[1]:  # older layout (π_S0a on S0b): zero-widen
    from hexasim.rl.s0_ppo import widen_actor_state

    state = widen_actor_state({k: v.cpu() for k, v in state.items()}, td["policy"].shape[1], 0.05)
policy.load_state_dict({k: v.to(env.device) for k, v in state.items()})
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
micro_count = torch.zeros(N, 6, device=env.device)
apexes = []
v0_se = torch.zeros(N, device=env.device)
coxa_traj = torch.zeros(T, N, len(coxa), device=env.device)
vx_traj = torch.zeros(T, N, device=env.device)
live_traj = torch.zeros(T, N, dtype=torch.bool, device=env.device)
last_td = torch.full((N, 6, 2), float("nan"), device=env.device)
stride_sum = torch.zeros(N, 6, device=env.device)
stride_count = torch.zeros(N, 6, device=env.device)
qd_sum = torch.zeros(N, device=env.device)
xy0 = robot.data.root_pos_w.torch[:, :2].clone()
xy_end = xy0.clone()

with torch.inference_mode():
    for t in range(T):
        act = policy(TensorDict({"policy": obs["policy"]}, batch_size=[N]))
        obs, _, terminated, truncated, _ = env.step(act)
        # post-step state belongs to the episode unless it was just reset
        live = alive & ~terminated & ~truncated
        vx_sum += robot.data.root_lin_vel_b.torch[:, 0] * live
        contact = torch.norm(sensor.data.net_forces_w.torch, dim=-1) > 1.0
        contact_sum += contact * live.unsqueeze(1)
        toe = toe_pos_w(robot, tibia)
        toe_z = toe[..., 2] - env.terrain_z(toe[..., :2])
        swing_peak = torch.where(contact, swing_peak, torch.maximum(swing_peak, toe_z))
        touchdown = contact & (swing_peak > 0) & live.unsqueeze(1)
        real = touchdown & (swing_peak > 0.005)  # a swing with apex > 5 mm; below that it is a micro-lift/stutter
        apexes.append(swing_peak[real])
        peak_sum += swing_peak * real
        swing_count += real
        micro_count += touchdown & ~real
        step_len = torch.norm(toe[..., :2] - last_td, dim=-1)
        counted = real & ~torch.isnan(step_len)
        stride_sum += torch.nan_to_num(step_len) * counted
        stride_count += counted
        last_td = torch.where(real.unsqueeze(-1), toe[..., :2], last_td)
        swing_peak = torch.where(contact, torch.zeros_like(swing_peak), swing_peak)
        if s0a:
            v0_se += (env.v_in - robot.data.root_lin_vel_b.torch).pow(2).sum(-1) * live
        coxa_traj[t] = robot.data.joint_pos.torch[:, coxa]
        vx_traj[t] = robot.data.root_lin_vel_b.torch[:, 0]
        live_traj[t] = live
        qd_sum += robot.data.joint_vel.torch.abs().sum(-1) * live
        xy_end = torch.where(live.unsqueeze(1), robot.data.root_pos_w.torch[:, :2], xy_end)
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
    "clearance_q25_50_75": torch.cat(apexes).quantile(torch.tensor([0.25, 0.5, 0.75], device=env.device)).tolist(),
    "swings_per_leg_per_s": (swing_count.sum(0) / (steps.sum() * env.step_dt)).tolist(),
    "micro_lifts_per_leg_per_s": (micro_count.sum(0) / (steps.sum() * env.step_dt)).tolist(),
    "coxa_hf_power": spec[freqs > 5.0].sum(0).mean().item(),
}
if s0a:
    m["v0_rmse"] = (v0_se.sum() / steps.sum()).sqrt().item()
if s0b:
    vx_c, vy_c, wz_c = args.cmd
    m.update(cmd=args.cmd, terrain_level=args.terrain_level, no_drops=args.no_drops)
    m["vx_rmse_step"] = ((vx_traj - vx_c).pow(2) * live_traj).sum().div(live_traj.sum()).sqrt().item()
    standing = vx_c == vy_c == wz_c == 0.0
    if standing:
        m["stand_drift_m"] = torch.norm(xy_end - xy0, dim=1).mean().item()
        m["stand_joint_speed"] = (qd_sum / steps).mean().item()
    else:
        f_ref = min(max(vx_c / 0.20, 0.8), 2.5)
        w = round(1.0 / (f_ref * env.step_dt))
        k = T // w
        win_v = vx_traj[: k * w].view(k, w, N).mean(1)[1:]
        win_live = live_traj[: k * w].view(k, w, N).all(1)[1:]
        m["vx_rmse_cycle"] = ((win_v - vx_c).pow(2) * win_live).sum().div(win_live.sum()).sqrt().item()
        m["stride"] = (stride_sum.sum() / stride_count.sum()).item()
        m["stride_per_leg"] = (stride_sum.sum(0) / stride_count.sum(0).clamp(min=1)).tolist()
        m["stride_ref"] = math.hypot(vx_c, vy_c) / f_ref
print("EVAL", json.dumps(m, indent=1))
if args.out:
    with open(args.out, "w") as f:
        json.dump(m, f, indent=1)
env.close()
app.close()
