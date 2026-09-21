# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnosis: does the warm-started S0a actor, on its deployable o_p, command what π_base commands on ground truth?

In Hexapod-S0a-v0, driven by π_base's own actions (from the Flat-style 75-D ground-truth observation, mapped to
the S0a rest pose), compare per step:
  a_base  = π_base(GT 75-D obs) expressed in S0a action space (a − Δ/0.5)
  a_s0a   = S0a actor(114-D obs)
then swap each 114-D block for its ground-truth equivalent and report how much of the gap each block explains.

Usage: python scripts/tests/diag_obs_equivalence.py --headless --ckpt <S0a checkpoint with the actor still frozen>
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--base", default="logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt")
parser.add_argument("--servo", nargs="*", default=[], help="AX-12A cfg overrides, e.g. compliance_slope=4")
parser.add_argument("--noclip", action="store_true")
parser.add_argument("--nopayload", action="store_true")
parser.add_argument("--oldreset", action="store_true", help="reset tibias around π_base's −30°")
parser.add_argument("--task", default="Hexapod-S0a-v0", help="Hexapod-Flat-v0 = control (π_base's own env)")
parser.add_argument("--gtclock", action="store_true", help="α = 0: clock and v_in from ground-truth velocity")
parser.add_argument("--implicit", action="store_true", help="π_base's implicit PD (Kp 20, Kd 0.5, 3 N·m) instead of AX-12A")
parser.add_argument("--idealpd", type=float, default=None, help="Isaac Lab explicit IdealPD, same gains, this armature")
parser.add_argument("--physics_hz", type=int, default=200, help="physics rate; control stays 50 Hz")
parser.add_argument("--kp", type=float, default=20.0, help="IdealPD stiffness (damping = kp / 40)")
parser.add_argument("--implicit_gains", type=float, nargs=3, default=None, metavar=("KP", "KD", "EFFORT"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

N = 64
cfg = parse_env_cfg(args.task, num_envs=N)
flat = args.task == "Hexapod-Flat-v0"
if not flat:
    cfg.v0_ramp_steps = (10**9, 10**9 + 1) if args.gtclock else (0, 1)
cfg.sim.dt = 1.0 / args.physics_hz
cfg.decimation = args.physics_hz // 50
cfg.sim.render_interval = cfg.decimation
for kv in args.servo:
    k, v = kv.split("=")
    a = cfg.scene.robot.actuators["ax12a"]
    old = getattr(a, k)
    setattr(a, k, float(v) if old is None else type(old)(float(v)))
if args.implicit:
    from hexasim.assets.robot import PXMARK4_CFG

    imp = PXMARK4_CFG.actuators["femur_tibia"].replace(joint_names_expr=[".*"])
    if args.implicit_gains:
        kp, kd, eff = args.implicit_gains
        imp = imp.replace(stiffness=kp, damping=kd, effort_limit=eff)
    imp.stall_torque = 3.0  # read by the env's drop model only
    cfg.scene.robot.actuators = {"ax12a": imp}
if args.idealpd is not None:
    from isaaclab.actuators import IdealPDActuatorCfg

    pd = IdealPDActuatorCfg(
        joint_names_expr=[".*"], stiffness=args.kp, damping=args.kp / 40, effort_limit=3.0, armature=args.idealpd
    )
    pd.stall_torque = 3.0  # read by the env's drop model only
    cfg.scene.robot.actuators = {"ax12a": pd}
if args.noclip:
    cfg.actions.joint_pos.clip = None
if args.nopayload:
    cfg.events.camera_mast_mass.params["mass_distribution_params"] = (0.15, 0.15)
if args.oldreset:
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    cfg.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_coxa", ".*_femur"])
    cfg.events.reset_tibia_old_pose = EventTermCfg(
        func=cfg.events.reset_robot_joints.func,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_tibia"]),
            "position_range": (-math.radians(60.0) - 0.25, -math.radians(60.0) + 0.25),
            "velocity_range": (-0.05, 0.05),
        },
    )
env = gym.make(args.task, cfg=cfg).unwrapped
ck = torch.load(args.ckpt, map_location=env.device, weights_only=False)
obs, _ = env.reset()
if not flat:
    env.v0.load_state_dict(ck["v0_state_dict"])
    env.v0_train = False
dist = {"class_name": "GaussianDistribution", "init_std": 1.0}


def model(dim, state):
    m = MLPModel(TensorDict({"policy": torch.zeros(1, dim)}, [1]), {"actor": ["policy"]}, "actor", 18,
                 [256, 128, 64], "elu", True, dict(dist)).to(env.device)
    m.load_state_dict(state)
    return m.eval()


s0a = model(114, ck["actor_state_dict"])
base = model(75, torch.load(args.base, map_location=env.device, weights_only=False)["actor_state_dict"])
robot = env.scene["robot"]
delta = torch.zeros(18, device=env.device)
delta[12:] = 0.0 if flat else math.radians(60.0)
old_rest = robot.data.default_joint_pos.torch[0] - delta


def gt_obs():
    """Flat-v0 policy obs from simulator ground truth, in π_base's frame (old rest pose, raw old actions)."""
    d = robot.data
    last_old = env.action_manager.action + delta / 0.5
    clock = torch.stack((torch.sin(env.cpg_phase), torch.cos(env.cpg_phase)), -1).view(N, 12)
    return torch.cat([d.root_lin_vel_b.torch, d.root_ang_vel_b.torch, d.projected_gravity_b.torch,
                      d.joint_pos.torch - old_rest, d.joint_vel.torch, last_old, clock], -1)


blocks = {"q": slice(0, 18), "qd": slice(18, 36), "act": slice(54, 72), "gyro": slice(75, 78), "grav": slice(78, 81),
          "clock": slice(81, 93), "v_in": slice(111, 114)}
gt_map = {"q": slice(9, 27), "qd": slice(27, 45), "act": slice(45, 63), "gyro": slice(3, 6), "grav": slice(6, 9),
          "clock": slice(63, 75), "v_in": slice(0, 3)}
gap = {k: 0.0 for k in ["total", *blocks]}
steps = 0
vx_hist = []
for attr in ("joint_friction_coeff", "joint_dynamic_friction_coeff", "joint_viscous_friction_coeff", "joint_vel_limits",
             "joint_effort_limits", "joint_armature", "joint_stiffness", "joint_damping"):
    if hasattr(robot.data, attr):
        v = getattr(robot.data, attr).torch[0]
        print(f"sim {attr}: {[round(x, 4) for x in v[[0, 6, 12]].tolist()]}")
names = robot.joint_names
types = {t: [i for i, n in enumerate(names) if n.endswith(t)] for t in ("_coxa", "_femur", "_tibia")}
trk = {t: [0.0, 0.0, 0.0] for t in types}  # |q_target − q|, q̇ rms², |τ applied|
with torch.inference_mode():
    for t in range(250):
        g = gt_obs()
        a_base = base(TensorDict({"policy": g}, [N])) - delta / 0.5
        o = obs["policy"]
        a_s0a = None if flat else s0a(TensorDict({"policy": o}, [N]))
        if t >= 25 and not flat:
            gap["total"] += (a_s0a - a_base).abs().mean().item()
            for k, sl in blocks.items():
                o2 = o.clone()
                gt = g[:, gt_map[k]]
                o2[:, sl] = gt - delta if k == "q" else gt - delta / 0.5 if k == "act" else gt
                gap[k] += (s0a(TensorDict({"policy": o2}, [N])) - a_base).abs().mean().item()
            steps += 1
        obs, *_ = env.step(a_base)
        if t >= 25:
            vx_hist.append(robot.data.root_lin_vel_b.torch[:, 0].mean().item())
            q_t = env.action_manager.get_term("joint_pos").processed_actions
            for tname, ids in types.items():
                trk[tname][0] += (q_t[:, ids] - robot.data.joint_pos.torch[:, ids]).abs().mean().item()
                trk[tname][1] += robot.data.joint_vel.torch[:, ids].pow(2).mean().item()
                trk[tname][2] += robot.data.applied_torque.torch[:, ids].abs().mean().item()
print(f"args: servo={args.servo} noclip={args.noclip} nopayload={args.nopayload} oldreset={args.oldreset}")
print(f"root vx under π_base actions (last step, mean over envs): {robot.data.root_lin_vel_b.torch[:, 0].mean().item():.3f} m/s")
vx_hist.append(0)
print(f"root vx under π_base actions, mean over steps 25–250 and envs: {sum(vx_hist) / max(len(vx_hist) - 1, 1):.3f} m/s")
n_trk = max(len(vx_hist) - 1, 1)
for tname, (e, v2, tau) in trk.items():
    print(f"track {tname[1:]:>5}: |q_target − q| {e / n_trk:.3f} rad, q̇ rms {(v2 / n_trk) ** 0.5:.2f} rad/s, |τ| {tau / n_trk:.2f}")
if not flat:
    print(f"mean |a_s0a − a_base| = {gap['total'] / steps:.4f}")
    for k in blocks:
        print(f"  with GT {k:>5}: {gap[k] / steps:.4f}")
env.close()
app.close()
