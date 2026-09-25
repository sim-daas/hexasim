# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feasibility spike (S1 plan §1, option A): per-step, per-foot friction through the PhysX tensor API.

Question: can a spatial friction field M(x, y) be realised by writing, every control step, the friction of each
tibia's collision shape(s) from the material under that toe? Valid only if (1) writes are per shape (per env, per
foot), (2) a write takes effect on an existing contact, (3) the USD-bound compliant-contact parameters survive a
friction write, (4) the cost is small against env.step at the training env count, (5) continuous values do not hit a
PhysX material-count limit.

Tests (flat terrain μ 1, combine multiply, zero actions, standing command):
  T1  gravity tilted 30°: even envs' feet written μ 0.2, odd envs' 1.0 → even slide, odd hold.
  T2  same tilt, all 1.0 for the settle, then half switched to 0.2 mid-contact → they start sliding; switched back
      to 1.0 → they stop.
  T3  θ = 0: toe sink of compliant vs rigid bucket envs before and after a friction write (compliance survives).
  T4  timing: env.step alone vs env.step + a per-step write of fresh continuous random values (N steps).

Usage: python scripts/tests/material_spike.py --headless [--num_envs 256] [--timing_envs 4096]
"""

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timing_envs", type=int, default=4096)
parser.add_argument("--timing_steps", type=int, default=200)
parser.add_argument("--t6_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
import warp as wp  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

from isaaclab.terrains import MeshPlaneTerrainCfg  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def make(theta_deg, num_envs, ground=None):
    cfg = parse_env_cfg("Hexapod-S0b-v0", num_envs=num_envs)
    if ground is not None:
        cfg.scene.terrain.physics_material = ground
    cfg.seed = 0
    cfg.scene.terrain.terrain_generator.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
    cfg.curriculum = None
    cfg.events.reset_robot_base.params["pose_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    th = math.radians(theta_deg)
    cfg.sim.gravity = (9.81 * math.sin(th), 0.0, -9.81 * math.cos(th))
    env = gym.make("Hexapod-S0b-v0", cfg=cfg).unwrapped
    env.reset()
    return env


class FootFriction:
    """Writes (μ_s, μ_d) per env per foot to the tibia shapes; restitution kept as read."""

    def __init__(self, env):
        self.robot = env.scene["robot"]
        view = self.robot.root_view
        paths = view.link_paths[0]
        per_body = [self.robot._physics_sim_view.create_rigid_body_view(p).max_shapes for p in paths]
        assert sum(per_body) == view.max_shapes, (per_body, view.max_shapes)
        starts = [sum(per_body[:i]) for i in range(len(per_body))]
        names = [p.split("/")[-1] for p in paths]
        # Shape columns per foot, in the env's tibia body order (LEG_NAMES, asserted at env build).
        tib_names = [self.robot.body_names[i] for i in env._tibia_body_ids]
        self.cols = []
        for tn in tib_names:
            j = names.index(tn)
            self.cols.append(list(range(starts[j], starts[j] + per_body[j])))
        print(f"shapes: total {view.max_shapes}, per tibia {[len(c) for c in self.cols]}")
        self.mat = wp.to_torch(view.get_material_properties()).clone()  # (N, S, 3) on CPU
        self.ids = torch.arange(env.num_envs, dtype=torch.int32)
        self.view = view

    def write(self, mu_s, mu_d):
        """mu_s, mu_d: (N, 6) tensors (any device)."""
        mu_s, mu_d = mu_s.cpu(), mu_d.cpu()
        for leg, cols in enumerate(self.cols):
            self.mat[:, cols, 0] = mu_s[:, leg : leg + 1]
            self.mat[:, cols, 1] = mu_d[:, leg : leg + 1]
        self.view.set_material_properties(wp.from_torch(self.mat, dtype=wp.float32), wp.from_torch(self.ids))

    def write_ids(self, ids, mu_s, mu_d):
        for leg, cols in enumerate(self.cols):
            self.mat[ids[:, None], cols, 0] = mu_s[:, leg : leg + 1]
            self.mat[ids[:, None], cols, 1] = mu_d[:, leg : leg + 1]
        self.view.set_material_properties(wp.from_torch(self.mat, dtype=wp.float32), wp.from_torch(ids.to(torch.int32)))

    def read(self):
        return wp.to_torch(self.view.get_material_properties())


def steps(env, n):
    zero = torch.zeros(env.num_envs, 18, device=env.device)
    with torch.inference_mode():
        for _ in range(n):
            env.step(zero)


def x(env):
    return env.scene["robot"].data.root_pos_w.torch[:, 0].clone()


ok = True
N = args.num_envs
if args.t6_only:
    # T6 — ground compliant (k_g, c_g) against feet with k written per step: does k follow the foot, does c_g act?
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

    even = torch.arange(N) % 2 == 0
    kf = torch.where(even, torch.tensor(2e3), torch.tensor(5e4))
    rows = []
    for c_g in (None, 10.0, 100.0, 1000.0):
        ground = PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            compliant_contact_stiffness=None if c_g is None else 1e6,
            compliant_contact_damping=c_g,
        )
        env = make(0.0, N, ground)
        ff = FootFriction(env)
        ff.mat[:, [c[0] for c in ff.cols], 2] = -kf[:, None]
        ff.view.set_material_properties(wp.from_torch(ff.mat, dtype=wp.float32), wp.from_torch(ff.ids))
        robot = env.scene["robot"]
        zero = torch.zeros(N, 18, device=env.device)
        vz = []
        with torch.inference_mode():
            for _ in range(60):
                env.step(zero)
                vz.append(robot.data.root_lin_vel_w.torch[:, 2].abs().cpu())
        vz = torch.stack(vz)
        toe = toe_pos_w(robot, env._tibia_body_ids)
        sk = (-(toe[..., 2] - env.terrain_z(toe[..., :2])).mean(dim=1)).cpu()
        up = robot.data.projected_gravity_b.torch[:, 2].cpu()
        rows.append(
            (
                c_g,
                sk[even].mean() * 1e3,
                sk[~even].mean() * 1e3,
                vz[3:30].pow(2).mean().sqrt(),
                vz[30:].max(),
                (up < -0.8).float().mean(),
            )
        )
        env.close()
    for c_g, s2, s5, rms, late, upr in rows:
        print(
            f"T6 ground c_g {c_g}: sink foot k2e3 {s2:.2f} mm / k5e4 {s5:.2f} mm; landing |v_z| rms {rms:.4f}, "
            f"late max {late:.4f}; upright {upr:.2f}"
        )
    app.close()
    raise SystemExit

even = torch.arange(N) % 2 == 0
lo, hi = torch.full((N, 6), 0.2), torch.full((N, 6), 1.0)
tan30 = math.tan(math.radians(30))

# T1 — per-shape writes before the robot lands.
env = make(30.0, N)
ff = FootFriction(env)
ff.write(torch.where(even[:, None], lo, hi), torch.where(even[:, None], lo, hi))
back = ff.read()
c0 = ff.cols[0][0]
print(f"T1 read-back env0 foot0 {back[0, c0].tolist()}, env1 foot0 {back[1, c0].tolist()}")
steps(env, 25)
x0 = x(env)
steps(env, 100)
d = (x(env) - x0).abs().cpu()
t1 = bool((d[even] > 0.05).all() and (d[~even] < 0.02).all())
print(
    f"T1 (tan30 {tan30:.3f}): μ0.2 envs drift min {d[even].min():.3f} m; μ1.0 envs drift max {d[~even].max():.4f} m"
    f" -> {'PASS' if t1 else 'FAIL'}"
)
ok &= t1

env.close()

# T2 — mid-contact switch on a fresh env: hold at 1.0, switch even envs to 0.2, then back to 1.0.
env = make(30.0, N)
ff = FootFriction(env)
ff.write(hi, hi)
steps(env, 100)
xa = x(env)
steps(env, 50)
held = (x(env) - xa).abs().cpu()
ff.write(torch.where(even[:, None], lo, hi), torch.where(even[:, None], lo, hi))
xb = x(env)
steps(env, 10)
slid = (x(env) - xb).abs().cpu()
ff.write(hi, hi)
steps(env, 25)  # decelerate (after 10 steps at 0.2 the robot moves ≈ 0.3 m/s)
xc = x(env)
steps(env, 50)
after = (x(env) - xc).abs().cpu()
t2 = bool(held.max() < 0.02 and (slid[even] > 0.01).all() and (slid[~even] < 0.02).all() and after.max() < 0.02)
print(
    f"T2 held max {held.max():.4f}; after switch μ0.2 min {slid[even].min():.3f} / μ1.0 max {slid[~even].max():.4f};"
    f" switched back max {after.max():.4f} -> {'PASS' if t2 else 'FAIL'}"
)
big = after > 0.02
print(
    f"   T2 after-switch-back movers: {big.sum().item()}"
    f" (even {big[even].sum().item()}, odd {big[~even].sum().item()});"
    f" their slide during μ0.2: {slid[big].tolist()[:8]}; after: {after[big].tolist()[:8]}"
)
up = env.scene["robot"].data.projected_gravity_b.torch[:, 2].cpu()
print(f"   T2 upright (g_b,z < −0.8): {(up < -0.8).sum().item()}/{N}; movers' g_b,z {up[big].tolist()[:8]}")
ok &= t2
env.close()

# T3 — compliance survives a friction write.
env = make(0.0, N)
ff = FootFriction(env)
robot = env.scene["robot"]
log_k = env.foot_material[:, 3].cpu()
rigid = log_k == 0


def sink():
    toe = toe_pos_w(robot, env._tibia_body_ids)
    return (-(toe[..., 2] - env.terrain_z(toe[..., :2])).mean(dim=1)).cpu()


steps(env, 100)
s_before = sink()
ff.write(torch.full((N, 6), 0.9), torch.full((N, 6), 0.8))
steps(env, 100)
s_after = sink()
gap_b = (s_before[~rigid].mean() - s_before[rigid].mean()) * 1e3
gap_a = (s_after[~rigid].mean() - s_after[rigid].mean()) * 1e3
t3 = bool(gap_a > 0.2 and abs(gap_a - gap_b) < 0.25 * gap_b)
print(f"T3 compliant − rigid sink: before {gap_b:.2f} mm, after write {gap_a:.2f} mm -> {'PASS' if t3 else 'FAIL'}")
ok &= t3

# T5 — compliant stiffness through the restitution slot (PhysX: restitution < 0 ⇒ compliant, k = −restitution).
ks = torch.where(even, torch.tensor(2e3), torch.tensor(5e4))
ff.mat[:, [c[0] for c in ff.cols], 2] = -ks[:, None]
ff.view.set_material_properties(wp.from_torch(ff.mat, dtype=wp.float32), wp.from_torch(ff.ids))
steps(env, 100)
s5 = sink()
base_rigid = s_after[rigid].mean()
for name, m in (("was rigid", rigid), ("was compliant", ~rigid)):
    print(
        f"T5 {name}: sink k=2e3 {s5[even & m].mean() * 1e3:.2f} mm, k=5e4 {s5[~even & m].mean() * 1e3:.2f} mm"
        f" (rigid ref {base_rigid * 1e3:.2f} mm);"
        f" max |root v| {env.scene['robot'].data.root_lin_vel_w.torch.norm(dim=-1).max():.3f}"
    )
env.close()

# T4 — timing at the training env count, continuous random values every step.
env = make(0.0, args.timing_envs)
ff = FootFriction(env)
steps(env, 20)
n = args.timing_steps
torch.cuda.synchronize()
t0 = time.perf_counter()
steps(env, n)
torch.cuda.synchronize()
t_step = (time.perf_counter() - t0) / n
zero = torch.zeros(env.num_envs, 18, device=env.device)
t_write = 0.0
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(n):
        mu_s = torch.rand(env.num_envs, 6, device=env.device) * 0.8 + 0.3
        tw = time.perf_counter()
        ff.write(mu_s, mu_s * 0.9)
        t_write += time.perf_counter() - tw
        env.step(zero)
torch.cuda.synchronize()
t_both = (time.perf_counter() - t0) / n
chk = ff.read()


def timed(label, ids_fn, value_fn):
    tw_sum = 0.0
    with torch.inference_mode():
        for _ in range(n):
            ids = ids_fn()
            mu = value_fn(len(ids))
            tw = time.perf_counter()
            ff.write_ids(ids, mu, mu * 0.9)
            tw_sum += time.perf_counter() - tw
    print(f"T4 {label}: write call {tw_sum / n * 1e3:.2f} ms")


E = env.num_envs
cont = lambda k: torch.rand(k, 6) * 0.8 + 0.3  # noqa: E731
quant = lambda k: (torch.randint(0, 32, (k, 6)) / 31.0) * 0.8 + 0.3  # noqa: E731
sub = lambda: torch.randperm(E)[:400]  # noqa: E731
timed("all envs, continuous", lambda: torch.arange(E), cont)
timed("all envs, 32 levels", lambda: torch.arange(E), quant)
timed("400 envs, continuous", sub, cont)
timed("400 envs, 32 levels", sub, quant)
timed("40 envs, continuous", lambda: torch.randperm(E)[:40], cont)
print(
    f"T4 {args.timing_envs} envs: env.step {t_step * 1e3:.2f} ms; with per-step write {t_both * 1e3:.2f} ms"
    f" (write call {t_write / n * 1e3:.2f} ms, +{(t_both / t_step - 1) * 100:.1f} %);"
    f" read-back matches last write: {torch.allclose(chk[:, ff.cols[0][0], 0], ff.mat[:, ff.cols[0][0], 0])}"
)
env.close()
print("MATERIAL_SPIKE", "PASS" if ok else "FAIL")
app.close()
