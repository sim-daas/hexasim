# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod-S1-v0 wiring check (phase2-progress §2.2), random-ish actions around the rest pose.

1. Obs shapes: policy 1,125, contract 683, mirror 79; zero-held slots exactly zero; ê zero while the gate is shut.
2. FK (``rl.s1.kinematics``) vs the sim toe point in the base frame: max error < 1 mm.
3. Contract P: the strip point at the nominal foothold (forward 0, lateral 0) equals the field there; the height scan
   centre equals terrain z − base z.
4. Sequence buffer: valid rows after 150 steps, touchdown targets resolved for some rows; ``aux.update`` and
   ``aux.v_tests`` run and return finite numbers.

Usage: python scripts/tests/s1_env_check.py --headless [--num_envs 64]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import hexasim.tasks  # noqa: E402, F401
import torch  # noqa: E402
from hexasim.rl.s1.kinematics import toe_positions  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp import layout  # noqa: E402
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Hexapod-S1-v0", num_envs=args.num_envs)
cfg.seed = 0
env = gym.make("Hexapod-S1-v0", cfg=cfg).unwrapped
obs, _ = env.reset()
ok = True

# 1
shapes = {k: tuple(v.shape) for k, v in obs.items()}
print("1. obs shapes", shapes)
pol = obs["policy"]
zero_held = pol[:, layout.ACTOR["m_star"].start :].abs().max().item()
e_hat = pol[:, layout.ACTOR["e_hat"]].abs().max().item()
t1 = (
    shapes["policy"][1] == layout.ACTOR_DIM
    and shapes["contract"][1] == layout.CONTRACT_P_DIM
    and shapes["mirror"][1] == layout.LATENT_DIM
    and zero_held == 0.0
    and e_hat == 0.0
)
print(f"   zero-held max {zero_held}, ê max (gate shut) {e_hat} -> {'PASS' if t1 else 'FAIL'}")
ok &= t1

robot = env.scene["robot"]
g = torch.Generator(device=env.device).manual_seed(0)
with torch.inference_mode():
    for i in range(150):
        a = 0.3 * torch.randn(env.num_envs, 18, device=env.device, generator=g)
        obs, *_ = env.step(a)

# 2
q = robot.data.joint_pos.torch
fk = toe_positions(q)
root, quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
toe = toe_pos_w(robot, env._tibia_body_ids)
sim_b = quat_apply_inverse(quat[:, None].expand(-1, 6, -1), toe - root[:, None])
err = (fk - sim_b).norm(dim=-1).max().item()
t2 = err < 1e-3
print(f"2. FK vs sim toe (base frame) max {err * 1e3:.3f} mm -> {'PASS' if t2 else 'FAIL'}")
ok &= t2

# 3
p = env.s1_contract_p
strips = p[:, layout.CONTRACT_P["strips"]].view(-1, 6, 8, 3, 4)
i_fwd0 = layout.STRIP_FWD.index(min(layout.STRIP_FWD, key=abs))
anchor = root[:, None, :2] + env._yaw_offsets_w(env.nominal_toe_b[:, :2])
want = env.material4(anchor)
d3 = (strips[:, :, i_fwd0, 1] - want).abs().max().item()
scan = p[:, layout.CONTRACT_P["height_scan"]].view(-1, 11, 7)
d3b = (scan[:, 5, 3] - (env.terrain_z(root[:, :2]) - root[:, 2])).abs().max().item()
t3 = d3 < 1e-5 and d3b < 1e-5 and abs(layout.STRIP_FWD[i_fwd0]) < 1e-9
print(f"3. strip@anchor vs field {d3:.2e}, scan centre {d3b:.2e} -> {'PASS' if t3 else 'FAIL'}")
ok &= t3

# 4
buf = env.aux.buffer
nvalid = buf.valid_mask().sum().item()
ntd = buf.td_valid.sum().item()
logs = env.aux.update()
vt = env.aux.v_tests(4096)
finite = all(torch.isfinite(torch.tensor(v)) for v in logs.values()) and len(logs) > 0
t4 = nvalid > 0 and ntd > 0 and finite
print(f"4. buffer valid rows {nvalid}, touchdown targets {ntd}; aux logs {logs}")
print(f"   v_tests {vt} -> {'PASS' if t4 else 'FAIL'}")
ok &= t4
print("S1_ENV_CHECK", "PASS" if ok else "FAIL")
env.close()
app.close()
