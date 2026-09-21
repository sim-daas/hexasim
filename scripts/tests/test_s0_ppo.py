# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge-gating checks for S0 (paradigm spec §4.3 item 4, §8.2). No simulator needed.

1. Warm start: widened S0a actor commands π_base's joint targets over 10,000 random observations, max |Δ| < 1e-6
   (rest-pose shift folded in).
2. Normaliser fold: swapping in re-estimated statistics leaves π(raw obs) unchanged.
3. Value-loss backward leaves every actor-side grad None.
4. A frozen-actor S0PPO update changes the critic and not the actor; std floor holds.
5. π_S0a → S0b widening (twist_cmd appended): widened actor == π_S0a over 10,000 random observations, < 1e-6.

Usage: python scripts/tests/test_s0_ppo.py [--ckpt <π_base.pt>] [--s0a_ckpt <π_S0a.pt>]
"""

import argparse
import copy
import math

import torch
from hexasim.rl.s0_ppo import (
    BASE_TO_S0A,
    S0A_DIM,
    S0B_DIM,
    S0PPO,
    fold_normalizer,
    warmstart_actor_state,
    widen_actor_state,
)
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt")
parser.add_argument(
    "--s0a_ckpt", default="logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt"
)
args = parser.parse_args()
torch.manual_seed(0)
N = 10_000


def actor(dim):
    obs = TensorDict({"policy": torch.zeros(1, dim)}, batch_size=[1])
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0}
    return MLPModel(obs, {"actor": ["policy"]}, "actor", 18, [256, 128, 64], "elu", True, dist)


base_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
base = actor(75)
base.load_state_dict(base_state)
# Rest pose moves tibia −30° → +30° (Δ = +60°), as in Hexapod-S0a-v0.
delta = torch.tensor([0.0] * 12 + [math.radians(60.0)] * 6)
new = actor(S0A_DIM)
new.load_state_dict(warmstart_actor_state(base_state, 0.05, delta))

# 1. warm-start equivalence; new-only dims get arbitrary values. Float64, so the check measures the
#    mapping rather than fp32 accumulation order (fp32 gives ~4e-6 from summation order alone).
base.double()
new.double()
x_old = (base_state["obs_normalizer._mean"] + 3 * base_state["obs_normalizer._std"] * torch.randn(N, 75)).double()
x_new = 5 * torch.randn(N, S0A_DIM, dtype=torch.float64)
for old, dst in BASE_TO_S0A:
    x_new[:, dst] = x_old[:, old]
d64 = delta.double()
x_new[:, 0:18] -= d64  # q relative to the new rest pose
x_new[:, 54:72] -= d64 / 0.5  # raw actions re-centred on the new rest pose
with torch.no_grad():
    y_old = base(TensorDict({"policy": x_old}, batch_size=[N]))
    y_new = new(TensorDict({"policy": x_new}, batch_size=[N]))
# compare joint targets (rest + 0.5·a), not raw actions
d1 = ((y_old - d64 / 0.5) - y_new).abs().max().item()
assert d1 < 1e-6, d1
assert new.distribution.std_param.min().item() >= 0.05

# 2. normaliser fold is function-preserving
shadow = copy.deepcopy(new.obs_normalizer)
shadow._mean += torch.randn_like(shadow._mean)
shadow._std *= torch.rand_like(shadow._std) + 0.5
fold_normalizer(new.mlp[0], new.obs_normalizer, shadow)
new.obs_normalizer.load_state_dict(shadow.state_dict())
with torch.no_grad():
    y_fold = new(TensorDict({"policy": x_new}, batch_size=[N]))  # still float64 here
d2 = (y_fold - y_new).abs().max().item()
assert d2 < 1e-6, d2

new.float()

# 3 + 4. separate critic, frozen-actor update
obs = TensorDict({"policy": torch.randn(64, S0A_DIM), "critic": torch.randn(64, 22)}, batch_size=[64])
groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
a = actor(S0A_DIM)
a.load_state_dict(warmstart_actor_state(base_state, 0.05, delta))
c = MLPModel(obs, groups, "critic", 1, [512, 256, 128], "elu", True)
storage = RolloutStorage("rl", 64, 8, obs, [18], "cpu")
alg = S0PPO(a, c, storage, schedule="fixed", entropy_coef=0.0005, num_mini_batches=2)
c(obs).pow(2).mean().backward()
assert all(p.grad is None for p in a.parameters()), "value loss reached the actor"
c.zero_grad(set_to_none=True)

actor_before = copy.deepcopy(a.state_dict())
critic_before = copy.deepcopy(c.state_dict())
for _ in range(8):
    o = TensorDict({"policy": torch.randn(64, S0A_DIM), "critic": torch.randn(64, 22)}, batch_size=[64])
    alg.act(o)
    alg.process_env_step(o, torch.randn(64), torch.zeros(64, dtype=torch.bool), {})
alg.compute_returns(o)
alg.update()
assert all(torch.equal(v, actor_before[k]) for k, v in a.state_dict().items() if "normalizer" not in k), "actor moved"
assert any(not torch.equal(v, critic_before[k]) for k, v in c.state_dict().items()), "critic did not train"

# 5. S0a → S0b widening
s0a_state = torch.load(args.s0a_ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
a_old, a_new = actor(S0A_DIM), actor(S0B_DIM)
a_old.load_state_dict(s0a_state)
a_new.load_state_dict(widen_actor_state(s0a_state, S0B_DIM, 0.05))
a_old.double()
a_new.double()
x = (s0a_state["obs_normalizer._mean"] + 3 * s0a_state["obs_normalizer._std"] * torch.randn(N, S0A_DIM)).double()
x_cmd = torch.cat([x, 5 * torch.randn(N, S0B_DIM - S0A_DIM, dtype=torch.float64)], dim=1)
with torch.no_grad():
    d5 = (a_old(TensorDict({"policy": x}, [N])) - a_new(TensorDict({"policy": x_cmd}, [N]))).abs().max().item()
assert d5 < 1e-6, d5
assert torch.equal(a_new.distribution.std_param, a_old.distribution.std_param.clamp(min=0.05))
print(f"S0 checks OK (float64): warm-start max|Δ| = {d1:.2e}, fold max|Δ| = {d2:.2e}, S0a→S0b max|Δ| = {d5:.2e}")
