# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge-gating checks for S1 (paradigm spec §4.3, §8.2, §9.1; phase2-progress §2.2). No simulator needed.

1. Widening π_S0 117 → 1,125: widened actor == π_S0 over 10,000 random observations with random values in every new
   block, max |Δ| < 1e-6 (float64); std carried at the floor.
2. Value-loss backward leaves every actor-side grad None (separate critic on sg[actor obs] ⊕ P).
3. PPO backward leaves every aux (E_p, V, probe) grad None: latents enter the observation under no_grad.
4. Aux-loss backward leaves every actor and critic grad None.
5. With ê gated to zero, its first-layer columns receive exactly zero gradient.
6. TCN receptive-field assert: RF ≥ W holds for W ∈ {25, 50, 75} and fires for a too-short stack.
7. β-NLL mean gradient ∝ −(y − μ)/σ (β = 0.5), not /σ².
8. Sequence buffer: history never crosses an episode boundary; touchdown targets are the first touchdown strictly
   after t, within the episode; Δt^TD = (s − t)·dt.
9. Layout mirror (``mdp.symmetry`` self-check): involution on the 1,125-D, 683-D and latent groups.

Usage: python scripts/tests/test_s1.py [--s0_ckpt <π_S0.pt>]
"""

import argparse

import torch
from hexasim.rl.s0_ppo import widen_actor_state
from hexasim.rl.s1.aux import CUR_DIM, FRAME, TD_EVENT, AuxCfg, S1Aux
from hexasim.rl.s1.losses import beta_nll
from hexasim.rl.s1.seq_buffer import SequenceBuffer
from hexasim.rl.s1.tcn import ProprioTCN
from hexasim.tasks.manager_based.hexasim.mdp import layout
from rsl_rl.models import MLPModel
from tensordict import TensorDict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--s0_ckpt",
    default="logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt",
)
args = parser.parse_args()
torch.manual_seed(0)
N = 10_000
A, P = layout.ACTOR_DIM, layout.CONTRACT_P_DIM


def actor(dim):
    obs = TensorDict({"policy": torch.zeros(1, dim)}, batch_size=[1])
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0}
    return MLPModel(obs, {"actor": ["policy"]}, "actor", 18, [256, 128, 64], "elu", True, dist)


# 1. widening
s0 = torch.load(args.s0_ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
old, new = actor(layout.OLD_ACTOR_DIM), actor(A)
old.load_state_dict(s0)
new.load_state_dict(widen_actor_state(s0, A, 0.05))
old.double()
new.double()
x = (s0["obs_normalizer._mean"] + 3 * s0["obs_normalizer._std"] * torch.randn(N, layout.OLD_ACTOR_DIM)).double()
x_new = torch.cat([x, 5 * torch.randn(N, A - layout.OLD_ACTOR_DIM, dtype=torch.float64)], dim=1)
with torch.no_grad():
    d1 = (old(TensorDict({"policy": x}, [N])) - new(TensorDict({"policy": x_new}, [N]))).abs().max().item()
assert d1 < 1e-6, d1
assert torch.equal(new.distribution.std_param, old.distribution.std_param.clamp(min=0.05))
new.float()

# 2–5. gradient routing
B = 32
aux = S1Aux(AuxCfg(batch=B, buffer_len=64, window=25), 4, torch.zeros(18), 0.02, "cpu")
hist = torch.randn(B, 25, FRAME)
lat = aux.latents(hist)  # what the env does: no_grad
obs_pol = torch.zeros(B, A)
obs_pol[:, : layout.OLD_ACTOR_DIM] = torch.randn(B, layout.OLD_ACTOR_DIM)
obs_pol[:, layout.ACTOR["z_p"]] = lat[:, :64]  # ê gated shut: stays zero
obs = TensorDict({"policy": obs_pol, "contract": torch.randn(B, P)}, batch_size=[B])
a = new
c = MLPModel(obs, {"actor": ["policy"], "critic": ["policy", "contract"]}, "critic", 1, [512, 256, 128], "elu", True)
c(obs).pow(2).mean().backward()
assert all(p.grad is None for p in a.parameters()), "value loss reached the actor"
a(obs).pow(2).mean().backward()  # surrogate stand-in: any function of the actor output
assert all(p.grad is None for p in aux.parameters()), "PPO loss reached the aux modules"
w_grad = a.mlp[0].weight.grad
assert w_grad[:, layout.ACTOR["e_hat"]].abs().max() == 0.0, "gated ê columns got gradient"
assert w_grad[:, layout.ACTOR["m_star"].start :].abs().max() == 0.0, "zero-held columns got gradient"
assert w_grad[:, layout.ACTOR["z_p"]].abs().max() > 0.0, "z_p columns get no gradient"
a.zero_grad(set_to_none=True)
c.zero_grad(set_to_none=True)
b = {
    "hist": hist,
    "cur": torch.rand(B, CUR_DIM),
    "td_pos_b": torch.randn(B, 6, 3),
    "td_dt": torch.rand(B, 6),
    "td_event": torch.randn(B, 6, TD_EVENT),
    "td_valid": torch.rand(B, 6) > 0.3,
}
b["cur"][:, :6] = (b["cur"][:, :6] > 0.5).float()
aux.losses(b)["total"].backward()
assert all(p.grad is None for p in a.parameters()) and all(p.grad is None for p in c.parameters())
assert all(p.grad is not None for p in aux.encoder.parameters()), "aux loss does not train E_p"

# 6. TCN receptive field
for w, rf in ((25, 61), (50, 61), (75, 125)):
    assert ProprioTCN(window=w).rf == rf
try:
    ProprioTCN(window=50, dilations=(1, 2))
    raise AssertionError("RF assert did not fire")
except AssertionError as e:
    assert "receptive field" in str(e), e

# 7. β-NLL gradient scaling
mu = torch.zeros(4, requires_grad=True)
logvar = torch.log(torch.tensor([0.25, 1.0, 4.0, 9.0]))
y = torch.ones(4)
beta_nll(mu, logvar, y).sum().backward()
sigma = torch.exp(0.5 * logvar)
assert torch.allclose(mu.grad, -(y - 0) / sigma, atol=1e-6), (mu.grad, sigma)

# 8. sequence buffer semantics (1 env, W 3, L 16, dt 0.02)
buf = SequenceBuffer(1, 16, 1, 1, 1, 3, "cpu")
pose = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]])
td_at = {4, 9}  # leg 0 touches down at steps 4 and 9; episode ends after step 11
for t in range(14):
    buf.add(torch.tensor([[float(t)]]), torch.tensor([[float(t)]]), pose)
    legs = torch.zeros(1, 6, dtype=torch.bool)
    legs[0, 0] = t in td_at
    buf.touchdown(legs, torch.full((1, 6, 3), float(t)), torch.full((1, 6, 1), float(t)))
    if t == 11:
        buf.episode_end(torch.tensor([0]))
valid = buf.valid_mask()[0]
steps = buf.row_step[0]
for s_ in range(16):
    t = int(steps[s_])
    if t < 0:
        continue
    # history within one episode: rows 12, 13 (new episode) lack 3 same-episode frames
    assert bool(valid[s_]) == (t >= 2 and not (12 <= t <= 13)), (t, valid[s_])
    want = 4 if t < 4 else 9 if t < 9 else None
    assert bool(buf.td_valid[0, s_, 0]) == (want is not None), (t, buf.td_valid[0, s_, 0])
    if want is not None:
        assert buf.td_step[0, s_, 0] == want and buf.td_event[0, s_, 0, 0] == want
out = buf.sample(64, 0.02)
ok_rows = out["td_valid"][:, 0]
t_rows = out["hist"][:, -1, 0]
exp_dt = torch.where(t_rows < 4, 4 - t_rows, 9 - t_rows) * 0.02
assert torch.allclose(out["td_dt"][ok_rows, 0], exp_dt[ok_rows])
assert ((out["hist"][:, 1:, 0] - out["hist"][:, :-1, 0]) == 1).all(), "history not contiguous"

# 10. normaliser block writes after rollout-time (inference-mode) updates — the ê-gate crash of the first pilot
from hexasim.rl.s1_ppo import set_block_stats  # noqa: E402

norm = MLPModel(
    TensorDict({"policy": torch.zeros(1, A)}, [1]), {"actor": ["policy"]}, "actor", 18, [8], "elu", True
).obs_normalizer
with torch.inference_mode():
    norm.update(torch.randn(64, A))
set_block_stats(norm, layout.ACTOR["e_hat"], torch.full((15,), 2.0), torch.full((15,), 4.0))
assert torch.equal(norm._std[0, layout.ACTOR["e_hat"]], torch.full((15,), 2.0))

# 9. mirror self-check
import runpy  # noqa: E402

runpy.run_module("hexasim.tasks.manager_based.hexasim.mdp.symmetry", run_name="__main__")
print(f"S1 checks OK: widening max|Δ| = {d1:.2e} (float64); routing, TCN RF, β-NLL, buffer, mirror")
