# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S0 reconditioning PPO (paradigm spec §4.2, §4.3, §8.1) and the π_base → S0a → S0b warm starts.

On top of rsl_rl's PPO (separate actor and critic networks, each with its own normaliser):

* separate learning rates: actor fixed 1e-4, critic 3e-4 (schedule "fixed");
* actor frozen (no gradient) until the critic has trained ≥ ``actor_freeze_iters`` iterations **and**
  reached explained variance ≥ ``actor_freeze_ev`` — whichever is later;
* actor normaliser re-estimated over the first ``norm_burnin_iters`` iterations in a shadow copy, then
  swapped in with the change folded into the first layer, so re-estimation leaves π(raw obs) unchanged;
  afterwards its count is capped at the burn-in count (a slow EMA instead of a frozen mean);
* action std floored at ``std_floor`` (S0 only; π_base's femur/tibia std sits inside the backlash band);
* the env-owned V_0 estimator is saved and loaded with the checkpoint.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.algorithms import PPO

# π_base 75-D layout (Hexapod-Flat-v0) → S0a 114-D layout (Hexapod-S0a-v0), as (old slice, new slice).
# New-only blocks — tracking error [36:54], IMU accel [72:75], mask [93:111] — get zero first-layer columns.
BASE_TO_S0A = [
    (slice(9, 27), slice(0, 18)),  # joint pos rel
    (slice(27, 45), slice(18, 36)),  # joint vel (PhysX → two-sample difference)
    (slice(45, 63), slice(54, 72)),  # last action
    (slice(3, 6), slice(75, 78)),  # base ang vel → IMU gyro
    (slice(6, 9), slice(78, 81)),  # projected gravity
    (slice(63, 75), slice(81, 93)),  # CPG clock
    (slice(0, 3), slice(111, 114)),  # base lin vel → v_in (GT → V_0 ramp)
]
S0A_DIM = 114


def warmstart_actor_state(
    base_actor: dict, std_floor: float, rest_delta: torch.Tensor, action_scale: float = 0.5
) -> dict:
    """Map π_base's 75-D actor state dict onto the 114-D S0a layout, function-preserving on the mapped dims.

    ``rest_delta`` (18) = new rest pose − π_base's rest pose. It is folded in so the widened actor commands
    exactly π_base's joint targets: joint-position inputs are relative to the new pose (mean − Δ), last
    actions and outputs are shifted by Δ/scale (target = rest + scale·a).
    """
    new = copy.deepcopy(base_actor)
    w_old = base_actor["mlp.0.weight"]
    w = torch.zeros(w_old.shape[0], S0A_DIM, dtype=w_old.dtype)
    mean = torch.zeros(1, S0A_DIM)
    var = torch.ones(1, S0A_DIM)
    for old, dst in BASE_TO_S0A:
        w[:, dst] = w_old[:, old]
        mean[:, dst] = base_actor["obs_normalizer._mean"][:, old]
        var[:, dst] = base_actor["obs_normalizer._var"][:, old]
    mean[:, 0:18] -= rest_delta
    mean[:, 54:72] -= rest_delta / action_scale
    last = max(int(k.split(".")[1]) for k in base_actor if k.startswith("mlp.") and k.endswith(".bias"))
    new[f"mlp.{last}.bias"] = base_actor[f"mlp.{last}.bias"] - rest_delta / action_scale
    new["mlp.0.weight"] = w
    new["obs_normalizer._mean"] = mean
    new["obs_normalizer._var"] = var
    new["obs_normalizer._std"] = var.sqrt()
    new["distribution.std_param"] = base_actor["distribution.std_param"].clamp(min=std_floor)
    return new


S0B_DIM = 117  # S0a layout ⊕ twist_cmd (3)


def widen_actor_state(actor: dict, new_dim: int, std_floor: float) -> dict:
    """Append new trailing inputs to an actor state dict, function-preserving (spec §4.3): zero first-layer columns,
    identity normaliser statistics for the new dims (re-estimated in the burn-in), std carried and floored."""
    new = copy.deepcopy(actor)
    w = actor["mlp.0.weight"]
    extra = new_dim - w.shape[1]
    assert extra > 0, (w.shape, new_dim)
    new["mlp.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], extra)], dim=1)
    for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        k = f"obs_normalizer.{key}"
        new[k] = torch.cat([actor[k], torch.full((1, extra), fill, dtype=actor[k].dtype)], dim=1)
    new["distribution.std_param"] = actor["distribution.std_param"].clamp(min=std_floor)
    return new


def fold_normalizer(first_layer: torch.nn.Linear, old, new) -> None:
    """Rewrite ``first_layer`` so that layer(norm_new(x)) == layer(norm_old(x)) for every x.

    rsl_rl normalises z = (x − μ)/(σ + ε); with s = (σ' + ε)/(σ + ε), d = (μ' − μ)/(σ + ε):
    W z + b = (W·diag(s)) z' + (b + W d).
    """
    eps = old.eps
    s = (new._std + eps) / (old._std + eps)
    d = (new._mean - old._mean) / (old._std + eps)
    with torch.no_grad():
        first_layer.bias += (first_layer.weight * d).sum(dim=1)
        first_layer.weight *= s


class S0PPO(PPO):
    def __init__(
        self,
        actor,
        critic,
        storage,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        std_floor: float = 0.05,
        norm_burnin_iters: int = 50,
        actor_freeze_iters: int = 300,
        actor_freeze_ev: float = 0.8,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        assert self.schedule == "fixed", "S0 uses fixed learning rates (spec §8.1)"
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_learning_rate},
                {"params": self.critic.parameters(), "lr": critic_learning_rate},
            ]
        )
        self.std_floor = std_floor
        self.norm_burnin_iters = norm_burnin_iters
        self.actor_freeze_iters = actor_freeze_iters
        self.actor_freeze_ev = actor_freeze_ev
        self.iteration = 0
        self.actor_frozen = True
        self.actor.requires_grad_(False)
        self._shadow = copy.deepcopy(self.actor.obs_normalizer)
        self._shadow.count.zero_()
        self.actor.obs_normalizer.until = 0  # frozen during burn-in; the shadow learns instead
        self._count_cap = None
        self.env = None

    @staticmethod
    def construct_algorithm(obs, env, cfg, device) -> S0PPO:
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg.env = env.unwrapped
        return alg

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self.iteration < self.norm_burnin_iters:
            self._shadow.update(torch.cat([obs[g] for g in self.actor.obs_groups], dim=-1))
        super().process_env_step(obs, rewards, dones, extras)
        if self._count_cap is not None:
            self.actor.obs_normalizer.count.clamp_(max=self._count_cap)

    def update(self) -> dict[str, float]:
        st = self.storage
        ev = (1.0 - (st.returns - st.values).var() / st.returns.var().clamp(min=1e-8)).item()
        if self.actor_frozen and self.iteration >= self.actor_freeze_iters and ev >= self.actor_freeze_ev:
            self.actor_frozen = False
            self.actor.requires_grad_(True)
        loss = super().update()
        with torch.no_grad():
            self.actor.distribution.std_param.clamp_(min=self.std_floor)
        self.iteration += 1
        if self.iteration == self.norm_burnin_iters:
            fold_normalizer(self.actor.mlp[0], self.actor.obs_normalizer, self._shadow)
            self.actor.obs_normalizer.load_state_dict(self._shadow.state_dict())
            self.actor.obs_normalizer.until = None
            self._count_cap = int(self._shadow.count.item())
        loss.update(
            {
                "explained_variance": ev,
                "actor_frozen": float(self.actor_frozen),
                "std_min": self.actor.distribution.std_param.min().item(),
            }
        )
        return loss

    def save(self) -> dict:
        saved = super().save()
        saved["v0_state_dict"] = self.env.v0.state_dict()
        saved["v0_optimizer_state_dict"] = self.env.v0_opt.state_dict()
        saved["s0_state"] = {
            "iteration": self.iteration,
            "actor_frozen": self.actor_frozen,
            "count_cap": self._count_cap,
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        ret = super().load(loaded_dict, load_cfg, strict)
        self.env.v0.load_state_dict(loaded_dict["v0_state_dict"])
        self.env.v0_opt.load_state_dict(loaded_dict["v0_optimizer_state_dict"])
        s = loaded_dict["s0_state"]
        self.iteration, self.actor_frozen, self._count_cap = s["iteration"], s["actor_frozen"], s["count_cap"]
        self.actor.requires_grad_(not self.actor_frozen)
        if self.iteration >= self.norm_burnin_iters:
            self.actor.obs_normalizer.until = None
        return ret
