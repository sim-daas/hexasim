# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S1 PPO (paradigm spec §4.3, §6, §8; phase2-progress §2.2).

On top of :class:`S0PPO` (per-network LRs, actor frozen until EV ≥ 0.8 and ≥ 300 iterations, normaliser burn-in
fold, std floor):

* the env-owned auxiliary learner (``env.aux``: E_p, V, probes, VICReg) is trained once per iteration after PPO and
  saved with the checkpoint;
* **frozen normaliser dims**: V_0 (its π_S0 statistics, so its per-episode dropout does not rescale it — D23), the
  zero-held arm slots (identity, spec §8.4) and ê while its gate is shut. They are restored after every normaliser
  update, the burn-in shadow included, so the fold leaves them untouched;
* **ê gate**: every ``v_test_every`` iterations V is tested (v7 §4.3: slip-stratified error < ``slip_ratio_max`` ×
  rigid, zeroing the 9 IMU dims raises the error ≥ ``imu_zero_min`` ×). After ``v_test_passes`` consecutive passes,
  ê's normaliser statistics are set from aux samples while its zero-init columns are still exactly zero (function-
  preserving) and the gate opens (binary; PPO grows the columns from zero);
* **V_0 retirement**: per-episode dropout probability 0 → 1 over ``v0_retire_iters``, starting ``v0_retire_delay``
  iterations after the gate opened.

S2 branch (``clock_anneal_iters`` > 0, run loaded from an S1 checkpoint with ``train.py --branch_from``):

* **clock dropout**: per-episode probability 0 → 1 over ``clock_anneal_iters`` from the branch iteration; the clock
  observation and every clock-phase reward term leave together (``env.clock_live``, spec §5);
* **re-burn-in**: actor frozen for ``reburn_iters`` after the branch (spec stage table, S2 critic row);
* **widening**: the 1,125-D trunk is widened to the 1,269-D S2 layout (H* appended) for every arm
  (``widen_checkpoint``, function-preserving, spec §4.3);
* **oracle arms**: the arm's block (``layout.ARM_BLOCK``: O-mat M*, O-geo H*) goes live; its normaliser statistics are
  set from the first batch and it leaves the frozen set. Every other non-S1 block stays zero-held, and ``update``
  asserts its first-layer columns are still exactly zero (an arm must never read a block it is not given).
"""

from __future__ import annotations

import torch

from hexasim.tasks.manager_based.hexasim.mdp import layout

from .s0_ppo import S0PPO


def set_block_stats(norm, cols: slice, mean: torch.Tensor, var: torch.Tensor) -> None:
    """Overwrite a normaliser's statistics on ``cols``. The buffers are refreshed inside the rollout's inference mode,
    which makes them inference tensors: in-place writes must happen in inference mode too (the first pilot launch
    crashed here when the ê gate opened, phase2-progress §2.2.5)."""
    with torch.inference_mode():
        norm._mean[:, cols] = mean
        norm._var[:, cols] = var
        norm._std[:, cols] = var.sqrt()


def _pad(t: torch.Tensor, at: int, n: int, fill: float = 0.0) -> torch.Tensor:
    """Insert ``n`` columns of ``fill`` at column ``at`` (last dim)."""
    return torch.cat([t[..., :at], t.new_full((*t.shape[:-1], n), fill), t[..., at:]], dim=-1)


def widen_checkpoint(ck: dict, new_dim: int) -> dict:
    """Widen an S1 runner checkpoint's actor input to ``new_dim``, in place, function-preserving (spec §4.3): zero
    first-layer columns and identity normaliser statistics appended to the actor and inserted after the actor block of
    the critic input (sg[actor obs] ⊕ P); the matching Adam moments padded with zeros; the new columns join the frozen
    (zero-held) set."""
    a, c = ck["actor_state_dict"], ck["critic_state_dict"]
    at = a["mlp.0.weight"].shape[1]
    n = new_dim - at
    assert n > 0, (at, new_dim)
    first = {a["mlp.0.weight"].shape, c["mlp.0.weight"].shape}
    for sd in (a, c):
        sd["mlp.0.weight"] = _pad(sd["mlp.0.weight"], at, n)
        for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            sd[f"obs_normalizer.{key}"] = _pad(sd[f"obs_normalizer.{key}"], at, n, fill)
    padded = 0
    for st in ck["optimizer_state_dict"]["state"].values():
        if st["exp_avg"].shape in first:
            st["exp_avg"], st["exp_avg_sq"] = _pad(st["exp_avg"], at, n), _pad(st["exp_avg_sq"], at, n)
            padded += 1
    assert padded == 2, padded
    s = ck["s1_state"]
    s["frozen"] = torch.cat([s["frozen"], torch.arange(at, new_dim)])
    return ck


class S1PPO(S0PPO):
    def __init__(
        self,
        actor,
        critic,
        storage,
        v_test_every: int = 100,
        v_test_passes: int = 3,
        slip_ratio_max: float = 3.0,
        imu_zero_min: float = 1.5,
        v0_retire_delay: int = 1000,
        v0_retire_iters: int = 1000,
        clock_anneal_iters: int = 0,
        reburn_iters: int = 50,
        ramp_terms: tuple[str, ...] = (),
        ramp_iters: int = 0,
        ramp_to: float = 2.0,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        self.v_test_every, self.v_test_passes = v_test_every, v_test_passes
        self.slip_ratio_max, self.imu_zero_min = slip_ratio_max, imu_zero_min
        self.v0_retire_delay, self.v0_retire_iters = v0_retire_delay, v0_retire_iters
        self.clock_anneal_iters, self.reburn_iters = clock_anneal_iters, reburn_iters
        self.ramp_terms, self.ramp_iters, self.ramp_to = tuple(ramp_terms), ramp_iters, ramp_to
        self._ramp_base: dict[str, float] = {}
        self.s2_start = -1  # iteration the S2 branch started (−1 = S1)
        self._live_pending: list[slice] = []
        self.passes = 0
        self.gate_iter = -1  # iteration the ê gate opened (−1 = shut)
        self.last_v_tests: dict[str, float] = {}
        a = layout.ACTOR
        dims = list(range(a["v0"].start, a["v0"].stop)) + list(range(a["e_hat"].start, a["e_hat"].stop))
        self._width = self.actor.mlp[0].in_features  # 1,125 (S1) or 1,269 (S2, with H*)
        dims += list(range(a["m_star"].start, self._width))
        self._frozen = torch.tensor(dims, device=self.device)
        self._frozen_stats = None  # taken at the first env step, i.e. after the warm start loaded π_S0's statistics

    def _snapshot(self):
        n = self.actor.obs_normalizer
        self._frozen_stats = {k: getattr(n, k)[:, self._frozen].clone() for k in ("_mean", "_var", "_std")}

    def _restore(self, norm):
        with torch.inference_mode():
            for k, v in self._frozen_stats.items():
                getattr(norm, k)[:, self._frozen] = v

    def _ramp(self) -> float:
        """Redesign reward ramp (phase2-progress §2.3.15): the weights of ``ramp_terms`` go from their configured value
        to ``ramp_to`` × it over ``ramp_iters`` iterations. Keyed to the iteration, so a resume continues it."""
        rm = self.env.reward_manager
        if not self._ramp_base:
            self._ramp_base = {n: rm.get_term_cfg(n).weight for n in self.ramp_terms}
        f = 1.0 + (self.ramp_to - 1.0) * min(self.iteration / self.ramp_iters, 1.0)
        for n, w in self._ramp_base.items():
            rm.get_term_cfg(n).weight = w * f
        return f

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self._frozen_stats is None:
            # Fresh S1-stage start with an oracle arm (redesign: the block is live from π_S0 on): its statistics are
            # estimated by the normaliser burn-in like every live block, so it must not be in the frozen set. Nothing
            # else is needed here — the critic and the optimiser are new (no stale statistics or Adam moments, §2.3.9).
            block = layout.ARM_BLOCK[self.env.cfg.arm]
            if block and self.iteration == 0:
                cols = layout.ARM_ACTOR[block]
                self._frozen = self._frozen[(self._frozen < cols.start) | (self._frozen >= cols.stop)]
            self._snapshot()
        for cols in self._live_pending:  # blocks going live at the branch: statistics from this batch
            x = obs["policy"][:, cols]
            mean, var = x.mean(0), x.var(0).clamp(min=1e-4)
            set_block_stats(self.actor.obs_normalizer, cols, mean, var)
            # The critic sees the block too (sg[actor obs] comes first in its input). Its statistics there are those
            # of a constant zero (std 0 → gain 1/eps) under a count too large to move, and its first-layer columns
            # are untrained random initialisation: set the statistics and zero the columns (function-preserving), or
            # the live block enters the critic as noise × 100 (phase2-progress §2.3.9, O-mat smoke: EV 0.3).
            set_block_stats(self.critic.obs_normalizer, cols, mean, var)
            with torch.no_grad():
                self.critic.mlp[0].weight[:, cols] = 0.0
            # Adam: the block's columns have zero moments but share the first layer's step count, so there is no bias
            # correction and their first steps would be ≈ 3 × lr × sign(g) whatever the gradient (√(1 − β₂) / (1 − β₁)).
            # On 576 strongly correlated inputs that wrecked the O-mat policy in the first unfrozen iteration
            # (§2.3.9). Seed the second moment with the trained proprioceptive columns' median: steps ∝ gradient.
            for net in (self.actor, self.critic):
                v = self.optimizer.state[net.mlp[0].weight]["exp_avg_sq"]
                v[:, cols] = v[:, : layout.OLD_ACTOR_DIM].median()
            self._frozen = self._frozen[(self._frozen < cols.start) | (self._frozen >= cols.stop)]
            self._snapshot()
        self._live_pending = []
        super().process_env_step(obs, rewards, dones, extras)
        self._restore(self.actor.obs_normalizer)
        self._restore(self._shadow)

    # ------------------------------------------------------------------ schedules
    def _open_gate(self):
        env, a = self.env, layout.ACTOR
        aux = env.aux
        b = aux.buffer.sample(16384, aux.dt)
        with torch.no_grad():
            _, e, _ = aux(b["hist"])
        set_block_stats(self.actor.obs_normalizer, a["e_hat"], e.mean(0), e.var(0))
        # ê now updates like any live block: drop it from the frozen set.
        keep = (self._frozen < a["e_hat"].start) | (self._frozen >= a["e_hat"].stop)
        self._frozen = self._frozen[keep]
        self._snapshot()
        env.e_gate = 1.0
        self.gate_iter = self.iteration

    def update(self) -> dict[str, float]:
        loss = super().update()
        env = self.env
        loss.update({f"aux_{k}": v for k, v in env.aux.update().items()})
        if self.iteration % self.v_test_every == 0 and env.aux.stats_frozen:
            vt = env.aux.v_tests()
            self.last_v_tests = vt
            ok = vt["v_rmse_slip"] < self.slip_ratio_max * vt["v_rmse_rigid"] and (
                vt["v_rmse_imu_zero"] >= self.imu_zero_min * vt["v_rmse"]
            )
            self.passes = self.passes + 1 if ok else 0
            if self.gate_iter < 0 and self.passes >= self.v_test_passes and not self.actor_frozen:
                self._open_gate()
        if self.gate_iter >= 0:
            t = (self.iteration - self.gate_iter - self.v0_retire_delay) / self.v0_retire_iters
            env.v0_drop_p = min(max(t, 0.0), 1.0)
        # Zero-held blocks (everything after ê except the arm's own block) must still have zero first-layer columns.
        held = torch.ones(self._width, dtype=torch.bool)
        held[: layout.ACTOR["m_star"].start] = False
        if layout.ARM_BLOCK[env.cfg.arm]:
            held[layout.ARM_ACTOR[layout.ARM_BLOCK[env.cfg.arm]]] = False
        assert self.actor.mlp[0].weight[:, held].abs().max() == 0.0, f"arm {env.cfg.arm}: a zero-held block got weight"
        if self.ramp_iters > 0:
            loss["reward_ramp"] = self._ramp()
        if self.s2_start >= 0:
            env.clock_drop_p = min((self.iteration - self.s2_start) / self.clock_anneal_iters, 1.0)
            loss["clock_drop_p"] = env.clock_drop_p
        loss.update({f"vtest_{k}": v for k, v in self.last_v_tests.items()})
        loss.update({"e_gate": env.e_gate, "v0_drop_p": env.v0_drop_p, "v_test_passes": float(self.passes)})
        return loss

    # ------------------------------------------------------------------ checkpoint
    def save(self) -> dict:
        saved = super().save()
        saved["aux_state_dict"] = self.env.aux.state_dict()
        saved["aux_optimizer_state_dict"] = self.env.aux.opt.state_dict()
        saved["s1_state"] = {
            "passes": self.passes,
            "gate_iter": self.gate_iter,
            "frozen": self._frozen.cpu(),
            "v0_drop_p": self.env.v0_drop_p,
            "window": self.env.cfg.aux.window,
            "s2_start": self.s2_start,
            "clock_drop_p": self.env.clock_drop_p,
            "arm": self.env.cfg.arm,
            "redesign": self.env.cfg.redesign,
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if loaded_dict["actor_state_dict"]["mlp.0.weight"].shape[1] < self._width:  # S1 trunk → S2 layout
            widen_checkpoint(loaded_dict, self._width)
        ret = super().load(loaded_dict, load_cfg, strict)
        env = self.env
        env.aux.load_state_dict(loaded_dict["aux_state_dict"])
        env.aux.opt.load_state_dict(loaded_dict["aux_optimizer_state_dict"])
        s = loaded_dict["s1_state"]
        self.passes, self.gate_iter = s["passes"], s["gate_iter"]
        self._frozen = s["frozen"].to(self.device)
        self._snapshot()
        env.e_gate = 1.0 if self.gate_iter >= 0 else 0.0
        env.v0_drop_p = s["v0_drop_p"]
        self.s2_start = s.get("s2_start", -1)  # S1 checkpoints predate the S2 fields
        env.clock_drop_p = s.get("clock_drop_p", 0.0)
        if self.ramp_iters > 0:
            self._ramp()
        if self.clock_anneal_iters > 0 and self.s2_start < 0:  # branching from an S1 trunk
            assert s.get("arm", "B0") == "B0", "branch from an S1 trunk, not from another S2 arm"
            self.s2_start = self.iteration
            self.actor_frozen = True
            self.actor.requires_grad_(False)
            self.actor_freeze_iters = self.iteration + self.reburn_iters
            if layout.ARM_BLOCK[env.cfg.arm]:
                self._live_pending = [layout.ARM_ACTOR[layout.ARM_BLOCK[env.cfg.arm]]]
        return ret
