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
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        self.v_test_every, self.v_test_passes = v_test_every, v_test_passes
        self.slip_ratio_max, self.imu_zero_min = slip_ratio_max, imu_zero_min
        self.v0_retire_delay, self.v0_retire_iters = v0_retire_delay, v0_retire_iters
        self.passes = 0
        self.gate_iter = -1  # iteration the ê gate opened (−1 = shut)
        self.last_v_tests: dict[str, float] = {}
        a = layout.ACTOR
        dims = list(range(a["v0"].start, a["v0"].stop)) + list(range(a["e_hat"].start, a["e_hat"].stop))
        dims += list(range(a["m_star"].start, layout.ACTOR_DIM))
        self._frozen = torch.tensor(dims, device=self.device)
        self._frozen_stats = None  # taken at the first env step, i.e. after the warm start loaded π_S0's statistics

    def _snapshot(self):
        n = self.actor.obs_normalizer
        self._frozen_stats = {k: getattr(n, k)[:, self._frozen].clone() for k in ("_mean", "_var", "_std")}

    def _restore(self, norm):
        with torch.inference_mode():
            for k, v in self._frozen_stats.items():
                getattr(norm, k)[:, self._frozen] = v

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self._frozen_stats is None:
            self._snapshot()
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
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
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
        return ret
