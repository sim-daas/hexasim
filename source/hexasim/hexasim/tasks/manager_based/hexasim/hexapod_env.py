# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence

import torch
from rsl_rl.modules.normalization import EmpiricalNormalization

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.proprio import ProprioReader
from .mdp.rewards import LEG_NAMES, toe_pos_w


class HexapodEnv(ManagerBasedRLEnv):
    """Hexapod env with a CPG clock, an adaptive toe-contact baseline and the velocity fed to the clock.

    Alternating tripod, duty factor ``cfg.duty_factor`` (0.5): swing occupies phase [0, π),
    stance [π, 2π). Tripod A = {RF, RB, LM} at offset 0, tripod B = {RM, LF, LB} at offset π.
    Leg order everywhere is ``LEG_NAMES`` (asserted below).
    """

    def load_managers(self):
        # CPG state must exist before the observation/reward managers probe their terms.
        robot = self.scene["robot"]
        self._tibia_body_ids = [i for i, n in enumerate(robot.data.body_names) if "tibia" in n]
        tibia_names = [robot.data.body_names[i] for i in self._tibia_body_ids]
        sensor_names = self.scene["contact_forces"].body_names
        for names in (tibia_names, sensor_names):
            assert len(names) == 6 and all(leg in n for leg, n in zip(LEG_NAMES, names)), f"leg order: {names}"
        joints_per_type = [[n for n in robot.joint_names if n.endswith(t)] for t in ("_coxa", "_femur", "_tibia")]
        for names in joints_per_type:
            assert all(n.startswith(leg) for leg, n in zip(LEG_NAMES, names)), f"joint order: {names}"
        assert self.cfg.duty_factor == 0.5, "tripod offsets {0, π} assume duty factor 0.5"

        self._cpg_base_offsets = torch.tensor([0.0, math.pi, 0.0, math.pi, 0.0, math.pi], device=self.device)
        self.cpg_phase = self._cpg_base_offsets.repeat(self.num_envs, 1)
        # Toe height at contact, EMA-tracked per leg; 0 = flat ground plane height.
        self.cpg_foot_baseline = torch.zeros(self.num_envs, 6, device=self.device)
        # Velocity the clock (and, in S0a, the actor) sees; ground truth here.
        self.v_in = torch.zeros(self.num_envs, 3, device=self.device)
        super().load_managers()
        # Per-env, per-leg peak load in tripod stance: total weight / 3, (N, 1). Read after the startup
        # events added mast and payload, which differ per env.
        self.tripod_leg_load = robot.data.body_mass.torch.sum(dim=1, keepdim=True) * 9.81 / 3.0

    def _velocity_input(self) -> torch.Tensor:
        return self.scene["robot"].data.root_lin_vel_b.torch

    def step(self, action: torch.Tensor):
        # f = v_x / 0.20 m stride, clamped [0.8, 2.5] Hz (v7 §2.2 stall floor).
        with torch.no_grad():
            self.v_in = self._velocity_input()
            f = torch.clamp(self.v_in[:, 0] / 0.20, min=0.8, max=2.5)
        self.cpg_phase = (self.cpg_phase + 2.0 * math.pi * f.unsqueeze(1) * self.step_dt) % (2.0 * math.pi)

        ret = super().step(action)

        # EMA of toe height on legs in solid contact (> 1 N); sensors are fresh after physics.
        forces = torch.norm(self.scene["contact_forces"].data.net_forces_w.torch, dim=-1)
        toe_z = toe_pos_w(self.scene["robot"], self._tibia_body_ids)[..., 2]
        self.cpg_foot_baseline = torch.where(
            forces > 1.0, 0.8 * self.cpg_foot_baseline + 0.2 * toe_z, self.cpg_foot_baseline
        )
        return ret

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # Per-env global phase jitter keeps tripod structure but breaks batch-wide phase lock (v7 §2.3).
        jitter = torch.rand(len(env_ids), 1, device=self.device) * 2.0 * math.pi
        self.cpg_phase[env_ids] = (self._cpg_base_offsets + jitter) % (2.0 * math.pi)
        self.cpg_foot_baseline[env_ids] = 0.0


class HexapodS0aEnv(HexapodEnv):
    """S0a reconditioning env (paradigm spec §4.2): deployable 111-D o_p frame + the V_0 velocity ramp.

    Once per control step (stamp-guarded, since obs terms are evaluated more than once around resets):
    joint reads through :class:`ProprioReader` (drops, ZOH, 2-sample q̇, mask), noisy IMU, the o_p frame,
    a 5-frame history for V_0, and ``v_in = (1-α)·v_true + α·V_0``, which feeds the actor and the clock.
    V_0 is a small MLP trained online here on each step's batch (Huber vs ground truth). Its output
    enters observations detached, so it cannot receive or send policy gradients.
    """

    HIST = 5

    def load_managers(self):
        c = self.cfg
        self.proprio = ProprioReader(self.num_envs, 18, self.step_dt, self.device, c.drop_p0, c.drop_k)
        self.o_p = torch.zeros(self.num_envs, 111, device=self.device)
        self._hist = torch.zeros(self.num_envs, self.HIST, 111, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._stamp = -1
        self.v0 = torch.nn.Sequential(
            EmpiricalNormalization(self.HIST * 111),
            torch.nn.Linear(self.HIST * 111, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 3),
        ).to(self.device)
        self.v0_opt = torch.optim.Adam(self.v0.parameters(), lr=c.v0_lr)
        self.v0_train = True
        self.v0_loss = torch.zeros((), device=self.device)
        self.v0_alpha = 0.0
        super().load_managers()

    def _velocity_input(self) -> torch.Tensor:
        self.refresh_proprio()
        return self.v_in

    def refresh_proprio(self):
        if self._stamp == self._sim_step_counter:
            return
        self._stamp = self._sim_step_counter
        c, robot = self.cfg, self.scene["robot"]
        q_true = robot.data.joint_pos.torch
        q_cmd = self.action_manager.get_term("joint_pos").processed_actions
        servo = robot.actuators["ax12a"]
        load = servo.applied_effort / servo.cfg.stall_torque
        valid = torch.rand_like(q_true) >= self.proprio.drop_probability(load)
        if self._fresh.any():
            self.proprio.reset(self._fresh, q_true[self._fresh])
        self.proprio.update(q_true, valid)
        p = self.proprio
        imu = self.scene["imu"].data
        noise = lambda x, s: x + s * torch.randn_like(x)  # noqa: E731
        self.o_p = torch.cat(
            [
                p.q - robot.data.default_joint_pos.torch,
                p.qd,
                q_cmd - p.q,
                self.action_manager.action,
                noise(imu.lin_acc_b.torch, c.imu_acc_std),
                noise(imu.ang_vel_b.torch, c.imu_gyro_std),
                noise(robot.data.projected_gravity_b.torch, c.gravity_std),
                torch.stack((torch.sin(self.cpg_phase), torch.cos(self.cpg_phase)), dim=-1).view(self.num_envs, 12),
                p.mask,
            ],
            dim=-1,
        )
        self._hist = torch.roll(self._hist, -1, dims=1)
        self._hist[:, -1] = self.o_p
        self._hist[self._fresh] = self.o_p[self._fresh].unsqueeze(1)
        self._fresh[:] = False

        v_true = robot.data.root_lin_vel_b.torch
        with torch.inference_mode(False), torch.enable_grad():
            x, y = self._hist.flatten(1).clone(), v_true.clone()
            if self.v0_train:
                self.v0[0].update(x)
            v_hat = self.v0(x)
            if self.v0_train:
                loss = torch.nn.functional.huber_loss(v_hat, y, delta=0.1)
                self.v0_opt.zero_grad()
                loss.backward()
                self.v0_opt.step()
                self.v0_loss = loss.detach()
        start, end = c.v0_ramp_steps
        self.v0_alpha = min(max((self.common_step_counter - start) / (end - start), 0.0), 1.0)
        self.v_in = (1.0 - self.v0_alpha) * v_true + self.v0_alpha * v_hat.detach()
        self.extras.setdefault("log", {}).update(
            {"S0a/v0_huber": self.v0_loss.item(), "S0a/v0_alpha": self.v0_alpha, "S0a/read_valid": p.mask.mean().item()}
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._fresh[env_ids] = True
