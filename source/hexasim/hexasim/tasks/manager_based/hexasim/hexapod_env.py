# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Sequence

import torch
from rsl_rl.modules.normalization import EmpiricalNormalization

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .mdp.proprio import ProprioReader
from .mdp.rewards import COXA_BACKWARD_SIGN, LEG_NAMES, coxa_leg_geometry, coxa_stroke_weight, toe_pos_w
from .mdp.terrain import HeightMap


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
        self._coxa_body_ids = [i for i, n in enumerate(robot.data.body_names) if "coxa" in n]
        self._femur_body_ids = [i for i, n in enumerate(robot.data.body_names) if "femur" in n]
        ids = (self._tibia_body_ids, self._coxa_body_ids, self._femur_body_ids)
        sensor_names = self.scene["contact_forces"].body_names
        for names in [[robot.data.body_names[i] for i in b] for b in ids] + [sensor_names]:
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
        # Envs whose clock runs and whose CPG terms count; S0b clears it for standing (zero-command) envs.
        self.cpg_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        super().load_managers()
        # Per-env, per-leg peak load in tripod stance: total weight / 3, (N, 1). Read after the startup
        # events added mast and payload, which differ per env.
        self.tripod_leg_load = robot.data.body_mass.torch.sum(dim=1, keepdim=True) * 9.81 / 3.0
        # Rest-pose horizontal toe reach from each coxa (env 0); normalises the coxa stroke term and is the footprint
        # target. At this point the articulation is still in the USD zero pose (joints 0 → legs straight out, reach
        # 0.208 m; the configured rest pose is applied at the first reset), so put it in the rest pose and let PhysX
        # refresh the link transforms once before measuring (phase2-progress §2.1.35). Reach depends on joint angles
        # only; the one 5 ms step before the first reset is harmless.
        q0 = robot.data.default_joint_pos.torch
        robot.write_joint_state_to_sim_index(position=q0, velocity=torch.zeros_like(q0))
        self.sim.step(render=False)
        self.scene.update(self.physics_dt)
        self.coxa_rest_reach = coxa_leg_geometry(self)[1][0, :, :2].norm(dim=-1)
        assert (self.coxa_rest_reach < 0.15).all(), f"rest reach looks like the zero pose: {self.coxa_rest_reach}"
        # The command-projected coxa stroke sign must reduce to the fixed backward sign for straight walking.
        forward = torch.tensor([[0.1, 0.0, 0.0]], device=self.device).expand(self.num_envs, 3)
        sign = torch.sign(coxa_stroke_weight(self, forward))
        assert (sign == torch.tensor(COXA_BACKWARD_SIGN, device=self.device)).all(), f"coxa stroke sign: {sign[0]}"

    def _velocity_input(self) -> torch.Tensor:
        return self.scene["robot"].data.root_lin_vel_b.torch

    def coxa_stroke_weight(self) -> torch.Tensor:
        """(N, 6) multiplier turning coxa rate into the stance stroke the gait needs; Flat/S0a: fixed backward sign."""
        return torch.tensor(COXA_BACKWARD_SIGN, device=self.device).expand(self.num_envs, 6)

    def _clock_frequency(self) -> torch.Tensor:
        # f = v_x / 0.20 m stride, clamped [0.8, 2.5] Hz (v7 §2.2 stall floor).
        return torch.clamp(self.v_in[:, 0] / 0.20, min=0.8, max=2.5)

    def terrain_z(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height at world xy (..., 2); flat ground plane at z = 0 here."""
        return torch.zeros(xy.shape[:-1], device=xy.device)

    def terrain_normal(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain unit normal at world xy (..., 2) → (..., 3); world z here."""
        return torch.tensor([0.0, 0.0, 1.0], device=xy.device).expand(*xy.shape[:-1], 3)

    def step(self, action: torch.Tensor):
        with torch.no_grad():
            self.v_in = self._velocity_input()
            f = self._clock_frequency()
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
        root_xy = self.scene["robot"].data.root_pos_w.torch[env_ids, :2]
        self.cpg_foot_baseline[env_ids] = self.terrain_z(root_xy).unsqueeze(1)


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
        self.imu_bias = torch.zeros(self.num_envs, 6, device=self.device)  # accel (3), gyro (3); S0b samples it
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
        load = servo.applied_effort / servo.stall
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
                noise(imu.lin_acc_b.torch + self.imu_bias[:, :3], c.imu_acc_std),
                noise(imu.ang_vel_b.torch + self.imu_bias[:, 3:], c.imu_gyro_std),
                noise(robot.data.projected_gravity_b.torch, c.gravity_std),
                # Clock; zeros (the spec's "absent" flag) while the clock is stopped — a frozen phase kept the
                # "swing" legs of that phase light in standing mode (phase2-progress §2.1.10).
                torch.stack((torch.sin(self.cpg_phase), torch.cos(self.cpg_phase)), dim=-1).view(self.num_envs, 12)
                * self.cpg_active.unsqueeze(1),
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
        self.v0_last = v_hat.detach()
        self.v_in = (1.0 - self.v0_alpha) * v_true + self.v0_alpha * v_hat.detach()
        self.extras.setdefault("log", {}).update(
            {"S0a/v0_huber": self.v0_loss.item(), "S0a/v0_alpha": self.v0_alpha, "S0a/read_valid": p.mask.mean().item()}
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._fresh[env_ids] = True


class HexapodS0bEnv(HexapodS0aEnv):
    """S0b env (paradigm spec §4.2): twist_cmd, command-referenced clock, standing mode, terrain, full DR.

    Clock f_ref = clamp(v_cmd,x / 0.20, 0.8, 2.5) Hz from the command (replaces S0a's V_0-driven clock); standing
    envs (zero command) freeze the clock and drop every CPG term (``cpg_active``); for the first
    ``cfg.stand_settle_s`` of a stand only the rest-pose pull acts, so the robot may step to re-pose, then the
    hold terms (contact, load share, joint speed) start (``stand_hold``). Terrain height and normal come
    from a height map raycast once from the terrain mesh. ``foot_material`` (N, 5) is set by the prestartup
    material event; the IMU bias is resampled per episode.
    """

    def load_managers(self):
        self.heightmap = HeightMap(self.scene.terrain.cfg.prim_path + "/terrain/mesh", self.device)
        self.spawn_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_dist = torch.zeros(self.num_envs, device=self.device)  # commanded path length this episode
        # Standing: seconds since the zero command began; the standing hold terms start after the settle window.
        self.stand_time = torch.zeros(self.num_envs, device=self.device)
        self.stand_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        super().load_managers()
        self.cmd_term = self.command_manager.get_term("base_velocity")

    def terrain_z(self, xy: torch.Tensor) -> torch.Tensor:
        return self.heightmap.height(xy)

    def terrain_normal(self, xy: torch.Tensor) -> torch.Tensor:
        return self.heightmap.normal(xy)

    def coxa_stroke_weight(self) -> torch.Tensor:
        return coxa_stroke_weight(self, self.cmd_term.command)

    def _clock_frequency(self) -> torch.Tensor:
        cmd = self.cmd_term.command
        return torch.clamp(cmd[:, 0] / 0.20, min=0.8, max=2.5) * self.cpg_active

    def step(self, action: torch.Tensor):
        self.cpg_active = ~self.cmd_term.is_standing_env
        self.stand_time = torch.where(self.cpg_active, 0.0, self.stand_time + self.step_dt)
        self.stand_hold = ~self.cpg_active & (self.stand_time >= self.cfg.stand_settle_s)
        self.cmd_dist += torch.norm(self.cmd_term.command[:, :2], dim=1) * self.step_dt
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)  # curriculum reads spawn_xy / cmd_dist before they are reset here
        self.spawn_xy[env_ids] = self.scene["robot"].data.root_pos_w.torch[env_ids, :2]
        self.cmd_dist[env_ids] = 0.0
        self.stand_time[env_ids] = 0.0
        c = self.cfg
        n = len(env_ids)
        self.imu_bias[env_ids] = torch.randn(n, 6, device=self.device) * torch.tensor(
            [c.imu_acc_bias_std] * 3 + [c.imu_gyro_bias_std] * 3, device=self.device
        )


class HexapodS1Env(HexapodS0bEnv):
    """S1 blind trunk env (paradigm spec §6, phase2-progress §2.2).

    On top of S0b:
    * spatial terrain materials (``mdp.materials``): static field + per-entry submeshes (prestartup event); per-foot
      μ_s, μ_d, k written through the PhysX tensor API from each env's per-episode palette; damping per env (D25);
    * per-env, per-episode drop-slope factor (D24);
    * the auxiliary learner ``aux`` (E_p, V, probes; ``hexasim.rl.s1.aux``), fed a W-step o_p history, and the
      sequence-buffer rows it trains from (labels, touchdown events);
    * the 1,125-D actor layout (``mdp.layout``): V_0 frozen (no online training) and retired by per-episode dropout
      (``v0_drop_p``, D23); ê enters through a binary gate (``e_gate``) opened by the algorithm once V passes v7 §4.3;
      all other new slots held at zero;
    * contract P (683-D) for the critic; z_p and ê recomputed on the mirrored history for the mirror augmentation.
    """

    def load_managers(self):
        from hexasim.rl.s1.aux import S1Aux
        from hexasim.rl.s1.kinematics import toe_positions

        from .mdp import layout
        from .mdp.materials import FootWriter, Palette

        # Everything the observation terms read must exist before the managers are built (they probe the terms).
        robot = self.scene["robot"]
        self.palette = Palette(self.num_envs, self.device)
        self.drop_k_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.e_gate = 0.0
        self.v0_drop_p = 0.0
        self.v0_keep = torch.ones(self.num_envs, 1, device=self.device)
        q0 = robot.data.default_joint_pos.torch[0]
        self.aux = S1Aux(self.cfg.aux, self.num_envs, q0, self.step_dt, self.device)
        self._whist = torch.zeros(self.num_envs, self.cfg.aux.window, 111, device=self.device)
        # Nominal footholds: rest-pose toes in the body frame (fixed, clock-independent; spec §2.2/§3.5).
        self.nominal_toe_b = toe_positions(q0).to(self.device)
        f = torch.tensor(layout.STRIP_FWD, device=self.device)
        la = torch.tensor(layout.STRIP_LAT, device=self.device)
        self._strip_off = torch.stack(torch.meshgrid(f, la, indexing="ij"), dim=-1).view(-1, 2)  # (24, 2)
        f = torch.tensor(layout.SCAN_FWD, device=self.device)
        la = torch.tensor(layout.SCAN_LAT, device=self.device)
        self._scan_off = torch.stack(torch.meshgrid(f, la, indexing="ij"), dim=-1).view(-1, 2)  # (77, 2)
        self._grid_off = torch.tensor(layout.LOCAL_GRID, device=self.device)  # (9, 2)
        self._prev_contact = torch.zeros(self.num_envs, 6, dtype=torch.bool, device=self.device)
        self._prev_toe = torch.zeros(self.num_envs, 6, 3, device=self.device)
        self._actor_zero = torch.zeros(
            self.num_envs, layout.ACTOR_DIM - layout.ACTOR["m_star"].start, device=self.device
        )
        self._base_idx = robot.find_bodies(".*base_link.*")[0][0]
        super().load_managers()
        self.v0_train = False  # frozen at π_S0's weights (D23)
        self.proprio.k = self.cfg.drop_k * self.drop_k_scale  # (N, 1), broadcasts over the 18 servos
        self.foot_writer = FootWriter(self)

    # ------------------------------------------------------------------ helpers
    def _yaw_offsets_w(self, off_b: torch.Tensor) -> torch.Tensor:
        """Body-frame horizontal offsets (..., K, 2) rotated by base yaw → world xy offsets (N, ..., K, 2)."""
        q = self.scene["robot"].data.root_quat_w.torch
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        c, s = torch.cos(yaw), torch.sin(yaw)
        shape = (-1,) + (1,) * (off_b.dim() - 1)
        c, s = c.view(shape), s.view(shape)
        return torch.stack((c * off_b[..., 0] - s * off_b[..., 1], s * off_b[..., 0] + c * off_b[..., 1]), dim=-1)

    def material4(self, xy: torch.Tensor) -> torch.Tensor:
        """(μ_s, μ_d, log10 k, log10(1 + c)) at world xy (N, ..., 2); damping c is per env (D25)."""
        m = self.palette.at(self.material_field, xy)
        c = self.foot_material[:, 4].view(-1, *([1] * (xy.dim() - 2)), 1).expand(*m.shape[:-1], 1)
        return torch.cat((m, c), dim=-1)

    def _local_heights(self, toe: torch.Tensor) -> torch.Tensor:
        """3 × 3 terrain heights at 4 cm around each toe (body-yaw axes), relative to the toe: (N, 6, 9)."""
        xy = toe[..., None, :2] + self._yaw_offsets_w(self._grid_off)[:, None]
        return self.terrain_z(xy) - toe[..., None, 2]

    # ------------------------------------------------------------------ per-step
    def refresh_proprio(self):
        if self._stamp == self._sim_step_counter:
            return
        fresh = self._fresh.clone()
        super().refresh_proprio()
        from .mdp.layout import CONTRACT_P_DIM
        from .mdp.observations import s0a_privileged_obs
        from .mdp.rewards import foot_forces_nt
        from .mdp.symmetry import mirror_frame

        robot = self.scene["robot"]
        # W-step history; a new episode starts with its first frame repeated.
        self._whist = torch.roll(self._whist, -1, dims=1)
        self._whist[:, -1] = self.o_p
        self._whist[fresh] = self.o_p[fresh].unsqueeze(1)
        lat = self.aux.latents(self._whist)
        lat_m = self.aux.latents(mirror_frame(self._whist))
        lat[:, 64:] *= self.e_gate
        lat_m[:, 64:] *= self.e_gate
        cmd = self.command_manager.get_command("base_velocity")
        self.s1_actor_obs = torch.cat([self.o_p, self.v_in * self.v0_keep, cmd, lat, self._actor_zero], dim=-1)
        self.s1_mirror_latents = lat_m

        # Labels (post-physics state of this step).
        root, quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        toe = toe_pos_w(robot, self._tibia_body_ids)
        forces = torch.norm(self.scene["contact_forces"].data.net_forces_w.torch, dim=-1)
        contact = forces > 1.0
        mat_toe = self.material4(toe[..., :2])  # (N, 6, 4)
        hgt = self._local_heights(toe)  # (N, 6, 9)
        fn, ft = foot_forces_nt(self, SceneEntityCfg("contact_forces"))
        toe_v = (toe - self._prev_toe) / self.step_dt
        slip = (toe_v[..., :2].norm(dim=-1) * contact).sum(1) / contact.sum(1).clamp(min=1)
        slip = torch.where(fresh, 0.0, slip)
        cur = torch.cat(
            [
                contact.float(),
                mat_toe.flatten(1),
                hgt.flatten(1),
                robot.data.root_lin_vel_b.torch,
                torch.stack((fn, ft), dim=-1).flatten(1),
                slip[:, None],
            ],
            dim=-1,
        )
        self.aux.buffer.add(self.o_p, cur, torch.cat((root, quat), dim=-1))
        td = contact & ~self._prev_contact & ~fresh[:, None]
        self.aux.buffer.touchdown(td, toe, torch.cat((mat_toe, hgt), dim=-1))
        self._prev_contact, self._prev_toe = contact, toe

        # Contract P.
        anchor_w = root[:, None, :2] + self._yaw_offsets_w(self.nominal_toe_b[:, :2])  # (N, 6, 2)
        strip_xy = anchor_w[:, :, None] + self._yaw_offsets_w(self._strip_off)[:, None]  # (N, 6, 24, 2)
        scan_xy = root[:, None, :2] + self._yaw_offsets_w(self._scan_off)  # (N, 77, 2)
        servo = robot.actuators["ax12a"]
        dr = torch.cat(
            [
                servo.stall / servo.cfg.stall_torque,
                servo.slope / 32.0,
                robot.data.body_mass.torch.sum(dim=1, keepdim=True).to(self.device),
                robot.data.body_com_pos_b.torch[:, self._base_idx, :].to(self.device),
                servo._delay.time_lags.float().view(-1, 1).to(self.device),
                self.drop_k_scale,
            ],
            dim=-1,
        )
        self.s1_contract_p = torch.cat(
            [
                s0a_privileged_obs(self),
                self.material4(strip_xy).flatten(1),
                self.terrain_z(scan_xy) - root[:, 2:3],
                dr,
            ],
            dim=-1,
        )
        assert self.s1_contract_p.shape[1] == CONTRACT_P_DIM

    def step(self, action: torch.Tensor):
        ret = super().step(action)
        robot = self.scene["robot"]
        toe = toe_pos_w(robot, self._tibia_body_ids)
        low = toe[..., 2] - self.terrain_z(toe[..., :2]) < self.foot_writer.low_z
        contact = torch.norm(self.scene["contact_forces"].data.net_forces_w.torch, dim=-1) > 1.0
        self.foot_writer.update(self.palette.at(self.material_field, toe[..., :2]), low | contact)
        log = self.extras.setdefault("log", {})
        log["S1/material_writes_per_step"] = self.foot_writer.writes
        log["S1/e_gate"] = self.e_gate
        log["S1/v0_keep"] = self.v0_keep.mean().item()
        self.foot_writer.writes = 0
        return ret

    def _reset_idx(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device)
        self.palette.redraw(ids)
        lo, hi = self.cfg.drop_k_scale_range
        self.drop_k_scale[ids] = lo + torch.rand(len(ids), 1, device=self.device) * (hi - lo)
        self.proprio.k[ids] = self.cfg.drop_k * self.drop_k_scale[ids]
        self.v0_keep[ids] = (torch.rand(len(ids), 1, device=self.device) >= self.v0_drop_p).float()
        self.aux.buffer.episode_end(ids)
        super()._reset_idx(env_ids)
