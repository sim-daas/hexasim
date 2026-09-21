# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamixel AX-12A servo model (v7 §5.1): P-only compliance curve on a DC motor, explicit torque.

Per physics step, per joint:

    q_goal  = delay(quantise(q_cmd))                       bus latency, whole physics steps, per env
    e       = q_goal - quantise(q)                         10-bit output-shaft potentiometer
    dz      = margin·tick + backlash/2                     compliance margin + gear-play dead zone
    u       = sign(e)·(punch + (1-punch)·clip((|e|-dz)/(slope·tick), 0, 1))   if |e| > dz, else 0
    τ       = clip(τ_stall·(u - q̇/ω_nl), ±τ_stall) − τ_c·tanh(q̇/0.05)      DC motor with back-EMF, gear friction

Back-EMF is the only damping: the servo firmware is P-only, there is no Kd to tune.
Stall torque, compliance slope, backlash, Coulomb friction τ_c and latency are per env, resampled at every
reset from the ``*_range`` / ``*_choices`` fields (dynamics randomisation, v7 §5.6); the defaults are single
datasheet values (S0a). ponytail: provisional until the bench characterisation (v7 action item 14); every
constant here is a calibration knob. Not modelled yet: thermal derating (lower ``stall_torque`` instead),
hysteretic backlash (approximated by the dead zone).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.configclass import configclass

TICK = math.radians(300.0) / 1023.0  # 10-bit over 300°: 0.00512 rad


def quantise(q: torch.Tensor) -> torch.Tensor:
    return torch.round(q / TICK) * TICK


class AX12AActuator(ActuatorBase):
    cfg: AX12AActuatorCfg
    is_implicit_model = False

    def __init__(self, cfg: AX12AActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        shape = (self._num_envs, 1)
        self.stall = torch.full(shape, cfg.stall_torque, device=self._device)
        self.slope = torch.full(shape, float(cfg.compliance_slope), device=self._device)
        self.backlash = torch.full(shape, cfg.backlash, device=self._device)
        self.coulomb = torch.zeros(shape, device=self._device)
        self.reset(slice(None))

    def reset(self, env_ids: Sequence[int] | slice):
        ids = None if isinstance(env_ids, slice) else env_ids
        n = self._num_envs if ids is None else len(ids)
        lag = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), dtype=torch.int, device=self._device)
        self._delay.set_time_lag(lag, ids)
        self._delay.reset(ids)
        c = self.cfg
        idx = slice(None) if ids is None else ids
        u = lambda r: torch.empty(n, 1, device=self._device).uniform_(*r)  # noqa: E731
        self.stall[idx] = c.stall_torque * u(c.stall_scale_range)
        slopes = torch.tensor(
            c.compliance_slope_choices or (c.compliance_slope,), dtype=torch.float, device=self._device
        )
        self.slope[idx] = slopes[torch.randint(len(slopes), (n, 1), device=self._device)]
        self.backlash[idx] = u(c.backlash_range or (c.backlash, c.backlash))
        self.coulomb[idx] = u(c.coulomb_range)

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        c = self.cfg
        q_goal = self._delay.compute(quantise(control_action.joint_positions))
        e = q_goal - quantise(joint_pos)
        dz = c.compliance_margin * TICK + 0.5 * self.backlash
        ramp = torch.clamp((e.abs() - dz) / (self.slope * TICK), 0.0, 1.0)
        u = torch.sign(e) * (c.punch + (1.0 - c.punch) * ramp) * (e.abs() > dz)
        tau = torch.clamp(self.stall * (u - joint_vel / c.no_load_speed), -self.stall, self.stall)
        tau = tau - self.coulomb * torch.tanh(joint_vel / 0.05)
        self.computed_effort = tau
        self.applied_effort = tau
        control_action.joint_efforts = tau
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class AX12AActuatorCfg(ActuatorBaseCfg):
    class_type: type = AX12AActuator
    stiffness: float = 0.0  # explicit model: PhysX drive off
    damping: float = 0.0
    armature: float = 0.003  # reflected rotor inertia ≈ 5e-8 kg·m² × 254², kg·m²
    stall_torque: float = 1.5  # N·m at 12 V
    no_load_speed: float = 59.0 * 2.0 * math.pi / 60.0  # rad/s at 12 V
    compliance_margin: int = 1  # ticks
    compliance_slope: int = 32  # ticks over which the duty ramps to full
    punch: float = 32.0 / 1023.0  # minimum duty once outside the dead zone
    backlash: float = math.radians(0.5)  # rad, total play
    min_delay: int = 1  # physics steps (5 ms): bus latency 4–12 ms
    max_delay: int = 2
    # Per-env randomisation, resampled at reset. Defaults (None / degenerate) = the single values above (S0a).
    stall_scale_range: tuple[float, float] = (1.0, 1.0)
    compliance_slope_choices: tuple[int, ...] | None = None
    backlash_range: tuple[float, float] | None = None
    coulomb_range: tuple[float, float] = (0.0, 0.0)  # N·m
