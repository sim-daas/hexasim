# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-read pipeline shared by sim and robot (paradigm spec §1, v7 §4.2a).

Per control step: 10-bit reads, per-servo dropped reads (probability rising with servo load),
zero-order hold of q, two-sample backward-difference velocity

    q̇_t = (q_t − q_{t−2}) / 2Δt,  held at the last valid q̇ if either endpoint is a held value,

and an 18-bit validity mask (1 = fresh read). No recursive filter. Pure torch logic (only the servo
constants come from ``hexasim.actuators``), meant to run unchanged on the robot; the drop model is sim-only.
"""

from __future__ import annotations

import torch

from hexasim.actuators.ax12a import TICK, quantise


class ProprioReader:
    def __init__(self, num_envs: int, num_joints: int, dt: float, device, p0: float = 0.005, k: float = 0.068):
        # ponytail: provisional drop model p = p0 + k·|τ|/τ_stall (≈5 % at 1 N·m, 7.3 % at stall). Load, not
        # tracking error: supply sag follows current, which saturates at stall while error grows unbounded.
        # Calibrate on the bench (v7 action item 6) — both are knobs.
        self.dt, self.p0, self.k = dt, p0, k
        shape = (num_envs, num_joints)
        self.q = torch.zeros(shape, device=device)  # ZOH-filled position at t
        self.q_prev = torch.zeros(shape, device=device)  # t−1
        self.q_prev2 = torch.zeros(shape, device=device)  # t−2
        self.qd = torch.zeros(shape, device=device)
        self.mask = torch.ones(shape, device=device)
        self.mask_prev = torch.ones(shape, device=device)
        self.mask_prev2 = torch.ones(shape, device=device)

    def reset(self, env_ids, q_true: torch.Tensor):
        q = quantise(q_true)
        for buf in (self.q, self.q_prev, self.q_prev2):
            buf[env_ids] = q
        self.qd[env_ids] = 0.0
        for buf in (self.mask, self.mask_prev, self.mask_prev2):
            buf[env_ids] = 1.0

    def drop_probability(self, load: torch.Tensor) -> torch.Tensor:
        """Per-servo drop probability from normalised load |τ|/τ_stall ∈ [0, 1]."""
        return self.p0 + self.k * load.abs().clamp(max=1.0)

    def update(self, q_true: torch.Tensor, valid: torch.Tensor):
        """Advance one control step. ``valid`` is the per-servo read success (bool), from the bus or the drop model."""
        self.q_prev2, self.q_prev = self.q_prev, self.q
        self.mask_prev2, self.mask_prev = self.mask_prev, self.mask
        self.q = torch.where(valid, quantise(q_true), self.q_prev)
        self.mask = valid.float()
        fresh = (self.mask * self.mask_prev2).bool()
        self.qd = torch.where(fresh, (self.q - self.q_prev2) / (2.0 * self.dt), self.qd)


if __name__ == "__main__":
    # Self-check: ZOH, velocity hold, mask bits, velocity quantum (spec §1: 0.128 rad/s at Δt = 20 ms).
    r = ProprioReader(1, 1, 0.02, "cpu")
    r.reset(slice(None), torch.zeros(1, 1))
    assert abs(TICK / (2 * 0.02) - 0.128) < 1e-3
    yes, no = torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)
    ramp = [torch.full((1, 1), 0.1 * t) for t in range(1, 8)]  # true q̇ = 5 rad/s
    r.update(ramp[0], yes)
    r.update(ramp[1], yes)
    assert abs(r.qd.item() - 5.0) < 0.2 and r.mask.item() == 1.0
    r.update(ramp[2], no)  # dropped: q held, q̇ held, mask 0
    assert r.q.item() == r.q_prev.item() and abs(r.qd.item() - 5.0) < 0.2 and r.mask.item() == 0.0
    r.update(ramp[3], yes)  # fresh, but t−2 endpoint was held → q̇ still held
    assert abs(r.qd.item() - 5.0) < 0.2 and r.mask.item() == 1.0
    r.update(ramp[4], yes)  # both endpoints fresh again
    assert abs(r.qd.item() - 5.0) < 0.2
    assert abs(r.q.item() - quantise(ramp[4]).item()) < 1e-9
    assert abs(r.drop_probability(torch.tensor(1.0 / 1.5)).item() - 0.05) < 0.001
    print("proprio self-check OK")
