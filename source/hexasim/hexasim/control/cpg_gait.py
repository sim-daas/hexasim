# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop tripod CPG baseline — the vendor's own gait law, not an invented one: `target_toe_body()` is a
direct port of `tripod_gait()` / `move_in_world()` in Interbotix's `interbotix_xs_modules/hexapod.py`
(https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/hexapod.py,
the stock gait generator shipped for this exact robot family, PhantomX/WidowX Mark IV). No terrain sensing, no
contact feedback, no learning — for comparison against the trained policy (`record_gait.py --controller cpg`)
only, never evidence of a deployable gait.

Original source (verbatim, for the derivation below): per discrete gait step `step_cntr` of `num_steps` per
cycle, `inc = 1/gait_factor * 0.5*(1 + sin(2*pi*step/num_steps - pi/2))` (gait_factor = 2.0 for tripod) and
`foot_height = max_foot_height * 0.5*(1 + sin(4*pi*step/num_steps - pi/2))`; `tripod_gait()` then applies, for
leg group A = {right_front, right_back, left_middle}: `p_f = [-x_inc, -y_inc, 0 if step < num_steps/2 else
foot_height]`, and for group B (the other three legs): `p_f = [x_inc, y_inc, 0 if step > num_steps/2 else
foot_height]`, as a translation offset from the leg's `self.foot_points[leg]` anchor (set once before the gait
starts — the same "fixed p_rest, offset from it every step" structure this module already used, now confirmed to
match the vendor's own convention, not just a convenient implementation choice).

Substituting the continuous phase `phi = 2*pi*step/num_steps` (this module's `self.phase`, advanced by `step()`
at a fixed frequency instead of a discrete per-call step counter) collapses `inc`/`foot_height` to the two
trig identities below (`sin(phi - pi/2) = -cos(phi)`, `sin(2*phi - pi/2) = -cos(2*phi)`):

    inc(phi)  = 0.25 * (1 - cos(phi))            # 0 -> stride/2 -> 0 over one full cycle (0.25 = 1/gait_factor * 0.5)
    lift(phi) = step_height * 0.5 * (1 - cos(2*phi))   # 0 -> step_height -> 0, twice per cycle

so group A's foot stays on the ground (dz = 0) and drags from the anchor to -stride/2 while phi runs 0 -> pi
(stance), then lifts (dz = lift(phi)) and swings back from -stride/2 to the anchor while phi runs pi -> 2*pi
(swing); group B runs the mirror image, offset by exactly half a cycle (swinging while A stances and vice
versa) — the classic duty-factor-0.5 tripod alternation, reproduced to the same formula the real robot runs.
`forward_kinematics`/`inverse_kinematics` (this package's `leg_kinematics.py`) are unchanged: they solve for the
joint angles that place the toe at this target, using this project's own simulator-validated geometry rather
than Interbotix's URDF-derived closed-form IK (a different parameterization of the same kinematic chain; mixing
the two risks exactly the sign/offset-angle errors this project already paid down once deriving the FK).
"""

from __future__ import annotations

import math

import torch

# Leg order throughout: right_front, right_middle, right_back, left_front, left_middle, left_back (matches
# hexasim...mdp.rewards.LEG_NAMES). True = group A (right_front, right_back, left_middle): stances first half of
# the cycle, swings second half. False = group B (the other three): swings first half, stances second half.
GROUP_A = (True, False, True, False, True, False)
# Project-wide invariant this port assumes (holds for Interbotix's tripod_gait: exactly two alternating groups).
DUTY_FACTOR = 0.5


class OpenLoopTripodGait:
    """`target_toe_body()` for the current phase, `step()` to advance it. Nothing about the trajectory depends
    on sensed state — the only external inputs are the fixed construction-time parameters."""

    def __init__(self, p_rest: torch.Tensor, cmd_vx: float, freq: float, step_height: float, step_dt: float):
        """p_rest: (1, 6, 3) toe rest position per leg, body frame. Measure ONCE at episode start
        (`coxa_leg_geometry(env)[0]`, right after `env.reset()`, before the first `env.step()`) and never update
        it — that is what makes this controller open-loop (no contact/terrain feedback into the foot target)."""
        device = p_rest.device
        self.p_rest = p_rest.clone()
        self.freq = freq
        self.step_height = step_height
        self.step_dt = step_dt
        self.group_a = torch.tensor(GROUP_A, device=device)  # (6,)
        self.phase = 0.0  # radians, scalar: one shared clock drives both leg groups, exactly as the source does

        # Adjacent same-side legs (right_front/right_middle, right_middle/right_back, left_front/left_middle,
        # left_middle/left_back) are in opposite tripod groups, so at phi = pi (the stance/swing turnaround) one
        # is at its rearmost x offset (-stride/2) and its neighbour at its foremost (+stride/2): the x gap between
        # them is toe_x_spacing - stride at that instant. A stride approaching or exceeding the rest-pose spacing
        # therefore drives their legs through each other — exactly the collision observed in the recorded CPG
        # video. Cap the achievable stride at 60% of the tightest such spacing (measured from p_rest, not a
        # hard-coded geometry constant) so every adjacent pair keeps >= 40% of its rest spacing as clearance at
        # the worst instant; --cmd above this stays open-loop but walks slower than commanded, which is reported
        # below rather than hidden.
        same_side_pairs = ((0, 1), (1, 2), (3, 4), (4, 5))
        min_spacing = min((p_rest[0, i, 0] - p_rest[0, j, 0]).abs().item() for i, j in same_side_pairs)
        max_stride = 0.6 * min_spacing
        self.raw_stride = cmd_vx / freq
        self.stride = min(self.raw_stride, max_stride)
        self.stride_clamped = self.stride < self.raw_stride

    def target_toe_body(self) -> torch.Tensor:
        """(1, 6, 3) target toe position, body frame, for the current phase."""
        phi = self.phase
        inc = 0.25 * (1.0 - math.cos(phi)) * self.stride
        lift = self.step_height * 0.5 * (1.0 - math.cos(2.0 * phi))
        in_first_half = phi < math.pi
        dz_a = 0.0 if in_first_half else lift  # group A: stance first half, swing second half
        dz_b = lift if in_first_half else 0.0  # group B: swing first half, stance second half
        target = self.p_rest.clone()
        dx = torch.where(self.group_a, torch.full_like(target[..., 0], -inc), torch.full_like(target[..., 0], inc))
        dz = torch.where(self.group_a, torch.full_like(target[..., 0], dz_a), torch.full_like(target[..., 0], dz_b))
        target[..., 0] += dx
        target[..., 2] += dz
        return target

    def step(self) -> None:
        self.phase = (self.phase + 2.0 * math.pi * self.freq * self.step_dt) % (2.0 * math.pi)
