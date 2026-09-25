# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env ring buffer for the auxiliary losses (paradigm spec §8.3, v7 §4.12).

Decoupled from the PPO rollout: every env writes one row per control step into a length-L ring. A sample at row t
uses the W-step input history [t − W + 1, t] (which must lie in one episode) and the row's labels.

Touchdown targets (§3.2) are written retroactively. Row t's target for leg ℓ is ℓ's first touchdown strictly after
t. When ℓ touches down at step s, every row of the same episode from ℓ's previous touchdown up to s − 1 receives
the touchdown's per-event label (world toe position, material and local heights at the ground-truth point) and s
itself. Row-dependent quantities are formed at sample time: Δt^TD = (s − t)·dt, and p^TD in row t's body frame from
the base pose stored with the row. A row whose leg has not touched down again before the episode ends, or before
the ring overwrote it, keeps ``td_valid = False`` and is masked from that leg's touchdown loss.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse


class SequenceBuffer:
    def __init__(
        self,
        num_envs: int,
        length: int,
        frame_dim: int,
        cur_dim: int,
        td_event_dim: int,
        window: int,
        device: torch.device | str,
        num_legs: int = 6,
    ):
        assert length > window
        self.N, self.L, self.W, self.legs = num_envs, length, window, num_legs
        self.device = device
        f = dict(device=device)
        self.frame = torch.zeros(num_envs, length, frame_dim, **f)
        self.cur = torch.zeros(num_envs, length, cur_dim, **f)
        self.pose = torch.zeros(num_envs, length, 7, **f)  # base pos_w (3), quat_w (4, w first)
        self.td_pos_w = torch.zeros(num_envs, length, num_legs, 3, **f)
        self.td_event = torch.zeros(num_envs, length, num_legs, td_event_dim, **f)
        self.td_step = torch.zeros(num_envs, length, num_legs, dtype=torch.long, **f)
        self.td_valid = torch.zeros(num_envs, length, num_legs, dtype=torch.bool, **f)
        self.row_step = torch.full((num_envs, length), -1, dtype=torch.long, **f)
        self.episode = torch.full((num_envs, length), -1, dtype=torch.long, **f)
        self.step = 0  # global write counter; slot = step % L
        self.ep_counter = torch.zeros(num_envs, dtype=torch.long, **f)
        # Per env per leg: first global step whose touchdown target is still open.
        self.open_from = torch.zeros(num_envs, num_legs, dtype=torch.long, **f)

    # ------------------------------------------------------------------ write
    def add(self, frame: torch.Tensor, cur: torch.Tensor, pose: torch.Tensor) -> None:
        """Row for the current control step."""
        s = self.step % self.L
        self.frame[:, s] = frame
        self.cur[:, s] = cur
        self.pose[:, s] = pose
        self.episode[:, s] = self.ep_counter
        self.row_step[:, s] = self.step
        self.td_valid[:, s] = False
        self.step += 1

    def touchdown(self, legs_mask: torch.Tensor, pos_w: torch.Tensor, event: torch.Tensor) -> None:
        """Legs (N, legs) that touched down at the step just added; ``pos_w`` (N, legs, 3), ``event`` (N, legs, D)."""
        now = self.step - 1
        env_idx, leg_idx = legs_mask.nonzero(as_tuple=True)
        if env_idx.numel() > 0:
            start = self.open_from[env_idx, leg_idx].clamp(min=max(0, now - self.L + 1))
            span = (now - start).clamp(min=0)  # rows start … now − 1
            max_span = int(span.max().item())
            if max_span > 0:
                offs = torch.arange(max_span, device=self.device)
                ok = offs[None, :] < span[:, None]
                k = torch.arange(env_idx.numel(), device=self.device)[:, None].expand(-1, max_span)[ok]
                g = (start[:, None] + offs[None, :])[ok]
                e, leg, slot = env_idx[k], leg_idx[k], g % self.L
                same = (self.episode[e, slot] == self.ep_counter[e]) & (self.row_step[e, slot] == g)
                e, leg, slot, k = e[same], leg[same], slot[same], k[same]
                self.td_pos_w[e, slot, leg] = pos_w[env_idx[k], leg_idx[k]]
                self.td_event[e, slot, leg] = event[env_idx[k], leg_idx[k]]
                self.td_step[e, slot, leg] = now
                self.td_valid[e, slot, leg] = True
        self.open_from[legs_mask] = now

    def episode_end(self, env_ids: torch.Tensor) -> None:
        """Envs whose episode ended at the step just added; the next row starts a new episode."""
        self.ep_counter[env_ids] += 1
        self.open_from[env_ids] = self.step

    # ------------------------------------------------------------------ sample
    def valid_mask(self) -> torch.Tensor:
        """(N, L) rows usable as sample position t: written, with a full W-step history in the same episode."""
        slots = torch.arange(self.L, device=self.device)
        back = (slots - (self.W - 1)) % self.L
        g_back = self.row_step[:, back]
        return (self.row_step >= 0) & (g_back == self.row_step - (self.W - 1)) & (self.episode[:, back] == self.episode)

    def sample(self, batch: int, dt: float) -> dict[str, torch.Tensor]:
        """Random valid rows. Returns hist (B, W, F), cur (B, C), td_pos_b (B, legs, 3), td_dt (B, legs),
        td_event (B, legs, D), td_valid (B, legs)."""
        flat = self.valid_mask().flatten().nonzero().squeeze(1)
        assert flat.numel() > 0, "sequence buffer has no valid samples yet"
        pick = flat[torch.randint(flat.numel(), (batch,), device=self.device)]
        e, s = pick // self.L, pick % self.L
        offs = torch.arange(-(self.W - 1), 1, device=self.device)
        hist = self.frame[e[:, None], (s[:, None] + offs[None, :]) % self.L]
        pose = self.pose[e, s]
        rel = self.td_pos_w[e, s] - pose[:, None, :3]
        td_pos_b = quat_apply_inverse(pose[:, None, 3:].expand(-1, self.legs, -1), rel)
        td_dt = (self.td_step[e, s] - self.row_step[e, s][:, None]).float() * dt
        return {
            "hist": hist,
            "cur": self.cur[e, s],
            "td_pos_b": td_pos_b,
            "td_dt": td_dt,
            "td_event": self.td_event[e, s],
            "td_valid": self.td_valid[e, s],
        }
