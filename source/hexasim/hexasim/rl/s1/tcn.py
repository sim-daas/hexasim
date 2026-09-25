# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""E_p — proprioceptive encoder: causal dilated TCN over the last W per-step frames (paradigm spec §2.1, §9.1).

Residual block pinned to the standard form (Bai et al. 2018): two causal convs (kernel k, same dilation) with ELU,
plus a residual path (1×1 conv when the width changes). Receptive field RF = 1 + n_c (k − 1) Σ d_i with n_c = 2;
asserted ≥ W at construction. k = 3, d ∈ {1, 2, 4, 8} gives RF 61 (W ≤ 50); W = 75 needs d ∈ {1, …, 16} (RF 125).
Output: the last time step's features → Linear → LayerNorm, z_p ∈ R^64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CONVS_PER_BLOCK = 2


def receptive_field(kernel: int, dilations: tuple[int, ...]) -> int:
    return 1 + CONVS_PER_BLOCK * (kernel - 1) * sum(dilations)


def dilations_for(window: int, kernel: int = 3) -> tuple[int, ...]:
    """Power-of-two dilation stack, at least {1, 2, 4, 8} (v7 §4.2), extended until the RF covers the window."""
    d = [1, 2, 4, 8]
    while receptive_field(kernel, tuple(d)) < window:
        d.append(d[-1] * 2)
    return tuple(d)


class CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv(c_in, c_out, kernel, dilation)
        self.conv2 = CausalConv(c_out, c_out, kernel, dilation)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.conv1(x))
        h = self.conv2(h)
        return F.elu(h + self.skip(x))


class ProprioTCN(nn.Module):
    """(B, W, in_dim) → z_p (B, out_dim). Frames ordered oldest → newest; the newest is at index −1."""

    def __init__(
        self,
        in_dim: int = 111,
        window: int = 50,
        hidden: int = 128,
        out_dim: int = 64,
        kernel: int = 3,
        dilations: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.window = window
        self.dilations = dilations if dilations is not None else dilations_for(window, kernel)
        self.rf = receptive_field(kernel, self.dilations)
        assert self.rf >= window, f"TCN receptive field {self.rf} < window {window}"
        blocks, c = [], in_dim
        for d in self.dilations:
            blocks.append(ResidualBlock(c, hidden, kernel, d))
            c = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # ponytail: evaluates all W positions although only the last is used; a receptive-cone evaluation was tried
        # (phase2-progress §2.2.5) and gained nothing over fp16 inference, and trained slower.
        assert hist.shape[-2] == self.window, (hist.shape, self.window)
        h = self.blocks(hist.transpose(1, 2))  # (B, hidden, W)
        return self.norm(self.head(h[..., -1]))
