# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auxiliary losses and heads for S1 (paradigm spec §3.2–3.4, v7 §4.11).

* β-NLL (Seitzer et al., ICLR 2022), β = 0.5: sg[σ^{2β}] · (½ log σ² + (y − μ)² / 2σ²); log σ² clamped to [−6, 4].
  Its mean gradient is −(y − μ)/σ (plain NLL: /σ²).
* VICReg variance + covariance on a batch of latents (no invariance term: there is no second view in S1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0


def beta_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Element-wise β-NLL; same shape as ``y``."""
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    nll = 0.5 * logvar + 0.5 * (y - mu) ** 2 * torch.exp(-logvar)
    return torch.exp(beta * logvar).detach() * nll


class GaussianHead(nn.Module):
    """2-layer MLP (256 hidden) → per-target mean and log-variance (§3.2: same head everywhere)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(), nn.Linear(hidden, 2 * out_dim))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(x).split(self.out_dim, dim=-1)
        return mu, logvar


def vicreg_var_cov(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge on per-dim std (target γ) and off-diagonal covariance, both averaged per dim."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / d
    return var_loss, cov_loss
