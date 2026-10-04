# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left–right mirror symmetry for S0b PPO data augmentation (phase2-progress §2.1.27, D21).

Reflection through the sagittal plane (body y → −y). Legs swap RF↔LF, RM↔LM, RB↔LB. Joint angles: coxa flips sign
(+coxa moves right toes forward and left toes backward), femur and tibia keep it (the rest pose is symmetric with
equal angles on both sides) — verified in sim by ``scripts/tests/mirror_test.py``. Polar vectors (velocity, accel,
gravity, CoM offset) negate y; the angular velocity is axial: (ω_x, ω_y, ω_z) → (−ω_x, ω_y, −ω_z); the command
(v_x, v_y, ω_z) → (v_x, −v_y, −ω_z). The per-leg clock is permuted with the legs: that swaps the two tripods, which is
the same clock shifted by π — a state the per-env phase jitter already produces.

Layouts (see ``mdp.observations``): actor 117 = q−q₀ | q̇ | q_cmd−q | a_{t−1} (4 × 18 joints) | IMU accel (3) |
gyro (3) | gravity (3) | clock (6 × (sin, cos)) | read mask (18) | V_0 (3) | twist_cmd (3); critic 151 = actor ⊕
[v_true (3), height (1), per-leg (F_n, F_t) (6 × 2), contact (6)] ⊕ DR [foot material (5), mass (1), CoM x/y (2),
servo scalars (4)]. Joint order [coxa × 6, femur × 6, tibia × 6], legs RF RM RB LF LM LB.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

LEG_SWAP = [3, 4, 5, 0, 1, 2]
_JOINT_SRC = [block * 6 + LEG_SWAP[k] for block in range(3) for k in range(6)]
_JOINT_SIGN = [-1.0] * 6 + [1.0] * 12
POLICY_DIM, CRITIC_DIM = 117, 151


def _joint(x: torch.Tensor) -> torch.Tensor:
    return x[..., _JOINT_SRC] * x.new_tensor(_JOINT_SIGN)


def _legs(x: torch.Tensor, per_leg: int) -> torch.Tensor:
    """Permute a block of 6 legs × ``per_leg`` values."""
    return x.view(*x.shape[:-1], 6, per_leg)[..., LEG_SWAP, :].reshape(x.shape)


def _polar(x: torch.Tensor) -> torch.Tensor:
    return x * x.new_tensor([1.0, -1.0, 1.0])


def mirror_policy_obs(o: torch.Tensor) -> torch.Tensor:
    assert o.shape[-1] == POLICY_DIM, o.shape
    return torch.cat(
        [
            _joint(o[..., 0:18]),
            _joint(o[..., 18:36]),
            _joint(o[..., 36:54]),
            _joint(o[..., 54:72]),
            _polar(o[..., 72:75]),
            o[..., 75:78] * o.new_tensor([-1.0, 1.0, -1.0]),
            _polar(o[..., 78:81]),
            _legs(o[..., 81:93], 2),
            o[..., 93:111][..., _JOINT_SRC],
            _polar(o[..., 111:114]),
            o[..., 114:117] * o.new_tensor([1.0, -1.0, -1.0]),
        ],
        dim=-1,
    )


def mirror_critic_obs(o: torch.Tensor) -> torch.Tensor:
    assert o.shape[-1] == CRITIC_DIM, o.shape
    return torch.cat(
        [
            mirror_policy_obs(o[..., :117]),
            _polar(o[..., 117:120]),
            o[..., 120:121],
            _legs(o[..., 121:133], 2),
            _legs(o[..., 133:139], 1),
            o[..., 139:145],
            o[..., 145:147] * o.new_tensor([1.0, -1.0]),
            o[..., 147:151],
        ],
        dim=-1,
    )


def mirror_actions(a: torch.Tensor) -> torch.Tensor:
    return _joint(a)


@torch.no_grad()
def mirror_augmentation(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """rsl_rl ``data_augmentation_func``: returns [original; mirrored] observations and actions."""
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        n = obs.batch_size[0]
        obs_aug["policy"][n:] = mirror_policy_obs(obs["policy"])
        obs_aug["critic"][n:] = mirror_critic_obs(obs["critic"])
    actions_aug = None if actions is None else torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs_aug, actions_aug


# ---------------------------------------------------------------------------------------------------------------------
# S1+: generic mirror driven by ``mdp.layout`` (each block names its rule).
# ---------------------------------------------------------------------------------------------------------------------

_STRIP_PTS = (8, 3)  # forward × lateral points per leg
_SCAN = (11, 7)  # forward × lateral


def _mirror_block(kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind == "joint":
        return _joint(x)
    if kind == "mask":
        return x[..., _JOINT_SRC]
    if kind == "polar":
        return _polar(x)
    if kind == "axial":
        return x * x.new_tensor([-1.0, 1.0, -1.0])
    if kind == "cmd":
        return x * x.new_tensor([1.0, -1.0, -1.0])
    if kind == "same":
        return x
    if kind.startswith("legs:"):
        return _legs(x, int(kind.split(":")[1]))
    if kind.startswith("front:"):
        k = int(kind.split(":")[1])
        return torch.cat([x[..., k:], x[..., :k]], dim=-1)
    if kind.startswith("strip:"):
        y = x.view(*x.shape[:-1], 6, *_STRIP_PTS, int(kind.split(":")[1]))[..., LEG_SWAP, :, :, :].flip(-2)
        return y.reshape(x.shape)
    if kind == "scan":
        return x.view(*x.shape[:-1], *_SCAN).flip(-1).reshape(x.shape)
    if kind == "dr":  # stall, slope, payload, CoM x, CoM y, CoM z, latency, drop k
        return x * x.new_tensor([1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    raise ValueError(kind)


def mirror_layout(x: torch.Tensor, blocks, latent: torch.Tensor | None = None) -> torch.Tensor:
    """Mirror a vector laid out as ``blocks``. Live "latent" blocks take their values from ``latent`` (the learned
    features recomputed on the mirrored history, concatenated in block order); zero-held ones stay zero."""
    out, i, j = [], 0, 0
    for b in blocks:
        xb = x[..., i : i + b.dim]
        if b.mirror == "latent":
            if b.live_s1:
                out.append(latent[..., j : j + b.dim])
                j += b.dim
            else:
                out.append(xb)
        else:
            out.append(_mirror_block(b.mirror, xb))
        i += b.dim
    assert i == x.shape[-1], (i, x.shape)
    return torch.cat(out, dim=-1)


def mirror_frame(o_p: torch.Tensor) -> torch.Tensor:
    """Mirror the 111-D o_p frame (the first nine actor blocks)."""
    from .layout import ACTOR_BLOCKS

    return mirror_layout(o_p, ACTOR_BLOCKS[:9])


@torch.no_grad()
def mirror_augmentation_s1(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """rsl_rl ``data_augmentation_func`` for S1 / S2: groups policy (1,125 or 1,269 with H*), contract (683), mirror
    (latents)."""
    from .layout import BLOCKS_BY_DIM, CONTRACT_P_BLOCKS

    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        n = obs.batch_size[0]
        blocks = BLOCKS_BY_DIM[obs["policy"].shape[1]]
        own = obs["policy"][:, [i for b in blocks if b.mirror == "latent" and b.live_s1 for i in _cols(b)]]
        obs_aug["policy"][n:] = mirror_layout(obs["policy"], blocks, obs["mirror"])
        obs_aug["contract"][n:] = mirror_layout(obs["contract"], CONTRACT_P_BLOCKS)
        obs_aug["mirror"][n:] = own  # mirror of the mirrored sample = the original latents
    actions_aug = None if actions is None else torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs_aug, actions_aug


def _cols(block):
    from .layout import ACTOR

    return range(ACTOR[block.name].start, ACTOR[block.name].stop)


if __name__ == "__main__":
    # Self-check: the mirror is an involution on every layout.
    p, c, a = torch.randn(64, POLICY_DIM), torch.randn(64, CRITIC_DIM), torch.randn(64, 18)
    assert torch.equal(mirror_policy_obs(mirror_policy_obs(p)), p)
    assert torch.equal(mirror_critic_obs(mirror_critic_obs(c)), c)
    assert torch.equal(mirror_actions(mirror_actions(a)), a)
    # RF coxa ↔ LF coxa with sign flip; RF femur ↔ LF femur without.
    x = torch.zeros(18)
    x[0], x[6] = 1.0, 2.0
    y = mirror_actions(x)
    assert y[3] == -1.0 and y[9] == 2.0 and y.abs().sum() == 3.0
    # S1 layouts: involution with consistent latents, and agreement with the S0b mirror on the old 117 dims.
    from .layout import ACTOR_BLOCKS, ACTOR_DIM, CONTRACT_P_BLOCKS, CONTRACT_P_DIM, LATENT_DIM

    x, lat = torch.randn(64, ACTOR_DIM), torch.randn(64, LATENT_DIM)
    for b in ACTOR_BLOCKS:  # zero-held blocks are zero
        if not b.live_s1:
            from .layout import ACTOR

            x[:, ACTOR[b.name]] = 0.0
    own = torch.cat([x[:, list(_cols(b))] for b in ACTOR_BLOCKS if b.mirror == "latent" and b.live_s1], dim=-1)
    m = mirror_layout(x, ACTOR_BLOCKS, lat)
    assert torch.equal(mirror_layout(m, ACTOR_BLOCKS, own), x)
    assert torch.equal(m[:, :117], mirror_policy_obs(x[:, :117]))
    # S2 layout: H* (one height per strip point) mirrors like the material strips, and stays an involution.
    from .layout import S2_ACTOR, S2_ACTOR_BLOCKS

    x2 = torch.cat([x, torch.randn(64, 144)], dim=-1)
    m2 = mirror_layout(x2, S2_ACTOR_BLOCKS, lat)
    assert torch.equal(mirror_layout(m2, S2_ACTOR_BLOCKS, own), x2) and torch.equal(m2[:, :ACTOR_DIM], m)
    h, hm = x2[:, S2_ACTOR["h_star"]].view(64, 6, 8, 3), m2[:, S2_ACTOR["h_star"]].view(64, 6, 8, 3)
    assert torch.equal(hm[:, 3, :, 0], h[:, 0, :, 2])  # LF's left-most column = RF's right-most
    pc = torch.randn(64, CONTRACT_P_DIM)
    assert torch.equal(mirror_layout(mirror_layout(pc, CONTRACT_P_BLOCKS), CONTRACT_P_BLOCKS), pc)
    print("symmetry self-check OK")
