# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S1 observation layouts — the single source of truth (paradigm spec §2.9, §3.5; S1 plan §2).

Actor: the union of every arm's layout except B1-K's, widened once at S1 from π_S0's 117-D input. The old 117 dims
come first (o_p | V_0 | twist_cmd), so widening only appends zero-init columns (§4.3). V_0 is not in the §2.9
table: it is π_S0's velocity input, kept as an old column and retired by per-episode dropout during S1 (deviation
D23). Blocks not live in S1 are held at exactly zero.

Critic: contract P (683-D, §3.5), identical for every arm, concatenated after sg[actor obs].

Each block names its mirror rule (left–right reflection, ``mdp.symmetry``):
  joint     18 joints, legs swapped, coxa sign flipped
  mask      18 joints, legs swapped, no sign
  polar     3-vector, y negated
  axial     angular velocity (−, +, −)
  cmd       (v_x, v_y, ω_z) → (v_x, −v_y, −ω_z)
  legs:k    6 legs × k values, legs swapped
  strip:k   6 legs × 8 forward × 3 lateral × k values (4 material values, or 1 height); legs swapped, lateral order
            reversed
  scan      11 forward × 7 lateral heights; lateral order reversed
  front:k   2 front legs (L1, R1) × k values, swapped
  dr        DR vector (stall, slope, payload, CoM x, CoM y, CoM z, latency, drop k); CoM y negated
  latent    learned features: no analytic mirror — the env recomputes them on the mirrored history
  same      unchanged
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    name: str
    dim: int
    mirror: str
    live_s1: bool


ACTOR_BLOCKS: tuple[Block, ...] = (
    # π_S0's 117 dims, unchanged order.
    Block("q", 18, "joint", True),
    Block("qd", 18, "joint", True),
    Block("track_err", 18, "joint", True),
    Block("last_action", 18, "joint", True),
    Block("imu_acc", 3, "polar", True),
    Block("imu_gyro", 3, "axial", True),
    Block("gravity", 3, "polar", True),
    Block("clock", 12, "legs:2", True),
    Block("read_mask", 18, "mask", True),
    Block("v0", 3, "polar", True),  # retired by per-episode dropout in S1 (D23)
    Block("twist_cmd", 3, "cmd", True),
    # New at S1 (zero-init columns).
    Block("z_p", 64, "latent", True),
    Block("e_hat", 15, "latent", True),  # [v̂_b (3), f̂ (6 × 2)], gated by g_ê until V passes v7 §4.3
    Block("m_star", 576, "strip:4", False),  # O-mat only, from S2
    Block("z_s_z_f", 20, "latent", False),  # B1-M, from S2
    Block("tokens", 192, "latent", False),  # u (B2) / ẑ (F, B3), from S2
    Block("v_front", 128, "front:64", False),  # v_ℓ ⊙ m_ℓ, L1/R1 (sampled features: recomputed, see §2 note)
    Block("m_front", 2, "front:1", False),
    Block("sigma_front", 8, "front:4", False),
    Block("rem", 3, "same", False),
)

CONTRACT_P_BLOCKS: tuple[Block, ...] = (
    Block("v_true", 3, "polar", True),
    Block("height", 1, "same", True),
    Block("force_nt", 12, "legs:2", True),
    Block("contact", 6, "legs:1", True),
    Block("strips", 576, "strip:4", True),
    Block("height_scan", 77, "scan", True),
    Block("dr", 8, "dr", True),
)

# S2 layout = the S1 layout ⊕ H* (O-geo only, research D37): terrain height at the strip points minus the terrain height
# at the leg's nominal foothold. Added after S1, so the 1,125-D trunks are widened by its zero columns at the S2 branch
# (``hexasim.rl.s1_ppo.widen_checkpoint``); the S1 task keeps the 1,125-D layout.
S2_ACTOR_BLOCKS: tuple[Block, ...] = (*ACTOR_BLOCKS, Block("h_star", 144, "strip:1", False))

OLD_ACTOR_DIM = 117  # π_S0


def offsets(blocks: tuple[Block, ...]) -> dict[str, slice]:
    out, i = {}, 0
    for b in blocks:
        out[b.name] = slice(i, i + b.dim)
        i += b.dim
    return out


def total(blocks: tuple[Block, ...]) -> int:
    return sum(b.dim for b in blocks)


ACTOR = offsets(ACTOR_BLOCKS)
S2_ACTOR = offsets(S2_ACTOR_BLOCKS)  # ⊇ ACTOR: same offsets, plus h_star
S2_ACTOR_DIM = total(S2_ACTOR_BLOCKS)
BLOCKS_BY_DIM = {total(b): b for b in (ACTOR_BLOCKS, S2_ACTOR_BLOCKS)}  # actor layout from the observation width
# Blocks an arm reads beyond the blind (B0) layout; every other non-S1 block is held at exactly zero.
ARM_BLOCK = {"B0": None, "O-mat": "m_star", "O-geo": "h_star"}
CONTRACT_P = offsets(CONTRACT_P_BLOCKS)
ACTOR_DIM = total(ACTOR_BLOCKS)
CONTRACT_P_DIM = total(CONTRACT_P_BLOCKS)
LATENT_BLOCKS = tuple(b for b in ACTOR_BLOCKS if b.mirror == "latent" and b.live_s1)  # z_p, e_hat
LATENT_DIM = total(LATENT_BLOCKS)

assert ACTOR["twist_cmd"].stop == OLD_ACTOR_DIM
assert ACTOR_DIM == 1125, ACTOR_DIM
assert S2_ACTOR_DIM == 1269, S2_ACTOR_DIM
assert CONTRACT_P_DIM == 683, CONTRACT_P_DIM

# Material strip geometry (§3.5 / O's M*): per leg 8 forward × 3 lateral points, 3 cm spacing, forward −6 … +15 cm,
# lateral −3 … +3 cm, anchored at the leg's rest-pose toe in the body frame; values (μ_s, μ_d, log k, log c).
STRIP_FWD = tuple(-0.06 + 0.03 * i for i in range(8))
STRIP_LAT = (-0.03, 0.0, 0.03)
# Height scan around the body (§3.5): 11 forward × 7 lateral, 5 cm, centred on the base.
SCAN_FWD = tuple(0.05 * (i - 5) for i in range(11))
SCAN_LAT = tuple(0.05 * (j - 3) for j in range(7))
# Probe local height grid (§3.2): 3 × 3 at 4 cm, body-frame axes, centred on the point.
LOCAL_GRID = tuple((0.04 * i, 0.04 * j) for i in (-1, 0, 1) for j in (-1, 0, 1))
