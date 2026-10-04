# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env foot contact material (v7 §5.6): friction, restitution and compliant-contact stiffness/damping.

The terrain is one mesh with one material (μ 1, restitution 1, combine multiply), so the per-env contact physics
lives on the foot side: μ_eff = μ_foot · 1 = the sampled value, e_eff likewise. PhysX's tensor API can change
friction/restitution but not compliance, so every property goes into USD materials bound before the sim starts
(prestartup; needs ``replicate_physics=False``). ``num_buckets`` distinct materials, one per env, drawn uniformly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def foot_materials(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    num_buckets: int,
    static_friction_range: tuple[float, float],
    dynamic_ratio_range: tuple[float, float],
    restitution_range: tuple[float, float],
    rigid_fraction: float,
    stiffness_range: tuple[float, float],
    damping_range: tuple[float, float],
):
    """Sets ``env.foot_material`` (N, 5) = (μ_s, μ_d, e, log10(1 + k/1000), log10(1 + c)); k, c = 0 when rigid.

    μ_d = μ_s × U(dynamic_ratio_range) ≤ μ_s. k, c log-uniform (N/m, N·s/m) for compliant buckets.
    """
    from pxr import UsdPhysics, UsdShade  # noqa: PLC0415 — USD must load after the simulation app starts

    assert not env.cfg.scene.replicate_physics, "per-env USD materials need replicate_physics=False"
    K = num_buckets
    u = lambda r: torch.empty(K).uniform_(*r)  # noqa: E731
    logu = lambda r: 10.0 ** torch.empty(K).uniform_(math.log10(r[0]), math.log10(r[1]))  # noqa: E731
    mu_s = u(static_friction_range)
    mu_d = mu_s * u(dynamic_ratio_range)
    rest = u(restitution_range)
    compliant = torch.rand(K) >= rigid_fraction
    k = logu(stiffness_range) * compliant
    c = logu(damping_range) * compliant
    for b in range(K):
        cfg = PhysxRigidBodyMaterialCfg(
            static_friction=mu_s[b].item(),
            dynamic_friction=mu_d[b].item(),
            restitution=rest[b].item(),
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            compliant_contact_stiffness=k[b].item() if compliant[b] else None,
            compliant_contact_damping=c[b].item() if compliant[b] else None,
        )
        cfg.func(f"/World/FootMaterials/bucket_{b}", cfg)

    stage = sim_utils.get_current_stage()
    robot0 = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    is_tibia = lambda p: "tibia" in p.GetName() and p.HasAPI(UsdPhysics.RigidBodyAPI)  # noqa: E731
    tibias = [p.GetPath().pathString for p in sim_utils.get_all_matching_child_prims(robot0.GetPath(), is_tibia)]
    assert len(tibias) == 6, tibias
    # The tibia colliders are instance proxies (instanceable mesh prims), which cannot be edited, so
    # ``sim_utils.bind_physics_material`` (applies to collider descendants) silently binds nothing. Bind on the
    # tibia rigid-body prim instead; physics-purpose binding resolution inherits it down to the collider.
    mats = [UsdShade.Material(stage.GetPrimAtPath(f"/World/FootMaterials/bucket_{b}")) for b in range(K)]
    bucket = torch.randint(K, (env.num_envs,))
    for i in range(env.num_envs):
        for t in tibias:
            prim = stage.GetPrimAtPath(t.replace("/env_0/", f"/env_{i}/"))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mats[bucket[i]], UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
            )
    table = torch.stack([mu_s, mu_d, rest, torch.log10(1 + k / 1000.0), torch.log10(1 + c)], dim=1)
    env.foot_material = table[bucket].to(env.device)


def terrain_materials(env: ManagerBasedEnv, env_ids: torch.Tensor | None, seed: int):
    """S1+: spatial terrain materials (``mdp.materials``): sets ``env.material_field`` and splits the terrain mesh into
    per-entry submeshes. Prestartup; the field seed is fixed across runs so every arm and seed sees one terrain."""
    from .materials import build_terrain_materials  # noqa: PLC0415

    env.material_field = build_terrain_materials(
        env.scene.terrain.cfg.prim_path + "/terrain/mesh", env.device, seed, env.cfg.material_entries
    )
