# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S0b terrain (paradigm spec §4.2): generator config, command-referenced curriculum and a height map.

The generator is laid out for 15 difficulty levels and builds the first 10 (``difficulty_range = (0, 10/15)``):
level ℓ gets difficulty (ℓ + U)/15 either way, so adding the 5 harder levels later is ``num_rows=15,
difficulty_range=(0, 1)`` (plus stairs), with levels 0–9 unchanged in distribution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.utils.configclass import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

LEVELS_BUILT, LEVELS_DESIGNED = 10, 15


def scaled_rough_terrain(difficulty: float, cfg: HfScaledRoughTerrainCfg):
    """Isaac Lab's random-uniform height field ignores difficulty; scale its amplitude with it."""
    a = round(difficulty * cfg.max_amplitude / cfg.noise_step) * cfg.noise_step
    return hf_terrains.random_uniform_terrain(difficulty, cfg.replace(noise_range=(-a, a)))


@configclass
class HfScaledRoughTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    function: str = "{DIR}.terrain:scaled_rough_terrain"
    max_amplitude: float = 0.0225  # ±, at difficulty 1 (level 15); ±1.5 cm at level 10
    noise_range: tuple[float, float] = (0.0, 0.0)  # set per tile from difficulty


_BORDER = 0.25
S0B_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(4.0, 4.0),  # ≥ 2 m from spawn at 0.2 m/s × 10 s
    border_width=5.0,
    num_rows=LEVELS_BUILT,
    num_cols=20,
    horizontal_scale=0.05,  # 5 cm cells: stride is 12–19 cm
    vertical_scale=0.002,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, LEVELS_BUILT / LEVELS_DESIGNED),
    seed=0,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "rough": HfScaledRoughTerrainCfg(
            proportion=0.25, noise_step=0.002, downsampled_scale=0.1, border_width=_BORDER
        ),
        # tan 15° at difficulty 1 → 10° at level 10
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, math.tan(math.radians(15.0))), platform_width=1.0, border_width=_BORDER
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, math.tan(math.radians(15.0))), platform_width=1.0, border_width=_BORDER
        ),
        # heights ∈ {±h, ±h/2}, h up to 3 cm at difficulty 1 → 2 cm at level 10
        "bumps": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.1, 0.4),
            obstacle_height_range=(0.0, 0.03),
            num_obstacles=40,
            platform_width=1.0,
            border_width=_BORDER,
        ),
    },
)


class HeightMap:
    """Terrain height and surface normal at world xy, from one downward raycast of the terrain mesh at startup."""

    def __init__(self, mesh_prim_path: str, device, res: float = 0.02):
        from pxr import UsdGeom

        import isaaclab.sim as sim_utils

        prim = sim_utils.get_current_stage().GetPrimAtPath(mesh_prim_path)
        mesh = UsdGeom.Mesh(prim)
        xform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        pts = (np.c_[pts, np.ones(len(pts))] @ xform.T)[:, :3].astype(np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        assert (np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3).all(), "terrain mesh must be triangles"
        wp_mesh = convert_to_warp_mesh(pts, faces, device=str(device))

        self.res, self.device = res, device
        self.x0, self.y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        nx = int((pts[:, 0].max() - self.x0) / res) + 1
        ny = int((pts[:, 1].max() - self.y0) / res) + 1
        xs = self.x0 + res * torch.arange(nx, device=device, dtype=torch.float32)
        ys = self.y0 + res * torch.arange(ny, device=device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).view(-1, 2)
        z0 = float(pts[:, 2].max()) + 1.0
        self.h = torch.empty(nx * ny, device=device)
        for i in range(0, len(grid), 2**22):  # chunked: keeps warp's buffers small
            g = grid[i : i + 2**22]
            starts = torch.cat([g, torch.full_like(g[:, :1], z0)], dim=-1)
            dirs = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(starts).contiguous()
            hits = raycast_mesh(starts.contiguous(), dirs, wp_mesh)[0]
            self.h[i : i + len(g)] = hits[:, 2]
        assert torch.isfinite(self.h).all(), "height-map ray missed the terrain"
        self.h = self.h.view(nx, ny)
        self.nx, self.ny = nx, ny

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear terrain height at world xy (..., 2) → (...)."""
        fx = ((xy[..., 0] - self.x0) / self.res).clamp(0, self.nx - 1.001)
        fy = ((xy[..., 1] - self.y0) / self.res).clamp(0, self.ny - 1.001)
        ix, iy = fx.long(), fy.long()
        ax, ay = fx - ix, fy - iy
        h = self.h
        return (
            h[ix, iy] * (1 - ax) * (1 - ay)
            + h[ix + 1, iy] * ax * (1 - ay)
            + h[ix, iy + 1] * (1 - ax) * ay
            + h[ix + 1, iy + 1] * ax * ay
        )

    def normal(self, xy: torch.Tensor) -> torch.Tensor:
        """Unit surface normal at world xy (..., 2) → (..., 3), central differences over one cell."""
        d = self.res
        ex = torch.tensor([d, 0.0], device=xy.device)
        ey = torch.tensor([0.0, d], device=xy.device)
        dhdx = (self.height(xy + ex) - self.height(xy - ex)) / (2 * d)
        dhdy = (self.height(xy + ey) - self.height(xy - ey)) / (2 * d)
        n = torch.stack([-dhdx, -dhdy, torch.ones_like(dhdx)], dim=-1)
        return n / n.norm(dim=-1, keepdim=True)


def terrain_levels_cmd(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Terrain curriculum on the commanded path length (Isaac Lab's ``terrain_levels_vel`` promotes at tile/2 = 2 m,
    out of reach at 0.05–0.20 m/s).

    Up: survived, commanded path ≥ 0.3 m and displacement ≥ 0.8 × commanded path. Down: fell, or displacement
    < 0.4 × commanded path. Standing-only episodes (path < 0.3 m) move only down, on a fall.
    """
    terrain = env.scene.terrain
    dist = torch.norm(env.scene["robot"].data.root_pos_w.torch[env_ids, :2] - env.spawn_xy[env_ids], dim=1)
    need = env.cmd_dist[env_ids]
    fell = env.termination_manager.terminated[env_ids]
    up = ~fell & (need >= 0.3) & (dist >= 0.8 * need)
    down = fell | (dist < 0.4 * need)
    terrain.update_env_origins(env_ids, up, down & ~up)
    return torch.mean(terrain.terrain_levels.float())


def pin_terrain_level(env: ManagerBasedRLEnv, level: int) -> None:
    """Evaluation: put every env on terrain level ``level`` (its own column type); takes effect at the next reset."""
    terrain = env.scene.terrain
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
