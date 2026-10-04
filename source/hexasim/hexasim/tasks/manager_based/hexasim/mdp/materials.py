# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spatial terrain materials for S1+ — hybrid of USD submeshes and per-foot tensor writes (phase2-progress §2.2).

* Field: per 4 m cell, Voronoi sites each carrying a library entry; cell type 1 site (uniform, 25 %), 2 (one straight
  boundary, 35 %), 3 collinear (stripes, 15 %) or 4–8 (mosaic, 25 %); per-cell transition width w ~ U[0, 0.8] m
  (sharp at the low end). Near the bisector of the two nearest sites the second site's weight is
  ½·(1 − clamp(2b/w, 0, 1)), b = distance to the bisector. Stored on a 2 cm grid (same origin as ``HeightMap``) as
  (id_a, id_b, weight_b). Static for the run; independent of the height field (v7 §5.4).
* USD (static, renderable): one visual submesh per entry (triangle → dominant entry at its centroid, semantic label).
  The original terrain mesh stays the single collider, rigid, μ 1, multiply (spike T6: a compliant ground overrides
  the foot's written stiffness and its damping has no visible effect → damping stays on the foot, per env, D25).
* Per-foot physics (dynamic): μ_s, μ_d and stiffness k (PhysX restitution slot = −k) are written to the tibia shapes
  from the field blend and the env's per-episode palette, only for envs whose foot target moved by more than a
  quantum while that foot is low or in contact (write cost ≈ 55 µs per env; spike T4).

Library values are provisional until the bench (v7 §5.3); approved by the user 2026-09-24. "Rigid" = k 1e6.
The last two entries (soft ground, research D39) were added after S1: a task uses the first ``cfg.material_entries``
entries (S1: 10, S2: 12), so the S1 field and palette stay reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

RIGID_K = 1.0e6
SOFT_SOIL_K = (4e2, 1e3)  # N/m nominal; PhysX applies k per contact point (effective ≈ 2.5 × nominal)
WET_CLAY_K = (6e2, 2e3)


@dataclass(frozen=True)
class Entry:
    name: str
    mu_s: tuple[float, float]
    k: tuple[float, float] | None  # N/m (log-uniform); None = rigid
    shiny: bool


LIBRARY: tuple[Entry, ...] = (
    Entry("concrete", (0.8, 1.0), None, False),
    Entry("rubber_mat", (0.9, 1.1), (2e4, 5e4), False),
    Entry("ptfe", (0.2, 0.3), None, False),
    Entry("talc_graphite", (0.25, 0.4), None, False),
    Entry("dry_dust", (0.35, 0.5), None, False),
    Entry("foam", (0.7, 0.9), (2e3, 8e3), False),
    Entry("wet_tile", (0.2, 0.35), None, True),
    Entry("wet_rubber", (0.6, 0.8), (1e4, 3e4), True),
    Entry("glossy_grit", (0.9, 1.1), None, True),
    Entry("polished_coating", (0.8, 1.0), None, True),
    # Soft ground (D39): elastic only, no permanent sinkage. k ranges from the solver check (phase2-progress §2.3.8).
    Entry("soft_soil", (0.4, 0.7), SOFT_SOIL_K, False),
    Entry("wet_clay", (0.25, 0.5), WET_CLAY_K, True),
)
S1_ENTRIES = 10
MU_D_RATIO = (0.8, 1.0)
# Visual-only override for the per-entry submeshes, keyed by entry name: the demo scripts put one MDL per
# entry here (``hexasim.tasks.manager_based.hexasim.demo_look``). None everywhere else, so training and eval
# runs keep the two flat greys below and are unaffected.
VISUAL_OVERRIDE: dict | None = None
CELL = 4.0
MAX_WIDTH = 0.8
LAYOUT_P = (0.25, 0.35, 0.15, 0.25)  # uniform, boundary, stripes, mosaic


class MaterialField:
    """(id_a, id_b, weight_b) on a regular grid; ``lookup(xy)`` → nearest-cell values at world xy (..., 2)."""

    def __init__(
        self, x0: float, y0: float, x1: float, y1: float, res: float, device, seed: int = 0, entries: int = S1_ENTRIES
    ):
        self.x0, self.y0, self.res, self.device, self.entries = x0, y0, res, device, entries
        self.nx = int((x1 - x0) / res) + 1
        self.ny = int((y1 - y0) / res) + 1
        g = torch.Generator().manual_seed(seed)
        cx, cy = math.ceil((x1 - x0) / CELL), math.ceil((y1 - y0) / CELL)
        self.sites, self.site_entry, self.site_cell = self._sites(cx, cy, g)
        self.cell_width = torch.rand(cx * cy, generator=g) * MAX_WIDTH
        self.cx, self.cy = cx, cy
        self.id_a = torch.empty(self.nx, self.ny, dtype=torch.uint8, device=device)
        self.id_b = torch.empty_like(self.id_a)
        self.w = torch.empty(self.nx, self.ny, dtype=torch.float16, device=device)
        self._fill()

    # ------------------------------------------------------------------ generation
    def _sites(self, cx: int, cy: int, g: torch.Generator):
        sites, entry, cell = [], [], []
        rnd = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
        for i in range(cx):
            for j in range(cy):
                c = torch.tensor([self.x0 + (i + 0.5) * CELL, self.y0 + (j + 0.5) * CELL])
                kind = int(torch.multinomial(torch.tensor(LAYOUT_P), 1, generator=g))
                d = torch.tensor([math.cos(a := 2 * math.pi * rnd(1).item()), math.sin(a)])
                if kind == 0:
                    pts = [c + (rnd(2) - 0.5)]
                elif kind == 1:  # straight boundary: bisector through a point within ±1 m of the centre
                    m = c + (rnd(2) - 0.5) * 2.0
                    pts = [m - 0.75 * d, m + 0.75 * d]
                elif kind == 2:  # stripes: 3 collinear sites 1–1.5 m apart
                    s = 1.0 + 0.5 * rnd(1).item()
                    pts = [c - s * d, c, c + s * d]
                else:
                    n = 4 + int(torch.randint(0, 5, (1,), generator=g))
                    pts = [c + (rnd(2) - 0.5) * CELL for _ in range(n)]
                ent = []
                for _ in pts:  # neighbouring sites of one cell never share an entry (a transition must exist)
                    e = int(torch.randint(0, self.entries, (1,), generator=g))
                    while self.entries > 1 and ent and e == ent[-1]:  # entries == 1: no other entry to pick
                        e = int(torch.randint(0, self.entries, (1,), generator=g))
                    ent.append(e)
                sites += pts
                entry += ent
                cell += [i * cy + j] * len(pts)
        return torch.stack(sites), torch.tensor(entry, dtype=torch.uint8), torch.tensor(cell)

    def _fill(self):
        dev = self.device
        sites, ent = self.sites.to(dev), self.site_entry.to(dev)
        width = self.cell_width.to(dev)
        # Candidate sites per cell: its own and its 8 neighbours' (padded to a fixed count).
        cell_ij = torch.stack((self.site_cell // self.cy, self.site_cell % self.cy), dim=1)
        cand = [[] for _ in range(self.cx * self.cy)]
        for s, (i, j) in enumerate(cell_ij.tolist()):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < self.cx and 0 <= b < self.cy:
                        cand[a * self.cy + b].append(s)
        k = max(len(c) for c in cand)
        cand_t = torch.full((self.cx * self.cy, k), -1, dtype=torch.long)
        for c, lst in enumerate(cand):
            cand_t[c, : len(lst)] = torch.tensor(lst)
        cand_t = cand_t.to(dev)
        xs = self.x0 + self.res * torch.arange(self.nx, device=dev)
        for i0 in range(0, self.nx, 256):  # chunked rows
            x = xs[i0 : i0 + 256]
            ys = self.y0 + self.res * torch.arange(self.ny, device=dev)
            p = torch.stack(torch.meshgrid(x, ys, indexing="ij"), dim=-1)  # (r, ny, 2)
            ci = ((p[..., 0] - self.x0) / CELL).long().clamp(0, self.cx - 1)
            cj = ((p[..., 1] - self.y0) / CELL).long().clamp(0, self.cy - 1)
            cell = ci * self.cy + cj
            c = cand_t[cell]  # (r, ny, k)
            s = sites[c.clamp(min=0)]
            d2 = ((s - p[..., None, :]) ** 2).sum(-1).masked_fill(c < 0, float("inf"))
            dist, idx = d2.topk(2, dim=-1, largest=False)
            (d_a, d_b), (ia, ib) = dist.unbind(-1), idx.unbind(-1)
            sa = c.gather(-1, ia[..., None])[..., 0]
            sb = c.gather(-1, ib[..., None])[..., 0]
            sep = (sites[sa] - sites[sb]).norm(dim=-1).clamp(min=1e-6)
            b = (d_b - d_a) / (2 * sep)  # distance to the bisector
            w = width[cell]
            wb = torch.where(w > 1e-6, 0.5 * (1 - (2 * b / w.clamp(min=1e-6)).clamp(0, 1)), torch.zeros_like(b))
            self.id_a[i0 : i0 + 256] = ent[sa]
            self.id_b[i0 : i0 + 256] = ent[sb]
            self.w[i0 : i0 + 256] = wb.half()

    # ------------------------------------------------------------------ query
    def _idx(self, xy: torch.Tensor):
        ix = ((xy[..., 0] - self.x0) / self.res).round().long().clamp(0, self.nx - 1)
        iy = ((xy[..., 1] - self.y0) / self.res).round().long().clamp(0, self.ny - 1)
        return ix, iy

    def lookup(self, xy: torch.Tensor):
        """(id_a, id_b, weight_b) at world xy (..., 2)."""
        ix, iy = self._idx(xy)
        return self.id_a[ix, iy].long(), self.id_b[ix, iy].long(), self.w[ix, iy].float()

    def dominant(self, xy: torch.Tensor) -> torch.Tensor:
        a, b, w = self.lookup(xy)
        return torch.where(w > 0.5, b, a)


class Palette:
    """Per-env, per-entry physics values (μ_s, μ_d, log10 k), redrawn at reset inside each entry's range."""

    def __init__(self, num_envs: int, device, entries: int = S1_ENTRIES):
        lib = LIBRARY[:entries]
        self.v = torch.zeros(num_envs, entries, 3, device=device)
        lo = torch.tensor([[e.mu_s[0], math.log10(e.k[0]) if e.k else math.log10(RIGID_K)] for e in lib])
        hi = torch.tensor([[e.mu_s[1], math.log10(e.k[1]) if e.k else math.log10(RIGID_K)] for e in lib])
        self.lo, self.hi = lo.to(device), hi.to(device)
        self.redraw(torch.arange(num_envs, device=device))

    def redraw(self, env_ids: torch.Tensor):
        n = len(env_ids)
        e = self.v.shape[1]
        u = torch.rand(n, e, 2, device=self.v.device)
        x = self.lo + u * (self.hi - self.lo)
        ratio = MU_D_RATIO[0] + torch.rand(n, e, device=self.v.device) * (MU_D_RATIO[1] - MU_D_RATIO[0])
        self.v[env_ids] = torch.stack((x[..., 0], x[..., 0] * ratio, x[..., 1]), dim=-1)

    def at(self, field: MaterialField, xy: torch.Tensor) -> torch.Tensor:
        """(μ_s, μ_d, log10 k) at world xy (N, ..., 2), blended with the field weight → (N, ..., 3)."""
        a, b, w = field.lookup(xy)
        n = xy.shape[0]
        flat = lambda i: i.reshape(n, -1)  # noqa: E731
        pa = self.v.gather(1, flat(a)[..., None].expand(-1, -1, 3))
        pb = self.v.gather(1, flat(b)[..., None].expand(-1, -1, 3))
        wb = flat(w)[..., None]
        return ((1 - wb) * pa + wb * pb).reshape(*xy.shape[:-1], 3)


class FootWriter:
    """Writes per-foot (μ_s, μ_d, −k) into the tibia collision shapes via the PhysX tensor API (spike T1–T5).

    ``update`` computes each foot's target from the field; envs where some foot is low (≤ ``low_z`` above terrain)
    or in contact and its target moved by > (``dmu``, ``dlogk``) since the last write get one write (whole env row;
    cost ≈ 55 µs per env). ``current`` (N, 6, 3) holds the values in effect.
    """

    def __init__(self, env, dmu: float = 0.01, dlogk: float = 0.05, low_z: float = 0.02):
        import warp as wp  # noqa: PLC0415

        self.wp = wp
        robot = env.scene["robot"]
        view = robot.root_view
        paths = view.link_paths[0]
        per_body = [robot._physics_sim_view.create_rigid_body_view(p).max_shapes for p in paths]
        assert sum(per_body) == view.max_shapes, (per_body, view.max_shapes)
        starts = [sum(per_body[:i]) for i in range(len(per_body))]
        names = [p.split("/")[-1] for p in paths]
        self.cols = []
        for b in env._tibia_body_ids:
            j = names.index(robot.body_names[b])
            self.cols.append(list(range(starts[j], starts[j] + per_body[j])))
        self.view = view
        self.mat = wp.to_torch(view.get_material_properties()).clone()  # (N, S, 3), CPU
        self.current = torch.full((env.num_envs, 6, 3), float("nan"), device=env.device)
        self.dmu, self.dlogk, self.low_z = dmu, dlogk, low_z
        self.writes = 0  # envs written since the last log read

    def write(self, env_ids: torch.Tensor, values: torch.Tensor):
        """values (len(env_ids), 6, 3) = (μ_s, μ_d, log10 k)."""
        ids_c, v = env_ids.cpu(), values.cpu()
        for leg, cols in enumerate(self.cols):
            for c in cols:
                self.mat[ids_c, c, 0] = v[:, leg, 0]
                self.mat[ids_c, c, 1] = v[:, leg, 1]
                self.mat[ids_c, c, 2] = -(10.0 ** v[:, leg, 2])  # restitution slot < 0 ⇒ compliant, k = −value
        self.view.set_material_properties(
            self.wp.from_torch(self.mat, dtype=self.wp.float32), self.wp.from_torch(ids_c.to(torch.int32))
        )
        self.current[env_ids] = values
        self.writes += len(env_ids)

    def update(self, target: torch.Tensor, low_or_contact: torch.Tensor):
        """target (N, 6, 3); low_or_contact (N, 6) bool."""
        d = (target - self.current).abs()
        moved = (d[..., 0] > self.dmu) | (d[..., 1] > self.dmu) | (d[..., 2] > self.dlogk) | d.isnan().any(-1)
        need = (moved & low_or_contact).any(dim=1)
        ids = need.nonzero().squeeze(1)
        if len(ids):
            # Feet that do not need a write keep their value in effect (NaN = never written → take the target).
            keep = ~(moved & low_or_contact)[ids] & ~self.current[ids].isnan().any(-1)
            vals = torch.where(keep[..., None], self.current[ids], target[ids])
            self.write(ids, vals)


def strip_course(env, start: float = 0.5, length: float = 0.5, low=(0.25, 0.22, 6.0), high=(0.9, 0.85, 6.0)):
    """Evaluation-only transition course (phase2-progress §2.3): replaces the env's material field by one strip of
    ``low`` (μ_s, μ_d, log10 k) at world x ∈ [start, start + length) m ahead of each env's spawn, ``high`` elsewhere.
    Patches ``env.palette.at``, so foot physics, probe labels, contract P and O's M* all see the course. Needs flat
    tiles and reset yaw 0 (the strip lies across world x)."""
    lo, hi = torch.tensor(low, device=env.device), torch.tensor(high, device=env.device)

    def at(field, xy: torch.Tensor) -> torch.Tensor:
        s = xy[..., 0] - env.spawn_xy[:, 0].view(-1, *([1] * (xy.dim() - 2)))
        return torch.where(((s >= start) & (s < start + length)).unsqueeze(-1), lo, hi)

    env.palette.at = at


def build_terrain_materials(mesh_path: str, device, seed: int, entries: int = S1_ENTRIES, res: float = 0.02):
    """Build the field over the terrain mesh's extent (same origin as ``HeightMap``) and add one visual submesh per
    library entry (dominant entry at each triangle's centroid, semantic label = entry). The source mesh stays the only
    collider (uniform rigid ground, μ 1, multiply: per-material physics lives on the feet) and is hidden. One collider:
    per-entry colliders each span the whole terrain and cost ≈ 60 ms per step at 4,096 envs (phase2-progress §2.2.5).
    Returns the field."""
    import numpy as np
    import trimesh

    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415 — USD must load after the simulation app starts

    import isaaclab.sim as sim_utils
    from isaaclab.terrains.utils import create_prim_from_mesh

    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(mesh_path)
    mesh = UsdGeom.Mesh(prim)
    xform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    pts_w = (np.c_[pts, np.ones(len(pts))] @ xform.T)[:, :3]
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    lo, hi = pts_w.min(axis=0), pts_w.max(axis=0)
    field = MaterialField(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]), res, device, seed, entries)
    cen = torch.tensor(pts_w[faces].mean(axis=1)[:, :2], dtype=torch.float32, device=field.device)
    dom = field.dominant(cen).cpu().numpy()
    parent = mesh_path.rsplit("/", 2)[0]
    assigned = 0
    for e, entry in enumerate(LIBRARY[:entries]):
        sel = faces[dom == e]
        assigned += len(sel)
        path = f"{parent}/materials/{entry.name}"
        if len(sel) == 0:
            continue
        sub = trimesh.Trimesh(vertices=pts_w, faces=sel, process=False)
        sub.remove_unreferenced_vertices()
        color = (0.25, 0.25, 0.3) if entry.shiny else (0.45, 0.42, 0.38)
        vis = (VISUAL_OVERRIDE or {}).get(entry.name) or sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        create_prim_from_mesh(path, sub, visual_material=vis)
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(f"{path}/mesh")).CreateCollisionEnabledAttr(False)  # visual only
        sim_utils.add_labels(stage.GetPrimAtPath(f"{path}/mesh"), labels=[entry.name], instance_name="class")
    assert assigned == len(faces), (assigned, len(faces))
    UsdGeom.Imageable(prim).MakeInvisible()  # still the collider
    return field


if __name__ == "__main__":
    # Self-check (CPU, no sim): blend weight range, dominant id consistency, library/palette ranges.
    f = MaterialField(0.0, 0.0, 16.0, 12.0, 0.02, "cpu", seed=1)
    assert f.w.float().min() >= 0 and f.w.float().max() <= 0.5 + 1e-3
    xy = torch.rand(4, 1000, 2) * torch.tensor([16.0, 12.0])
    a, b, w = f.lookup(xy)
    assert torch.equal(f.dominant(xy), torch.where(w > 0.5, b, a))
    # A point on a site is fully that site's entry.
    s0 = f.sites[0]
    a0, _, w0 = f.lookup(s0[None])
    assert a0.item() == f.site_entry[0].item() and w0.item() < 1e-3
    pal = Palette(4, "cpu")
    v = pal.at(f, xy)
    assert (v[..., 1] <= v[..., 0] + 1e-6).all() and v[..., 0].min() >= 0.2 - 1e-6 and v[..., 0].max() <= 1.1 + 1e-6
    # The soft entries only exist in a 12-entry field; the default (S1) field never draws them.
    assert f.site_entry.max() < S1_ENTRIES
    f12 = MaterialField(0.0, 0.0, 40.0, 40.0, 0.1, "cpu", seed=1, entries=len(LIBRARY))
    assert f12.site_entry.max() == len(LIBRARY) - 1 and Palette(4, "cpu", len(LIBRARY)).v.shape[1] == len(LIBRARY)
    blended = (w > 0.05).float().mean().item()
    print(f"materials self-check OK: {len(f.sites)} sites, blended-cell share {blended:.3f}")
