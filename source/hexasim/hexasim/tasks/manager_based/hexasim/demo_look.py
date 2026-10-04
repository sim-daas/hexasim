# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Showcase rendering for the demo scripts — visual only, never physics (phase2-progress §2.4).

The training envs are deliberately minimal to render: two bare lights, no sky, terrain painted in two flat greys
(``mdp.materials.build_terrain_materials``), and Isaac Lab's ``balanced`` renderer with reflections, ambient
occlusion and global illumination switched off by ``apps/isaaclab.python.headless.rendering.kit``. That is right for
training and unusable for a video a reviewer is meant to watch.

``apply(cfg)`` upgrades an **already-parsed** env cfg in place, before ``gym.make``:

* RTX settings (``cfg.sim.render``), recording resolution (``cfg.video_recorder``),
* an HDRI sky dome + a raking sun + a rim light in place of the two flat lights,
* one MDL per material-library entry instead of the two flat greys (``materials.VISUAL_OVERRIDE``).

Optionally (``robot_mdl=``) it also re-skins the robot. Default is off: the PhantomX really is matte black, and
against a lit terrain under an HDRI sky it reads as a silhouette, which is honest and legible.

Nothing here touches the MDP, the collider, the actuator model, dynamics randomisation, sensor noise or dropped
reads: the showcase runs the same trained condition as an eval. The terrain materials it replaces are the per-entry
*visual* submeshes, which ``build_terrain_materials`` already creates with collision disabled — the single collider
is the (invisible) source mesh. Everything is opt-in: with ``apply`` never called, behaviour is bit-identical.

Assets stream from the Isaac asset root on first use (the local Omniverse hub cache starts empty), so the first
showcase run is slow to start and later ones are not.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import RenderCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from .mdp import materials

# --------------------------------------------------------------------------------------------------------------
# Skies. Short key → path under {NVIDIA_NUCLEUS_DIR}/Assets/Skies (every one of these was checked to exist).
# --------------------------------------------------------------------------------------------------------------
SKIES: dict[str, str] = {
    "noon_grass": "Clear/noon_grass_4k.hdr",  # default: bright, neutral, best terrain/background separation
    "qwantani": "Clear/qwantani_4k.hdr",
    "sunflowers": "Clear/sunflowers_4k.hdr",
    "partly_cloudy": "Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",  # softest, most forgiving
    "lakeside": "Cloudy/lakeside_4k.hdr",
    "table_mountain": "Cloudy/table_mountain_1_4k.hdr",
    "evening": "Evening/evening_road_01_4k.hdr",  # dramatic, long shadows
    "studio": "Studio/photo_studio_01_4k.hdr",
    "night": "Night/kloppenheim_02_4k.hdr",
    "warehouse": "Indoor/ZetoCG_com_WarehouseInterior2b.hdr",  # for the future indoor backdrop
}

# --------------------------------------------------------------------------------------------------------------
# One MDL per material-library entry (``materials.LIBRARY``), so the friction/compliance patchwork the policy
# walks over is visible instead of being two flat greys. Paths are relative to
# {NVIDIA_NUCLEUS_DIR}/Materials/Base and were each checked to exist. texture_scale is in metres of UV per tile:
# the robot's stride is 12–19 cm, so ~0.5 m tiles read at the scale of the gait.
# --------------------------------------------------------------------------------------------------------------
TERRAIN_MDL: dict[str, tuple[str, float, float]] = {
    # entry name            MDL under Materials/Base                 texture_scale  albedo_brightness
    "concrete": ("Masonry/Concrete_Rough.mdl", 0.5, 1.0),
    "rubber_mat": ("Plastics/Rubber_Textured.mdl", 0.4, 1.0),
    "ptfe": ("Masonry/Concrete_Polished.mdl", 0.6, 1.3),
    "talc_graphite": ("Masonry/Concrete_Smooth.mdl", 0.5, 0.75),  # a slick pale powder, not the dark metal it was
    "dry_dust": ("Natural/Sand.mdl", 0.5, 1.1),
    "foam": ("Stone/Granite_Light.mdl", 0.6, 1.2),
    "wet_tile": ("Stone/Marble.mdl", 0.7, 1.0),
    "wet_rubber": ("Plastics/Rubber_Smooth.mdl", 0.4, 0.9),
    "glossy_grit": ("Stone/Gravel.mdl", 0.35, 1.0),
    "polished_coating": ("Stone/Terrazzo.mdl", 0.6, 1.0),
    "soft_soil": ("Natural/Dirt.mdl", 0.5, 1.0),
    "wet_clay": ("Stone/Retaining_Block.mdl", 0.5, 0.9),
}

# Optional robot re-skin. The robot USD carries one material, ``interbotix_black``, bound directly on each of its
# 12 visual meshes *inside instance prototypes* — so it cannot be edited on the composed stage, and an ancestor
# binding only wins because ``spawn_from_usd`` binds with ``strongerThanDescendants`` (from_files.py:405-416).
# That makes a spawn-time ``visual_material`` the one reliable way in, which is what ``robot_mdl`` uses.
# One material over the whole terrain. The 12-entry patchwork above shows the physics, but for a presentation it
# reads as a patchwork; ``terrain_mdl`` picks one of these instead (``"multi"`` keeps the patchwork).
GROUND_MDL: dict[str, tuple[str, float, float]] = {
    "dirt": ("Natural/Dirt.mdl", 0.5, 1.0),
    "sand": ("Natural/Sand.mdl", 0.5, 1.1),
    "gravel": ("Stone/Gravel.mdl", 0.35, 1.0),
    "grass": ("Natural/Grass_Countryside.mdl", 0.6, 1.0),
    "concrete": ("Masonry/Concrete_Rough.mdl", 0.5, 1.0),
    "granite": ("Stone/Granite_Light.mdl", 0.6, 1.1),  # default: pale speckled stone, the robot reads against it
    "terrazzo": ("Stone/Terrazzo.mdl", 0.6, 1.0),
    "plywood": ("Wood/Plywood.mdl", 0.5, 1.0),
}

ROBOT_MDL: dict[str, tuple[str, float]] = {
    "abs": ("Plastics/Plastic_ABS.mdl", 0.05),  # closest to the real 3D-printed/moulded black parts
    "rubber": ("Plastics/Rubber_Smooth.mdl", 0.05),
    "anodized": ("Metals/Aluminum_Anodized.mdl", 0.05),  # a "machined prototype" look, not what the robot is
}

# Plain surfaces, for when a textured material is not the point. The robot is really matte black, but black on a
# lit ground renders as a hole with no readable shape, so grey is the presentable choice.
# ``diffuse_color`` is linear, and under the showcase light rig it renders far lighter than the number looks: 0.45
# came out near white. These are set against the dark grey the robot has in the existing gait videos.
ROBOT_SURFACE: dict[str, tuple[tuple[float, float, float], float, float]] = {
    #                      diffuse (linear)      roughness metallic
    "gunmetal": ((0.11, 0.115, 0.125), 0.45, 0.15),  # default: the dark grey of the logs/videos/gait takes
    "dark_grey": ((0.06, 0.065, 0.07), 0.45, 0.10),
    "grey": ((0.19, 0.20, 0.21), 0.45, 0.08),
    "light_grey": ((0.34, 0.35, 0.36), 0.40, 0.05),
    "steel": ((0.26, 0.27, 0.29), 0.28, 0.70),
}


def _robot_material(key: str):
    """The spawn-time visual material for ``robot_mdl``: a plain surface preset, or one of the MDLs."""
    if key in ROBOT_SURFACE:
        diffuse, roughness, metallic = ROBOT_SURFACE[key]
        return sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse, roughness=roughness, metallic=metallic)
    rel, scale = ROBOT_MDL[key]
    return sim_utils.MdlFileCfg(
        mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/{rel}",
        project_uvw=True,  # the shells' own UVs are laid out for one flat tile, not for a material
        texture_scale=(scale, scale),
    )


def _quat_zy(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for Rz(yaw)·Ry(pitch). A light's emission direction is its local −Z, so pitch −90° is horizontal
    and 0° is straight down."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def _quat_zx(yaw_deg: float, roll_deg: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for Rz(yaw)·Rx(roll). Used for the sky dome: ``roll`` tilts the latlong image, ``yaw`` spins it."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return (cy * cr, cy * sr, sy * sr, sy * cr)


def _render_cfg(quality: str) -> RenderCfg:
    """RTX settings. ``high`` is the RT2 (path-traced) preset; ``mid`` is RT1 and fits alongside a training run."""
    common = dict(
        enable_reflections=True,
        enable_ambient_occlusion=True,
        enable_direct_lighting=True,
        enable_shadows=True,
        enable_dl_denoiser=True,
    )
    if quality == "high":
        return RenderCfg(
            rendering_mode="quality",
            dome_light_upper_lower_strategy=0,  # full IBL from the HDRI; 4 (approximated) renders noticeably flatter
            enable_global_illumination=True,
            max_bounces=3,  # ≥ 3 is required for GI
            samples_per_pixel=4,
            antialiasing_mode="DLAA",
            ambient_occlusion_denoiser_mode=0,  # 0 = higher quality
            **common,
        )
    if quality == "mid":
        return RenderCfg(
            rendering_mode="balanced",
            dome_light_upper_lower_strategy=4,  # approximated: cheaper, and RT1 cannot do much with full IBL anyway
            samples_per_pixel=2,
            antialiasing_mode="DLSS",
            dlss_mode=2,  # quality
            **common,
        )
    raise ValueError(f"quality must be 'high' or 'mid', not {quality!r}")


def apply(
    cfg,
    *,
    sky: str = "noon_grass",
    quality: str = "high",
    res: tuple[int, int] = (1920, 1080),
    hdri_rot: float = 0.0,
    sky_tilt: float = 270.0,
    sun: tuple[float, float] = (35.0, -55.0),
    light_gain: float = 1.0,
    robot_mdl: str | None = None,
    terrain_mdl: str = "granite",
    backdrop: float = 400.0,
    backdrop_z: float = -0.05,
) -> None:
    """Make an already-parsed env cfg presentable. Call before ``gym.make``; visual only.

    Args:
        cfg: a parsed ``ManagerBasedRLEnvCfg`` (any hexapod task).
        sky: a key of :data:`SKIES`.
        quality: ``"high"`` (RT2 path tracing) or ``"mid"`` (RT1, for running next to a training job).
        res: recorded video resolution, (width, height).
        hdri_rot: sky rotation about +Z, degrees — turns the sun/horizon without changing the light rig.
        sky_tilt: sky rotation about +X, degrees. A ``UsdLuxDomeLight`` wraps its latlong texture with the poles on
            the **Y** axis while this stage is Z-up, so the sky needs a quarter turn to stand upright. 270 is the
            one that renders the right way up here; 90 puts it upside down and 180 brings the ground hemisphere
            above the horizon (both seen in test renders). Only change it if a different HDRI lands wrong.
        sun: (yaw, pitch) of the sun in degrees. Pitch −90° is horizontal, 0° straight down; a raking angle is what
            gives the legs long shadows and the terrain visible relief.
        light_gain: scales all three light intensities together. The defaults are set for the clear-sky HDRIs; a
            darker one (``evening``, ``night``) wants more.
        robot_mdl: a key of :data:`ROBOT_SURFACE` (plain grey presets) or :data:`ROBOT_MDL` (textured), or None to
            keep the robot's own flat-black material, which renders as a silhouette with no readable shape.
        terrain_mdl: a key of :data:`GROUND_MDL` to put one material over the whole terrain, or ``"multi"`` for the
            per-entry patchwork of :data:`TERRAIN_MDL` (which shows the physics, but looks like a patchwork).
        backdrop: side length, in metres, of a visual-only ground slab under the terrain (0 disables it). The
            generated terrain is a small island, so a low camera looks straight past its edge into the sky dome's
            **lower** hemisphere — the HDRI's own ground, which reads as an upside-down landscape hanging above the
            horizon. The slab extends the ground to the horizon and hides it. It carries no collider
            (``collision_props`` is left None, shapes.py:296), so it cannot touch the robot.
        backdrop_z: height of the slab's top face. It must sit **below the terrain's lowest point**, or the plane
            cuts through the height field and its flat intersection contours show up as smooth untextured blobs
            among the terrain texture. −0.05 clears flat, rough and bumps; stairs and slopes descend much further
            (a 15° slope drops ≈ 0.5 m across a 4 m tile), so pass about −0.7 for those.
    """
    if sky not in SKIES:
        raise ValueError(f"unknown sky {sky!r}; options: {sorted(SKIES)}")
    if robot_mdl is not None and robot_mdl not in ROBOT_MDL and robot_mdl not in ROBOT_SURFACE:
        raise ValueError(f"unknown robot_mdl {robot_mdl!r}; options: {sorted({**ROBOT_MDL, **ROBOT_SURFACE})}")
    if terrain_mdl != "multi" and terrain_mdl not in GROUND_MDL:
        raise ValueError(f"unknown terrain_mdl {terrain_mdl!r}; options: 'multi', {sorted(GROUND_MDL)}")

    cfg.sim.render = _render_cfg(quality)
    cfg.video_recorder.window_width, cfg.video_recorder.window_height = res

    # Sky dome: the HDRI is both the background and the ambient light, which is what separates terrain from sky.
    cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),  # no tint: the HDRI carries the colour
            intensity=1500.0 * light_gain,
            texture_file=f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/{SKIES[sky]}",
            texture_format="latlong",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_zx(hdri_rot, sky_tilt)),
    )
    # Sun: raking, soft-edged (``angle`` widens the source, so contact shadows are not razor-sharp).
    cfg.scene.distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.96, 0.88),
            intensity=4000.0 * light_gain,
            angle=1.2,
            enable_color_temperature=True,
            color_temperature=5600.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_zy(*sun)),
    )
    # Rim light from low behind-left: the robot is matte black, and without this it merges into dark ground.
    cfg.scene.rim_light = AssetBaseCfg(
        prim_path="/World/RimLight",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.88, 1.0), intensity=900.0 * light_gain, angle=3.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_zy(sun[0] + 160.0, -72.0)),
    )

    # The generator's random per-tile vertex colours fight the MDL materials (they are visible on the tasks with no
    # material field, where the source mesh is not hidden).
    terrain = getattr(cfg.scene, "terrain", None)
    if terrain is not None and getattr(terrain, "terrain_generator", None) is not None:
        terrain.terrain_generator = terrain.terrain_generator.replace(color_scheme="none")
    # Flat tasks keep a ground plane instead, and ``GroundPlaneCfg`` defaults to a black grid.
    ground = getattr(cfg.scene, "ground", None)
    if ground is not None:
        ground.spawn.color = (0.30, 0.29, 0.27)

    # One MDL per library entry. ``project_uvw`` is mandatory: the visual submeshes are built from raw triangles by
    # ``build_terrain_materials`` and carry no UVs, so a textured material without projection renders untextured.
    if terrain_mdl == "multi":
        table = TERRAIN_MDL
    else:
        table = dict.fromkeys(TERRAIN_MDL, GROUND_MDL[terrain_mdl])  # the same material under every entry
    materials.VISUAL_OVERRIDE = {
        name: sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/{rel}",
            project_uvw=True,
            texture_scale=(scale, scale),
            albedo_brightness=brightness,
        )
        for name, (rel, scale, brightness) in table.items()
    }
    missing = {e.name for e in materials.LIBRARY} - set(materials.VISUAL_OVERRIDE)
    assert not missing, f"no MDL for library entries {sorted(missing)}"

    if backdrop > 0.0:
        rel, scale, brightness = GROUND_MDL[terrain_mdl] if terrain_mdl != "multi" else GROUND_MDL["dirt"]
        cfg.scene.demo_backdrop = AssetBaseCfg(
            prim_path="/World/DemoBackdrop",
            spawn=sim_utils.CuboidCfg(
                size=(backdrop, backdrop, 0.4),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/{rel}",
                    project_uvw=True,
                    texture_scale=(scale, scale),
                    albedo_brightness=brightness,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, backdrop_z - 0.2)),  # 0.4 thick → top at backdrop_z
        )

    if robot_mdl is not None:
        cfg.scene.robot.spawn.visual_material = _robot_material(robot_mdl)
