# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod flat-ground locomotion environment — PhantomX Mark IV.

Reward design (v2 — locomotion)
--------------------------------
Goal        : Walk forward as fast as possible in the robot's head-forward
              direction without falling over.

Reward terms
~~~~~~~~~~~~
+ forward_velocity      : velocity-scaled reward for positive x-body-frame speed
- stand_still           : penalty if v_x < 0.1 m/s (cancels alive bonus)
- lateral_velocity      : small penalty for sideways drift (y-body-frame)
- base_height           : penalty if root drops below 0.13 m (weight -60)
- ang_vel_xy            : penalise roll/pitch rates — stability proxy
- base_orientation      : two-zone pitch/roll angle penalty (projected gravity)
+ foot_clearance        : reward each foot for being off the ground (anti-slide)
+ joint_motion          : reward joint speed minus acceleration penalty
- joint_torque_excess   : penalise torques above threshold (2.5 Nm) — efficiency
- action_rate           : penalise rapid action changes — smooth control
- non_foot_body         : penalise femur links approaching ground (bent-knee fix)
+ alive                 : constant bonus each step
- terminating           : penalty on episode end by fall

Reset
~~~~~
Every episode:
  - Root position  : random x/y offset ±0.5 m, random yaw ∈ [-π, π]
  - Joint angles   : ±0.25 rad offset from rest pose (wider than before)
"""

import math

# Isaac Lab built-in MDP primitives
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

# Local custom MDP functions (rewards.py in this package)
from . import mdp as hex_mdp  # isort:skip

# PhantomX Mark IV asset configuration
from hexasim.assets.robot import PXMARK4_AX12A_CFG, PXMARK4_CFG  # isort:skip

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Scene definition
##


@configclass
class HexapodSceneCfg(InteractiveSceneCfg):
    """Scene with flat ground, dome/distant lighting, hexapod robot, and foot contact sensors."""

    # -----------------------------------------------------------------------
    # Ground plane
    # -----------------------------------------------------------------------
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(200.0, 200.0),
            # Multiply outranks PhysX's default average, so effective μ = μ_ground × μ_foot(1.0) = μ_ground.
            # μ_ground 0.75 = π_base's training friction: the robot USD has no collider material, so its feet
            # used PhysX's default 0.5, averaged with the old ground's 1.0. At μ 1.0, π_base falls in 9 % more
            # episodes (logs/eval/pi_base_ablate_*.json). ponytail: flat-ground placeholder until the per-surface
            # μ bench measurement; the terrain material library replaces it at S0b.
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=0.75,
                dynamic_friction=0.75,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # Lighting
    # -----------------------------------------------------------------------
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.98, 0.9), intensity=3000.0),
    )

    # -----------------------------------------------------------------------
    # Robot  (contact sensors enabled in robot.py via activate_contact_sensors=True)
    # Body contact forces are available via asset.data.body_contact_forces
    # without needing a separate ContactSensorCfg in the scene.
    # -----------------------------------------------------------------------
    robot: ArticulationCfg = PXMARK4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -----------------------------------------------------------------------
    # Foot Contact Sensors
    # -----------------------------------------------------------------------
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*tn__pxmark4.*_tibia_link.*",
        history_length=3,
        track_air_time=True,
    )


##
# MDP: Actions
##


@configclass
class ActionsCfg:
    """Joint position residual targets for all 18 joints.

    Network output is treated as a residual (in radians) added to the
    default joint positions from the init_state.  Scale of 0.5 rad limits
    each joint's commanded deviation to ±0.5 rad from rest.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


##
# MDP: Observations
##


@configclass
class ObservationsCfg:
    """Observation specifications for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """75-dimensional observation vector.

        [3]  base linear velocity  (body frame)
        [3]  base angular velocity (body frame)
        [3]  projected gravity
        [18] joint positions (relative to default)
        [18] joint velocities
        [18] previous actions
        [12] cpg phase clock (sin/cos for 6 legs)
        Total: 75
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        cpg_clock = ObsTerm(func=hex_mdp.cpg_clock)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# MDP: Events (resets / randomisation)
##


@configclass
class EventCfg:
    """Reset and randomisation events."""

    # Reset all 18 joints with ±0.25 rad offset from rest pose.
    # Wider range than before (was ±0.1 rad) to improve reset diversity
    # and stop the agent memorising a single start configuration.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.25, 0.25),
            "velocity_range": (-0.05, 0.05),
        },
    )

    # Foot (all robot bodies) μ = 1.0 so the multiply combine returns the ground's μ unchanged.
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Camera mast (D435 + bracket) on every arm, blind ones included (paradigm spec §1).
    # ponytail: provisional 0.15 kg at (+0.10, 0, +0.10) m from the base origin, until weighed on the robot;
    # modelled as added base mass + the resulting base CoM shift 0.15·(0.10, 0, 0.10)/1.672 kg.
    camera_mast_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*"),
            "mass_distribution_params": (0.15, 0.15),
            "operation": "add",
        },
    )
    camera_mast_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*"),
            "com_range": {"x": (0.009, 0.009), "z": (0.009, 0.009)},
        },
    )

    # Reset root pose: random x/y offset, random yaw (full 360°).
    # Yaw randomisation is critical — without it the agent never learns
    # to walk relative to its own heading; it just aligns with world-x.
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),  # full random heading every episode
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# MDP: Rewards
##


@configclass
class RewardsCfg:
    """Locomotion reward shaping for the PhantomX Mark IV.

    Term weights — design rationale
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    forward_velocity  +2.0   Primary signal. 1 m/s forward → +2 per step.
                             Dominates over height and contact penalties
                             so the agent prioritises movement.

    lateral_vel       -0.3   Gentle penalty so lateral drift is discouraged
                             but not eliminated during early training.
                             Scaled to ~-0.15 at 0.5 m/s lateral speed.

    stand_still      -10.0   Cancels the alive bonus when v_x < 0.1 m/s
                             so standing/sliding with no leg effort is neutral.

    base_height      -60.0   Strong height maintenance. Kicks in if root drops
                             below 0.13 m (raised from 0.095 m — forces legs
                             to extend fully, directly counteracting bent-knee
                             local optima).
                             0.1 m height error → -6.0 / step.

    ang_vel_xy        -0.5   Stability proxy: penalise roll and pitch rates.
                             At 1 rad/s roll → -0.5 / step.

    base_orientation  -5.0   Two-zone pitch/roll angle penalty.
                             Uses projected_gravity_b[:2] = sin(tilt_angle).
                             Soft zone [0°, 8°]: gentle quadratic → -0.04/step at 5°.
                             Hard zone [8°, ∞]: 10× steep ramp → -2.74/step at 15°,
                             which exceeds the +2.0 forward velocity reward and
                             forces the robot to level out.

    foot_clearance    +0.8   Reward each of the 6 feet for being above the
                             ground plane (max 1.0 per foot, 6.0 total).
                             Directly counters sliding: all-feet-down = 0.
                             Saturates at 2 cm clearance per foot.

    joint_motion      +0.05  Reward total joint speed minus 10 % acceleration
                             penalty.  Forces leg pumping without vibration.
                             At π rad/s per joint × 18 joints → ~2.8 / step.

    torque_excess     -0.05  Penalise torques above 2.5 Nm (per-joint).
                             Below threshold: free. Above: -0.05 × excess Nm.

    action_rate       -0.01  Smooth control signal.

    non_foot_body    -30.0   Penalise femur links if their Z-height drops below
                             0.05 m (ground proximity). Directly targets the
                             bent-knee pathology where the femur drags.
                             At ground level: 0.05 m × 30 × 6 = -9.0/step max.
                             Per single femur: 0.05 × 30 = -1.5/step min.

    alive             +1.0   Constant per-step bonus.
    terminating       -5.0   Penalty on episode termination (not timeout).
    """

    # (1) Primary locomotion incentive — velocity-scaled forward reward
    forward_velocity = RewTerm(
        func=hex_mdp.forward_velocity,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (2) Lateral drift penalty (body-frame y-axis)
    lateral_velocity = RewTerm(
        func=hex_mdp.lateral_velocity_penalty,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (3) Stand still penalty
    #     Penalizes the agent if forward velocity < 0.1 m/s.
    #     Counteracts the +1.0 alive bonus if the agent just stands still.
    #     Weight -5.0 means if velocity is 0, penalty is -0.5,
    #     but we can set weight to -10.0 so at v=0 penalty is -1.0 (cancels alive).
    stand_still = RewTerm(
        func=hex_mdp.stand_still_penalty,
        weight=-10.0,
        params={
            "threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (4) Height penalty — only penalise if height drops below 0.13 m.
    #     Raised from 0.095 m: forces legs to extend fully at all times,
    #     directly counteracting bent-knee local optima where the robot
    base_height = RewTerm(
        func=hex_mdp.base_height_penalty,
        weight=-60.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.13,
        },
    )

    # (4) Stability — penalise roll/pitch rates
    ang_vel_xy = RewTerm(
        func=hex_mdp.ang_vel_xy_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (4b) Orientation penalty — two-zone roll/pitch angle (NOT rate)
    #      ang_vel_xy only fires while rotating; this fires on sustained lean.
    #      Soft zone [0°, 8°]: barely felt (-0.04/step at 5°).
    #      Hard zone [8°, ∞]: 10× ramp — at 15° lean costs -2.74/step,
    #      which overpowers the +2.0 forward velocity bonus.
    base_orientation = RewTerm(
        func=hex_mdp.base_orientation_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "soft_threshold_deg": 8.0,
            "hard_multiplier": 10.0,
        },
    )

    # (4c) Foot clearance reward — Option A (anti-sliding)
    #      Rewards each of the 6 feet for lifting off the ground.
    #      Sliding → all feet on ground → 0 reward.
    #      Proper swing phase → feet airborne → +up to 6.0 per step.
    #      body_names regex selects the six *_foot_link bodies without
    #      needing a ContactSensor (avoids the USD segfault).
    foot_clearance = RewTerm(
        func=hex_mdp.foot_clearance_reward,
        weight=0.4,
        params={
            # The URDF foot_link is merged into the tibia at import; the term derives
            # the toe point from the tibia pose (mdp.rewards.toe_pos_w).
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*tibia.*"]),
            "target_clearance": 0.02,  # 2 cm lift saturates the reward
        },
    )

    # (4c) Joint motion reward with acceleration penalty
    #      Rewards total joint speed (encourages leg use) minus a scaled
    #      acceleration penalty (discourages vibration/jitter).
    #      Net positive only when joints move smoothly and consistently.
    joint_motion = RewTerm(
        func=hex_mdp.joint_motion_reward,
        weight=0.035,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "accel_penalty_scale": 0.1,
        },
    )

    # (5) Torque excess penalty — soft effort budget of 2.5 Nm per joint
    #     (URDF hard limit is 3.0 Nm; this penalises at 83% of limit)
    joint_torque_excess = RewTerm(
        func=hex_mdp.joint_torque_threshold_penalty,
        weight=-0.05,
        params={
            "threshold": 2.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (6) Smooth control signal
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # (7) Non-foot body ground proximity penalty — bent-knee fix.
    #     Penalises femur links when their Z-height drops below 0.05 m.
    #     Threshold is tight (= termination floor) so it only fires on
    #     severe drooping, not during normal leg swing.
    #     A single femur at z=0: penalty = 0.05 m × weight -30 = -1.5/step
    #     — enough to overcome foot_clearance + joint_motion gain from the
    #     broken-leg posture.
    non_foot_body = RewTerm(
        func=hex_mdp.non_foot_body_ground_penalty,
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*femur.*"]),
            "minimum_height": 0.05,
        },
    )

    # (8) Alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    # CPG Tracking
    cpg_z_target = RewTerm(
        func=hex_mdp.cpg_z_target_l1_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*tibia.*"]),
            "step_height": 0.038,
        },
    )
    cpg_stance_rigidity = RewTerm(
        func=hex_mdp.cpg_stance_rigidity_penalty,
        weight=0.0,
        params={
            # Coxa deliberately excluded: the backward coxa sweep IS the stance stroke that cpg_coxa_contact
            # pays for (+1 per rad/s); penalising it here (−0.5 per rad/s) would recreate the v7 §2.6 coxa
            # contradiction. Stance slip is policed by stance_rigidity_contact (toe velocity) from S0b.
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_femur", ".*_tibia"]),
        },
    )
    cpg_force_target = RewTerm(
        func=hex_mdp.cpg_force_target_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*tibia.*"]),
        },
    )
    cpg_coxa_contact = RewTerm(
        func=hex_mdp.cpg_coxa_contact_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_coxa"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*tibia.*"]),
        },
    )


##
# MDP: Terminations
##


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Robot has fallen — root height goes below 0.05 m
    base_contact = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.05},
    )


@configclass
class CurriculaCfg:
    """Curriculum terms for the MDP."""

    foot_clearance = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "foot_clearance",
            "initial_weight": 0.8,
            "target_weight": 0.0,
            "start_step": 1500,
            "end_step": 2000,
        },
    )

    joint_motion = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "joint_motion",
            "initial_weight": 0.05,
            "target_weight": 0.0,
            "start_step": 1500,
            "end_step": 2000,
        },
    )

    cpg_z = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "cpg_z_target",
            "initial_weight": 0.0,
            "target_weight": -10.0,
            "start_step": 1500,
            "end_step": 45000,
        },
    )
    cpg_rigidity = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "cpg_stance_rigidity",
            "initial_weight": 0.0,
            "target_weight": -0.5,
            "start_step": 1500,
            "end_step": 8000,
        },
    )
    cpg_force = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "cpg_force_target",
            "initial_weight": 0.0,
            "target_weight": -0.5,
            "start_step": 1500,
            "end_step": 8000,
        },
    )

    cpg_coxa = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "cpg_coxa_contact",
            "initial_weight": 0.0,
            "target_weight": 1.0,
            "start_step": 1500,
            "end_step": 8000,
        },
    )
    # Ease off the primary velocity incentive once CPG phase-tracking takes
    # over (step 3000): 2.0 -> 1.0 so gait-shaping objectives dominate the
    # late-training reward economy without removing the drive to move.
    forward_velocity = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "forward_velocity",
            "initial_weight": 2.0,
            "target_weight": 1.7,
            "start_step": 1500,
            "end_step": 45000,
        },
    )

    base_height = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "base_height",
            "initial_weight": -60.0,
            "target_weight": 0.0,
            "start_step": 1500,
            "end_step": 2000,
        },
    )

    non_foot_body = CurrTerm(
        func=hex_mdp.LinearAnnealRewardWeight,
        params={
            "term_name": "non_foot_body",
            "initial_weight": -30.0,
            "target_weight": 0.0,
            "start_step": 1500,
            "end_step": 2000,
        },
    )


##
# Environment configuration
##


@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """PhantomX Mark IV hexapod — flat ground locomotion environment."""

    # Single source for every CPG term's swing/stance split (tripod offsets {0, π} require 0.5).
    duty_factor: float = 0.5

    # Scene
    scene: HexapodSceneCfg = HexapodSceneCfg(
        num_envs=4096,
        env_spacing=10.0,  # 10 m between robots — room for 10 s episodes
    )

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculaCfg = CurriculaCfg()

    def __post_init__(self) -> None:
        """Post initialisation."""
        self.decimation = 4  # 200 Hz physics / 4 = 50 Hz policy
        self.episode_length_s = 10.0  # 10 s × 50 Hz = 500 steps
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class HexapodEnvCfg_PLAY(HexapodEnvCfg):
    """Single-env evaluation variant — no observation noise."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5


@configclass
class HexapodFlatResumeEnvCfg(HexapodEnvCfg):
    """Resume variant — no curriculum, max target weights from start."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Disable all curricula to prevent weight resets on --resume
        self.curriculum = None

        # Manually apply the final target weights from the original curriculum
        self.rewards.foot_clearance.weight = 0.0
        self.rewards.joint_motion.weight = 0.0
        self.rewards.cpg_z_target.weight = -10.0
        self.rewards.cpg_stance_rigidity.weight = -0.5
        self.rewards.cpg_force_target.weight = -0.5
        self.rewards.cpg_coxa_contact.weight = 1.0
        self.rewards.forward_velocity.weight = 1.7
        self.rewards.base_height.weight = 0.0


##
# S0a — actuator and observation reconditioning (paradigm spec §4.2)
##


@configclass
class HexapodS0aSceneCfg(HexapodSceneCfg):
    """AX-12A servos, rest pose femur +30° / tibia +30°, and a base IMU."""

    robot: ArticulationCfg = PXMARK4_AX12A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/tn__pxmark4base_link_VP")


@configclass
class S0aActionsCfg:
    """Joint targets = rest pose + 0.5·a, clipped to the servo CW/CCW limits (= URDF joint limits, rad)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        clip={".*_coxa": (-0.9599, 0.9599), ".*_femur": (-1.7104, 1.5708), ".*_tibia": (-0.7854, 2.1817)},
    )


@configclass
class S0aObservationsCfg:
    """Actor: o_p ⊕ v_in (114-D). Critic: the same ⊕ privileged 22-D. Noise is applied inside the env."""

    @configclass
    class PolicyCfg(ObsGroup):
        o_p_v_in = ObsTerm(func=hex_mdp.s0a_policy_obs)

    @configclass
    class CriticCfg(ObsGroup):
        o_p_v_in = ObsTerm(func=hex_mdp.s0a_policy_obs)
        privileged = ObsTerm(func=hex_mdp.s0a_privileged_obs)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HexapodS0aEnvCfg(HexapodFlatResumeEnvCfg):
    """S0a: π_base's final reward weights (curriculum off) under the AX-12A model and deployable observations."""

    scene: HexapodS0aSceneCfg = HexapodS0aSceneCfg(num_envs=4096, env_spacing=10.0)
    actions: S0aActionsCfg = S0aActionsCfg()
    observations: S0aObservationsCfg = S0aObservationsCfg()

    # Read pipeline (provisional, calibrate on the bench): p_drop = drop_p0 + drop_k·|τ|/τ_stall.
    drop_p0: float = 0.005
    drop_k: float = 0.068
    imu_acc_std: float = 0.05  # m/s²
    imu_gyro_std: float = 0.02  # rad/s
    gravity_std: float = 0.02  # unit vector
    # V_0 substitution ramp, env steps (16 per iteration): α 0 → 1 from the end of the 300-iteration critic
    # burn-in to 50 % of a 3,000-iteration run.
    v0_lr: float = 1e-3
    v0_ramp_steps: tuple[int, int] = (300 * 16, 1500 * 16)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Camera mast 0.15 kg + payload U[0.1, 1.0] kg, per env (static envelope at +1.0 kg: femur 14 %,
        # tibia 42 % of stall).
        self.events.camera_mast_mass.params["mass_distribution_params"] = (0.25, 1.15)
        # applied_torque is now the servo model's own torque; penalise the top 20 % of the 1.5 N·m stall.
        self.rewards.joint_torque_excess.params["threshold"] = 1.2
        # Explicit (effort) actuators need PhysX TGS to apply external forces in every solver iteration; with the
        # default False the servo torques barely move the joints (phase2-progress §2.0.9).
        self.sim.physics = PhysxCfg(enable_external_forces_every_iteration=True)
        # Reward economics (phase2-progress §2.0.11): bounded positive toe-swing reward instead of the L1 penalty,
        # and an alive bonus sized so that standing > falling and walking > standing. Weights set from
        # scripts/tests/reward_economy.py measurements.
        self.rewards.cpg_z_target.func = hex_mdp.cpg_z_target_kernel_reward
        # σ 1 cm (run 4 on; 2 cm in run 3): swings on target keep ~90 % of their credit, stutter micro-lifts
        # earn ≈ nothing — pressure on the right-middle leg's inherited stutter (phase2-progress §2.0.13).
        self.rewards.cpg_z_target.params["sigma"] = 0.01
        self.rewards.cpg_z_target.weight = 5.0
        self.rewards.alive.weight = 10.0
        # stand_still at v = 0 costs 1.5 × alive, so standing (+0.029/step) stays below a falling-prone π_base gait
        # (+0.074/step) and above falling (−0.1 once): walking > standing > falling.
        self.rewards.stand_still.weight = -75.0


@configclass
class HexapodS0aEnvCfg_PLAY(HexapodS0aEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5


##
# S0b — commands, clock, standing, terrain, dynamics randomisation (paradigm spec §4.2)
##


@configclass
class HexapodS0bSceneCfg(HexapodS0aSceneCfg):
    """Generated terrain (μ 1, restitution 1, multiply: the per-env contact physics sits on the feet) and servo DR.

    ``replicate_physics=False``: each env's feet carry their own USD material (``mdp.events.foot_materials``).
    Servo ranges are provisional until the bench (v7 §5.6; latency stays 1–2 physics steps = 5–10 ms).
    """

    ground = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=hex_mdp.S0B_TERRAIN_CFG,
        max_init_terrain_level=2,
        physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )
    robot: ArticulationCfg = PXMARK4_AX12A_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "ax12a": PXMARK4_AX12A_CFG.actuators["ax12a"].replace(
                stall_scale_range=(0.9, 1.1),
                compliance_slope_choices=(16, 32, 64),
                backlash_range=(math.radians(0.3), math.radians(1.0)),
                coulomb_range=(0.0, 0.05),
            )
        },
    )
    replicate_physics: bool = False


@configclass
class S0bObservationsCfg:
    """Actor: o_p ⊕ v_in ⊕ twist_cmd (117-D). Critic: the same ⊕ privileged 22-D ⊕ DR parameters 12-D."""

    @configclass
    class PolicyCfg(ObsGroup):
        o_p_v_in_cmd = ObsTerm(func=hex_mdp.s0b_policy_obs)

    @configclass
    class CriticCfg(ObsGroup):
        o_p_v_in_cmd = ObsTerm(func=hex_mdp.s0b_policy_obs)
        privileged = ObsTerm(func=hex_mdp.s0a_privileged_obs)
        dr = ObsTerm(func=hex_mdp.s0b_dr_obs)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class S0bCommandsCfg:
    """twist_cmd (spec §1 + D19): resampled every 5 s; 10 % standing; of the rest 15 % turn on the spot
    (|yaw| 0.1–0.4 rad/s), 30 % walk with v_y ±0.05 m/s and yaw ±0.4 rad/s, 55 % walk straight."""

    base_velocity = hex_mdp.HexapodVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_side_envs=0.3,
        rel_turn_envs=0.15,
        turn_yaw_range=(0.1, 0.4),
        heading_command=False,
        debug_vis=False,
        ranges=hex_mdp.HexapodVelocityCommandCfg.Ranges(
            lin_vel_x=(0.05, 0.20), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.4, 0.4)
        ),
    )


@configclass
class S0bRewardsCfg(RewardsCfg):
    """S0a's terms plus command tracking, the phase-free floors (spec §5) and the standing-mode term.

    Economics (phase2-progress §2.1): tracking is the primary signal; the floors are small next to it.
    """

    # Per-step velocity tracking (τ = dt), exp σ 0.05 and L1, ×30: the policy was nearly command-insensitive at ×10
    # (0.05 / 0.10 / 0.15 m/s commanded → 0.110 / 0.128 / 0.146 walked); tracking must dominate the gait-shaping
    # terms (phase2-progress §2.1.9).
    track_lin_vel = RewTerm(
        func=hex_mdp.track_lin_vel_xy,
        weight=30.0,
        params={"command_name": "base_velocity", "std": 0.05, "l1_scale": 1.0},
    )
    # Yaw-rate tracking on the same footing as linear (phase2-progress §2.1.21): exp σ 0.1 rad/s + L1, ×10.
    track_yaw = RewTerm(
        func=hex_mdp.track_ang_vel_z,
        weight=10.0,
        params={"command_name": "base_velocity", "std": 0.1, "l1_scale": 1.0},
    )
    swing_clearance = RewTerm(
        func=hex_mdp.swing_clearance,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "height": 0.03},
    )
    stance_rigidity_contact = RewTerm(
        func=hex_mdp.stance_rigidity_contact_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "joint_scale": 0.1,
        },
    )
    # Standing mode (phase2-progress §2.1.5): ×5 so a lifted leg costs 0.83/step; the joint-speed term covers what
    # the contact-conditioned stance floor cannot see (a leg held in the air).
    standing_contact = RewTerm(
        func=hex_mdp.standing_contact_reward, weight=5.0, params={"sensor_cfg": SceneEntityCfg("contact_forces")}
    )
    standing_joint_speed = RewTerm(func=hex_mdp.standing_joint_speed_penalty, weight=-0.2)
    # Standing mode: distance of every joint from the rest pose (§2.1.14).
    standing_pose = RewTerm(func=hex_mdp.standing_pose_penalty, weight=-3.0)
    # Joint targets requested past the servo limits (§2.1.12): no gradient returns them otherwise.
    action_beyond_clip = RewTerm(
        func=hex_mdp.action_beyond_clip_penalty, weight=-10.0, params={"action_name": "joint_pos"}
    )
    # Even weight on all six feet (phase2-progress §2.1.7): the contact test above only says "touching" at 1 N.
    standing_load_share = RewTerm(
        func=hex_mdp.standing_load_share_penalty,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )


@configclass
class S0bTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=hex_mdp.root_height_above_terrain_below, params={"minimum_height": 0.05})


@configclass
class S0bCurriculumCfg:
    terrain_levels = CurrTerm(func=hex_mdp.terrain_levels_cmd)


@configclass
class HexapodS0bEnvCfg(HexapodS0aEnvCfg):
    """S0b: S0a + twist_cmd, command clock, standing mode, terrain curriculum, full dynamics randomisation."""

    scene: HexapodS0bSceneCfg = HexapodS0bSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: S0bObservationsCfg = S0bObservationsCfg()
    commands: S0bCommandsCfg = S0bCommandsCfg()
    rewards: S0bRewardsCfg = S0bRewardsCfg()
    terminations: S0bTerminationsCfg = S0bTerminationsCfg()

    # Per-episode IMU bias, provisional (bench).
    imu_acc_bias_std: float = 0.05  # m/s²
    imu_gyro_bias_std: float = 0.01  # rad/s
    # Standing settle window (phase2-progress §2.1.18): with six feet planted the pose can only change by stepping,
    # so the hold terms wait one gait period (1.25 s at 0.8 Hz, rounded up) while the rest-pose pull brings it home.
    stand_settle_s: float = 1.5
    # V_0 was ramped in during S0a; S0b runs on it alone.
    v0_ramp_steps: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.curriculum = S0bCurriculumCfg()  # the S0a parent clears it
        r = self.rewards
        # Replaced by command tracking: forward speed, lateral drift and the 0.1 m/s stand-still threshold (which
        # would punish the 0.05 m/s commands). Flat-only / zero-weight Phase 1 terms go too.
        r.forward_velocity = r.lateral_velocity = r.stand_still = None
        r.base_height = r.foot_clearance = r.joint_motion = None
        ev = self.events
        ev.robot_material = None  # contact physics is per-env on the feet (prestartup materials)
        ev.foot_material = EventTerm(
            func=hex_mdp.foot_materials,
            mode="prestartup",
            params={
                "num_buckets": 64,
                "static_friction_range": (0.3, 1.1),  # provisional: must contain every bench-measured μ
                "dynamic_ratio_range": (0.8, 1.0),
                "restitution_range": (0.0, 0.1),
                "rigid_fraction": 0.5,
                "stiffness_range": (2e3, 5e4),  # N/m, provisional until the foam is characterised
                "damping_range": (10.0, 200.0),  # N·s/m
            },
        )
        ev.camera_mast_com.params["com_range"] = {"x": (-0.001, 0.019), "y": (-0.01, 0.01), "z": (0.009, 0.009)}
        ev.reset_robot_base.params["pose_range"]["z"] = (0.0, 0.02)  # clears 2 cm bumps at the spawn point


@configclass
class HexapodS0bEnvCfg_PLAY(HexapodS0bEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
