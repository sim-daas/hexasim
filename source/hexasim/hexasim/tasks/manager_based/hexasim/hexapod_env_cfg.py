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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

# Isaac Lab built-in MDP primitives
import isaaclab.envs.mdp as mdp

# Local custom MDP functions (rewards.py in this package)
from . import mdp as hex_mdp  # isort:skip

# PhantomX Mark IV asset configuration
from hexasim.assets.robot import PXMARK4_CFG  # isort:skip

from isaaclab.sensors import ContactSensorCfg

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
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
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
                "yaw": (-math.pi, math.pi),   # full random heading every episode
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
            # NOTE: The URDF *_foot_link bodies are massless/geometry-less and
            # are silently dropped by the Isaac Sim USD importer.  The tibia
            # links are the true foot-tip equivalents available at runtime.
            # Their mangled USD names follow the pattern:
            #   tn__pxmark4{side}_{pos}_tibia_link_{hash}
            # The regex ".*tibia.*" matches all six reliably.
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
            "duty_factor": 0.5,
            "step_height": 0.038,
        },
    )
    cpg_stance_rigidity = RewTerm(
        func=hex_mdp.cpg_stance_rigidity_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_femur", ".*_tibia"]),
            "duty_factor": 0.5,
        },
    )
    cpg_force_target = RewTerm(
        func=hex_mdp.cpg_force_target_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*tibia.*"]),
            "duty_factor": 0.5,
            # 11.8 N robot / 3 tripod support legs ≈ 3.93 N per foot
            "max_force": 4.0,
        },
    )
    cpg_coxa_contact = RewTerm(
        func=hex_mdp.cpg_coxa_contact_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_coxa"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*tibia.*"]),
            "duty_factor": 0.5,
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

    # Scene
    scene: HexapodSceneCfg = HexapodSceneCfg(
        num_envs=4096,
        env_spacing=10.0,   # 10 m between robots — room for 10 s episodes
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
        self.decimation = 4                    # 200 Hz physics / 4 = 50 Hz policy
        self.episode_length_s = 10.0           # 10 s × 50 Hz = 500 steps
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
