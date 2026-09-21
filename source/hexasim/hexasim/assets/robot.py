# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the PhantomX Mark IV hexapod robot by Interbotix.

USDA generated from URDF via Isaac Sim URDF Importer extension.
Robot: PhantomX Mark IV (pxmark4) — 6 legs × 3 DOF = 18 revolute joints.

Joint layout (per leg):
  - coxa  : rotation about Z-axis, limits [-55°, +55°]
  - femur : rotation about Y-axis, limits [-98°, +90°]
  - tibia : rotation about Y-axis, limits [-45°, +125°]

Reference:
    https://www.trossenrobotics.com/phantomx-ax-hexapod.aspx
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from hexasim.actuators import AX12AActuatorCfg

##
# Robot USDA path
##

# Absolute path to the hexapod.usda produced by Isaac Sim URDF Importer.
# The file uses the "physx" physics variant which includes PhysicsDriveAPI joints.
_HEXAPOD_USD_PATH = (
    "/home/ubuntu/gitrepos/interbotix_ros_crawlers/"
    "interbotix_ros_xshexapods/"
    "interbotix_xshexapod_descriptions/"
    "urdf/hexapod/hexapod.usda"
)

##
# Joint name groups
# Derived from the Physics scope in payloads/Physics/physics.usda:
#   right_front_coxa / right_front_femur / right_front_tibia
#   right_middle_coxa / right_middle_femur / right_middle_tibia
#   right_back_coxa  / right_back_femur  / right_back_tibia
#   left_front_coxa  / left_front_femur  / left_front_tibia
#   left_middle_coxa / left_middle_femur / left_middle_tibia
#   left_back_coxa   / left_back_femur   / left_back_tibia
##

# All 18 joints in order (Isaac Lab regex-friendly names)
_ALL_JOINTS = [
    "right_front_coxa",
    "right_front_femur",
    "right_front_tibia",
    "right_middle_coxa",
    "right_middle_femur",
    "right_middle_tibia",
    "right_back_coxa",
    "right_back_femur",
    "right_back_tibia",
    "left_front_coxa",
    "left_front_femur",
    "left_front_tibia",
    "left_middle_coxa",
    "left_middle_femur",
    "left_middle_tibia",
    "left_back_coxa",
    "left_back_femur",
    "left_back_tibia",
]

# Coxa joints (hip rotation, Z-axis, ±55 deg)
_COXA_JOINTS_REGEX = ".*_coxa"

# Femur + tibia joints (leg swing & extension, Y-axis)
_FEMUR_TIBIA_JOINTS_REGEX = ".*(femur|tibia)"

##
# ArticulationCfg
##

PXMARK4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_HEXAPOD_USD_PATH,
        # Select the "physx" physics variant which provides
        # PhysicsDriveAPI:angular drives on all revolute joints.
        variants={"Physics": "physx"},
        activate_contact_sensors=True,  # required for ContactSensorCfg in scene
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    # Place the robot slightly above ground to avoid interpenetration at spawn.
    # The body "pxmark4" root frame is at the base_link centroid.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),  # 0.18 m matches target standing height
        joint_pos={
            # Rest pose: coxa centred, femur & tibia at a stable standing angle.
            # Coxa : 0 rad  — limits [-55°, +55°]
            ".*_coxa": 0.0,
            # Femur: +30°  — limits [-98°, +90°] — safely inside
            ".*_femur": math.radians(30.0),
            # Tibia: -30°  — limits [-45°, +125°] = [-0.785, 2.182] rad
            # NOTE: -60° (-1.047 rad) was below the lower limit of -45° (-0.785 rad)
            ".*_tibia": math.radians(-30.0),
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # ------------------------------------------------------------------
        # Coxa actuators — hip rotation about Z-axis
        # Effort limit (urdf:limit:effort) = 3 Nm per joint
        # Velocity limit (urdf:limit:velocity) = π rad/s
        # ------------------------------------------------------------------
        "coxa": ImplicitActuatorCfg(
            joint_names_expr=[_COXA_JOINTS_REGEX],
            effort_limit=3.0,
            velocity_limit=math.pi,
            stiffness=20.0,  # position gain (Kp)
            damping=0.5,  # velocity gain (Kd)
        ),
        # ------------------------------------------------------------------
        # Femur + tibia actuators — leg swing / extension about Y-axis
        # ------------------------------------------------------------------
        "femur_tibia": ImplicitActuatorCfg(
            joint_names_expr=[_FEMUR_TIBIA_JOINTS_REGEX],
            effort_limit=3.0,
            velocity_limit=math.pi,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
"""ArticulationCfg for the PhantomX Mark IV hexapod robot (pxmark4).

Key parameters
--------------
- 18 revolute joints, all driven with :class:`ImplicitActuatorCfg`.
- Effort limit : 3 Nm (from URDF).
- Velocity limit: π rad/s (from URDF).
- Default rest pose places coxa at 0°, femur at +30°, tibia at -30°.

To override actuator gains or spawn height in a specific environment,
use :meth:`~isaaclab.utils.configclass.ConfigBase.replace`::

    from hexasim.assets.robot import PXMARK4_CFG

    MY_ROBOT_CFG = PXMARK4_CFG.replace(
        init_state=PXMARK4_CFG.init_state.replace(pos=(0.0, 0.0, 0.25))
    )
"""

PXMARK4_AX12A_CFG = PXMARK4_CFG.replace(
    actuators={"ax12a": AX12AActuatorCfg(joint_names_expr=[".*"])},
    init_state=PXMARK4_CFG.init_state.replace(
        joint_pos={".*_coxa": 0.0, ".*_femur": math.radians(30.0), ".*_tibia": math.radians(30.0)},
    ),
)
"""PhantomX Mark IV with the AX-12A servo model (v7 §5.1) on all 18 joints — S0 onward.

Rest pose femur +30°, tibia +30° (was −30°): the static torque envelope
(``scripts/tests/static_torque_envelope.py``, lever-arm torques from contact forces) puts tripod-stance femur
torque at 12 % of the 1.5 N·m stall vs 134 % at the Phase 1 pose (spec §4.2 gate: ≤ 60 %).
:data:`PXMARK4_CFG` keeps the Phase 1 pose and idealised implicit PD (π_base reference runs only).
"""
