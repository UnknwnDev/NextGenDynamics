# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from ..paths import SPIDER_USD_PATH


SPIDER_JOINT_INFO = {
    "default_pos": {
        "joint_body_leg_hip_.*": math.radians(0.0),
        "joint_leg_hip_leg_upper_.*": math.radians(30.0),
        "joint_leg_upper_leg_middle_.*": math.radians(-75.0),
        "joint_leg_middle_leg_lower_.*": math.radians(-45.0),
    },
    "limit_min": {
        "joint_body_leg_hip_.*": math.radians(-45.0),
        "joint_leg_hip_leg_upper_.*": math.radians(-10.0),
        "joint_leg_upper_leg_middle_.*": math.radians(-105.0),
        "joint_leg_middle_leg_lower_.*": math.radians(-85.0),
    },
    "limit_max": {
        "joint_body_leg_hip_.*": math.radians(45.0),
        "joint_leg_hip_leg_upper_.*": math.radians(60.0),
        "joint_leg_upper_leg_middle_.*": math.radians(-35.0),
        "joint_leg_middle_leg_lower_.*": math.radians(-5.0),
    },
}

VEL_LIMIT = 6.0


SPIDER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        "joint_body_leg_hip_.*",
        "joint_leg_hip_leg_upper_.*",
        "joint_leg_upper_leg_middle_.*",
        "joint_leg_middle_leg_lower_.*",
    ],
    effort_limit_sim={
        "joint_body_leg_hip_.*": 40.0,
        "joint_leg_hip_leg_upper_.*": 40.0,
        "joint_leg_upper_leg_middle_.*": 80.0,
        "joint_leg_middle_leg_lower_.*": 60.0,
    },
    stiffness={
        "joint_body_leg_hip_.*": 80.0,
        "joint_leg_hip_leg_upper_.*": 70.0,
        "joint_leg_upper_leg_middle_.*": 45.0,
        "joint_leg_middle_leg_lower_.*": 20.0,
    },
    velocity_limit_sim={
        "joint_body_leg_hip_.*": VEL_LIMIT,
        "joint_leg_hip_leg_upper_.*": VEL_LIMIT,
        "joint_leg_upper_leg_middle_.*": 0.75 * VEL_LIMIT,
        "joint_leg_middle_leg_lower_.*": 0.5 * VEL_LIMIT,
    },
    damping={
        "joint_body_leg_hip_.*": 1.5,
        "joint_leg_hip_leg_upper_.*": 1.5,
        "joint_leg_upper_leg_middle_.*": 1.0,
        "joint_leg_middle_leg_lower_.*": 1.0,
    },
    friction=0.05,
    armature=0.005,
)


SPIDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SPIDER_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos=SPIDER_JOINT_INFO["default_pos"],
    ),
    actuators={"legs": SPIDER_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)

