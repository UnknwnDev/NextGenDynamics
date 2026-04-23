# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import IntEnum
import math

import numpy as np

from ChargeProject.tasks.direct.chargeproject.environment.environments import MySceneCfg
from ChargeProject.tasks.direct.chargeproject.environment.terrain_gen.custom_terrain_generator import terrain_generator
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from gymnasium import spaces


class RobotState(IntEnum):
    WAYPOINT = 0    # Target point navigation
    PATROL = 1      # Patrol mode
    # Future states:
    # INVESTIGATE = 2, LOOK = 3, ATTACK = 4, HIDE = 5

@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):

    #always should be on
    log = True

    # env
    episode_length_s = 60.0
    # - spaces definition
    #action_space = 12
    action_space = 24
    #observation_space = 51
    #observation_space = 87 # without height scanner
    # observation_space = 376 # with height scanner
    observation_space = 722 # with height scanner and lidar
    
    # observations: 131 = 120 (base) + 8 (far_staleness) + 1 (can_see) + 2 (one_hot_state)
    observation_space = spaces.Dict({
        "observations": spaces.Box(-math.inf, math.inf, shape=(131,), dtype=np.float32),
        "height_data": spaces.Box(-math.inf, math.inf, shape=(64, 64), dtype=np.float32),
        "bev_data": spaces.Box(-math.inf, math.inf, shape=(3, 64, 64), dtype=np.float32),
        "nav_data": spaces.Box(-math.inf, math.inf, shape=(1, 33, 33), dtype=np.float32)  # staleness only
    })
    state_space = 0 #idk why this is here
    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation,
        physx=PhysxCfg(
            #gpu_collision_stack_size = 2**27,
            gpu_max_rigid_patch_count = 2**19
        )
    )
    
    # Spider Robot (base, leg_hip_i, leg_middle_i, leg_lower_i, leg_foot_i)
    base_name = "body"
    foot_names = "leg_foot_.*"
    undesired_contact_body_names = "body|leg_upper_.*|leg_middle_.*|leg_lower_.*"
    middle_leg_joint_names = "joint_leg_upper_leg_middle_.*"
    lower_leg_names = "leg_lower_.*"
    lower_leg_joint_names = "joint_leg_middle_leg_lower_.*"
    hip_joint_names = "joint_body_leg_hip_.*"

    # Unitree Go2
    #base_name = "base"
    #foot_names = ".*_foot"
    #undesired_contact_body_names = ".*_thigh"

    # Spot
    # base_name = "body"
    # foot_names = ".*_foot"
    # undesired_contact_body_names = ".*_uleg"

    # Anymal
    # base_name = "base"
    # foot_names = ".*FOOT"
    # undesired_contact_body_names = ".*THIGH"

    
    # scene
    scene: MySceneCfg = MySceneCfg( # type: ignore
        num_envs= int(1024.0 * 0.5),
        env_spacing=4.0,
        replicate_physics=True
    )

    height_map_size_x = terrain_generator.config.size[0]
    height_map_size_y = terrain_generator.config.size[1]
    height_map_meter_per_grid = terrain_generator.config.meter_per_grid

    # --- Game AI Settings ---
    # Patrol settings
    patrol_size = 18.0  # Meters (Area the robot is expected to patrol)
    staleness_dim = 18  # Grid resolution for staleness map (patrol_size / staleness_res)
    staleness_decay_rate = 1.0  # Rate at which areas become "stale" again

    # Nav data settings (egocentric observation)
    nav_size = 24.0  # Size of nav observation in meters
    nav_dim = 33  # Grid resolution for nav observation (must match observation_space)

    # Patrol reward scales
    patrol_exploration_reward_scale = 2.0  # Reward for exploring new areas
    patrol_boundary_penalty_scale = -0.25  # Penalty for leaving patrol zone

    spawn_padding = 20.0  # meters from edge of height map to spawn robots within
    target_sample_attempts = 24  # tries per env to sample a target that avoids obstacles
    target_obstacle_margin = 0.5  # extra meters to keep away from obstacle radius
    
    point_max_distance = 10 #20 #6.0
    point_min_distance = 5 #10 #4.0
    success_tolerance = 0.5 #1  # meters
    time_out_per_target = 30.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds
    base_on_ground_time = 1.0 #seconds before death if base is on ground

    log_targets_reached_max = 10
    log_targets_reached_step = 1


    marker_colors = 57

    # Final rewards
    action_scale = 0.75
    
    progress_reward_scale = 5.0e3 # 500 # linear version ish
    progress_pow = 1.3
    distance_lookback = 8
    velocity_alignment_reward_scale = 2.5e2 # 10.0 #2 #6
    # Multiplied by targets hit reward
    reach_target_reward_scale = 1.0e5 # 1.0e4
    life_time_reward_scale = 0.005
    death_penalty_scale = -5.0e5 # -1000.0 # -500.0
    speed_reward_scale = 2.5e2
    jump_penalty_scale = -2.5e3
    feet_contact_penalty_scale = -1.0e-7
    joint_torque_reward_scale = -0.25
    joint_accel_reward_scale = -2.5e-04 # idk this works # -1.5e-07
    action_rate_reward_scale = -0.5
    body_angular_velocity_penalty_scale = -15.0
    body_vertical_acceleration_penalty_scale = -3.0
    feet_air_time_reward_scale = 5.0e2
    feet_ground_time_penalty_scale = -5.0e1
    contact_threshold = 1.0e-2
    undesired_contact_reward_scale = -2.0e2 ## -0.75
    flat_orientation_reward_scale = -2.0e3
    wall_proximity_penalty_scale = 0.0 # -1.0e3
    wall_close_threshold = 1.5 # meters
    wall_height_threshold = -0.2
