# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from . import mdp
from .commands.feature_cache_command import FeatureCacheCommandTerm
from .commands.map_command import MapCommandTerm
from .commands.mode_command import ModeCommandTerm
from .commands.robot_cache_command import RobotCacheCommandTerm
from .commands.spawn_command import SpawnCommandTerm
from .commands.terrain_command import TerrainCommandTerm
from .commands.waypoint_command import WaypointCommandTerm
from .environment.spider_robot import SPIDER_ACTUATOR_CFG, SPIDER_CFG
from .paths import CUSTOM_TERRAIN_USD_PATH

##
# Scene definition
##


@configclass
class SpiderBotAISceneCfg(InteractiveSceneCfg):
    """Configuration for SpiderBotAIProject scene."""

    terrain: terrain_gen.TerrainImporterCfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="usd",
        usd_path=str(CUSTOM_TERRAIN_USD_PATH),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot = SPIDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[2.52, 2.52]),  # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/terrain"],
    )

    lidar_sensor: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        mesh_prim_paths=["/World/terrain"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-60.0, 60.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / 129.0,
        ),
    )

    dome = AssetBaseCfg(
        prim_path="/World/Lights/Dome",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SPIDER_ACTUATOR_CFG.joint_names_expr,
        scale=0.75,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class CommandsCfg:
    """Stateful components (owned by CommandTerms)."""

    terrain = CmdTerm(class_type=TerrainCommandTerm, resampling_time_range=(1.0e9, 1.0e9))
    spawn = CmdTerm(class_type=SpawnCommandTerm, resampling_time_range=(1.0e9, 1.0e9))
    robot_cache = CmdTerm(class_type=RobotCacheCommandTerm, resampling_time_range=(1.0e9, 1.0e9))
    map = CmdTerm(class_type=MapCommandTerm, resampling_time_range=(1.0e9, 1.0e9))
    waypoint = CmdTerm(class_type=WaypointCommandTerm, resampling_time_range=(1.0e9, 1.0e9))
    mode = CmdTerm(class_type=ModeCommandTerm, resampling_time_range=(1.0e9, 1.0e9))
    features = CmdTerm(class_type=FeatureCacheCommandTerm, resampling_time_range=(1.0e9, 1.0e9))


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        observations = ObsTerm(func=mdp.policy_observations)
        height_data = ObsTerm(func=mdp.height_data)
        bev_data = ObsTerm(func=mdp.bev_data)
        nav_data = ObsTerm(func=mdp.nav_data)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms (thin wrappers over command-owned buffers)."""

    # NOTE: We keep rewards unscaled in mdp functions and apply scaling here via manager weights.
    #       This keeps the RewardManager table meaningful and avoids hiding scales inside env.cfg.* fields.
    life_time = RewTerm(func=mdp.life_time_reward, weight=0.005)
    progress = RewTerm(func=mdp.progress_reward, weight=5.0e4)
    velocity_alignment = RewTerm(func=mdp.velocity_alignment_reward, weight=2.5e2)
    reach_target = RewTerm(func=mdp.reach_target_reward, weight=1.0e5)

    death_penalty = RewTerm(func=mdp.death_penalty, weight=-5.0e5)
    feet_ground_time = RewTerm(func=mdp.feet_ground_time_penalty, weight=-5.0e1)
    jump_penalty = RewTerm(func=mdp.jump_penalty, weight=-2.5e3)
    body_angular_velocity = RewTerm(func=mdp.body_angular_velocity_penalty, weight=-15.0)
    speed = RewTerm(func=mdp.speed_reward, weight=1.0e3)
    body_vertical_acceleration = RewTerm(func=mdp.body_vertical_acceleration_penalty, weight=-3.0)
    dof_torques = RewTerm(func=mdp.dof_torques_l2, weight=-0.25)
    dof_acc = RewTerm(func=mdp.dof_acc_l2, weight=-2.5e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    feet_air_time = RewTerm(func=mdp.feet_air_time_reward, weight=5.0e2)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts_penalty, weight=-2.0e2)
    feet_contact_force = RewTerm(func=mdp.feet_contact_force_penalty, weight=-1.0e-7)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0e3)
    wall_proximity = RewTerm(func=mdp.wall_proximity_penalty, weight=0.0)

    patrol_exploration = RewTerm(func=mdp.patrol_exploration_reward, weight=2.0)
    patrol_boundary = RewTerm(func=mdp.patrol_boundary_penalty, weight=-0.25)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    per_target_time_out = DoneTerm(func=mdp.per_target_time_out, time_out=True)
    died = DoneTerm(func=mdp.died)
    on_ground = DoneTerm(func=mdp.on_ground)


@configclass
class EventCfg:
    """No-op events.

    All reset/randomization logic is intentionally implemented as CommandTerms.
    """


##
# Environment configuration
##


@configclass
class SpiderBotAIEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SpiderBotAISceneCfg = SpiderBotAISceneCfg(num_envs=int(1024.0 * 0.5), env_spacing=4.0, replicate_physics=True)

    # Managers
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # --- Task parameters (ported) ---
    episode_length_s = 60.0

    base_name = "body"
    foot_names = "leg_foot_.*"
    undesired_contact_body_names = "body|leg_upper_.*|leg_middle_.*|leg_lower_.*"

    # Commands / reset
    spawn_z_offset = 10.0
    spawn_yaw_range = 0.5

    # Terrain / sampling (ported)
    height_map_size_x = 196.0
    height_map_size_y = 196.0
    height_map_meter_per_grid = 0.15
    spawn_padding = 20.0
    target_sample_attempts = 24
    target_obstacle_margin = 0.5
    spawn_z_offset_small = 1.0
    target_z_offset = 0.25

    # Patrol settings
    patrol_size = 18.0
    staleness_dim = 18
    staleness_decay_rate = 1.0
    nav_size = 24.0
    nav_dim = 33

    # Target settings
    distance_lookback = 8
    point_max_distance = 10.0
    point_min_distance = 5.0
    success_tolerance = 0.5
    time_out_per_target = 30.0
    time_out_decrease_per_target = 0.075
    min_time_out = 1.0

    # Reward shaping parameters (scales live in RewardsCfg weights).
    progress_pow = 1.3
    wall_close_threshold = 1.5
    wall_height_threshold = -0.2

    # Sensors / contacts
    contact_threshold = 1.0e-2
    base_on_ground_time = 1.0

    def __post_init__(self) -> None:
        self.decimation = 2
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
