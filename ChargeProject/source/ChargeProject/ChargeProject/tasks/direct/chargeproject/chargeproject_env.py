from __future__ import annotations
import math
import colorsys
from time import time
from enum import IntEnum

import gymnasium as gym
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from ChargeProject.tasks.direct.chargeproject.utils.cloudpoint_to_bev import BEVDebugVisualizer  # debug only
from ChargeProject.tasks.direct.chargeproject.environment.terrain_gen.custom_terrain_generator import terrain_generator
from ChargeProject.tasks.direct.chargeproject.environment.terrain_gen.mesh_loader import get_obstacle_radius
from ChargeProject.tasks.direct.chargeproject.environment.map_manager import MapManager, MapManagerOutput
from .environment.spider_robot import SPIDER_ACTUATOR_CFG

from .chargeproject_env_cfg import ChargeprojectEnvCfg, RobotState

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
import omni.physics.tensors as tensors

import inspect
import os


class ChargeprojectEnv(DirectRLEnv):
    cfg: ChargeprojectEnvCfg

    def __init__(
        self, cfg: ChargeprojectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self._robot.find_joints(SPIDER_ACTUATOR_CFG.joint_names_expr)
        
        # Target positions
        self._desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._next_desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._time_since_target = torch.zeros(self.num_envs, device=self.device)
        self._targets_reached = torch.zeros(self.num_envs, device=self.device)
        self._time_outs = torch.zeros(self.num_envs, device=self.device)
        self._life_time = torch.zeros(self.num_envs, device=self.device)
        

        self._actions = torch.zeros(
            (self.num_envs, gym.spaces.flatdim(self.single_action_space)),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )


        self._actions = torch.zeros(
            (self.num_envs, gym.spaces.flatdim(self.single_action_space)),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        self._last_targets_reached = torch.zeros(self.num_envs, device=self.device)

        self.died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        # self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.contact_sensor_base_ids, _ = self._contact_sensor.find_bodies(self.cfg.base_name)
        self.contact_sensor_feet_ids, _ = self._contact_sensor.find_bodies(self.cfg.foot_names)
        self.undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
            self.cfg.undesired_contact_body_names
        )
        self.body_ids, _ = self._robot.find_bodies(self.cfg.base_name)
        self.lower_leg_ids, _ = self._robot.find_bodies(self.cfg.lower_leg_names)

        # Change to initialize with nulls
        self._distance_buffer = torch.zeros(self.num_envs, self.cfg.distance_lookback, device=self.device)

        # Spawn points (terrain_height_map is now managed by MapManager)
        self._spawn_points = torch.from_numpy(terrain_generator.spawn_points).to(dtype=torch.float32, device=self.device)

        # --- Game AI State Machine ---
        self.robot_state = torch.full((self.num_envs,), RobotState.PATROL, dtype=torch.int64, device=self.device)

        # --- MapManager for patrol/exploration ---
        self.map_manager = MapManager(
            config=self.cfg,
            num_envs=self.num_envs,
            device=self.device,
            height_scanner=self._height_scanner,
            lidar_sensor=self._lidar_sensor
        )
        self.last_exploration_bonus = torch.zeros(self.num_envs, device=self.device)
        self.far_staleness = torch.zeros(self.num_envs, 8, device=self.device)

        # Load obstacle circles for collision checking (x, y, radius)
        if terrain_generator.obstacle_placement is not None:
            obstacle_circles = []
            for obs_type, data in terrain_generator.obstacle_placement.items():
                positions = data.get("positions")  # (N, 3)
                scales = data.get("scales")        # (N, 3)
                if positions is None or scales is None:
                    continue
                radii = get_obstacle_radius(obs_type, scales)
                for i in range(len(positions)):
                    obstacle_circles.append([positions[i, 0], positions[i, 1], radii[i]])
            self._obstacle_circles = torch.tensor(obstacle_circles, dtype=torch.float32, device=self.device) if obstacle_circles else None
        else:
            self._obstacle_circles = None

        # Get Gravity and Up Direction of World
        sim_view = tensors.create_simulation_view("torch")
        sim_view.set_subspace_roots("/")
        g = sim_view.get_gravity()                           # carb.Float3
        g = torch.tensor(g, device=self.device, dtype=torch.float32)
        self._up_dir = -g / (g.norm() + 1e-8)

        log_dir = self.cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._bev_debug_visualizer = BEVDebugVisualizer()

        self.extras["log"] = dict()

        # Save the variables in ChargeprojectEnvCfg to a text file for reference
        with open(os.path.join(log_dir, "env_config.txt"), "w") as f:
            for attr, value in vars(self.cfg).items():
                f.write(f"{attr}: {value}\n")

        current_file = inspect.getfile(inspect.currentframe())
        with open(os.path.join(log_dir, "env_code.py.txt"), "w") as f:
            with open(current_file, "r") as current_f:
                f.write(current_f.read())

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.scene.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.scene.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self._lidar_sensor = RayCaster(self.cfg.scene.lidar_sensor)
        self.scene.sensors["lidar_sensor"] = self._lidar_sensor
        
        self.goal_pos_visualizer = self._create_sphere_markers(
            self.cfg.marker_colors, 0.25, "/Visuals/Command/goal_position"
        )
        self.identifier_visualizer = self._create_arrow_markers(
            self.cfg.marker_colors, "/Visuals/Command/identifier_arrow"
        )


        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot

        # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

        self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        target_distance = torch.linalg.norm(self._desired_pos - self._robot.data.root_pos_w, dim=1)

 
        self._visualize_markers(target_distance)

        index = self._sim_step_counter % self.cfg.distance_lookback
        
        self._distance_buffer[:, index] = target_distance.squeeze(-1)

    def _apply_action(self) -> None:
        self.processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:, self.dof_idx]
        )

        # For testing standing on 3 legs
        #if self.common_step_counter % 400 >= 200:
        #    self.processed_actions = self._robot.data.default_joint_pos[:, self.dof_idx]
        #    even_joints = self._robot.find_joints(".*0.*|.*2.*|.*4.*")[0]
        #    even_joints = [j for j in even_joints if j in self.dof_idx]
        #    self.processed_actions[:, even_joints] = 0
        #else:
        #    self.processed_actions = 0
        
        # print the actions of the first env
        #if self._sim_step_counter % 2 == 0:
        #    print(self.processed_actions[0].shape, self.processed_actions[0])
        self._robot.set_joint_position_target(self.processed_actions, joint_ids=self.dof_idx)
        # self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
    
    def _get_relative_target_info(
        self, target_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_target_pos_w = target_pos_w - self._robot.data.root_pos_w
        distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)

        # Rotate the world vector into the robot's local body frame
        relative_target_pos_b = math_utils.quat_apply_inverse(
            self._robot.data.root_quat_w, relative_target_pos_w
        )

        # Normalize the resulting 2D body-frame vector to get the unit vector
        unit_vector_b = relative_target_pos_b / (distance + 1e-6)

        return unit_vector_b, distance
    

    def _get_observations_impl(self) -> dict:

        target_unit_vector, target_distance = self._get_relative_target_info(
            self._desired_pos
        )

        next_target_unit_vector, next_target_distance = self._get_relative_target_info(
            self._next_desired_pos
        )

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        self.is_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces, dim=-1
                ),
                dim=1,
            )[0]
            > self.cfg.contact_threshold
        )

        self._target_distance = target_distance.squeeze(-1)

        # --- MapManager update: get all map data in one call ---
        map_output = self.map_manager.update(
            env_origins=self.scene.env_origins,
            robot_pos_w=self._robot.data.root_pos_w,
            robot_yaw_w=self._robot.data.heading_w.unsqueeze(-1),
            dt=self.step_dt
        )
        self.last_exploration_bonus = map_output.exploration_bonus
        self.far_staleness = map_output.far_staleness

        # can_see: hardcoded to False for now (player not implemented yet)
        can_see = torch.zeros(self.num_envs, 1, device=self.device)

        # one_hot_state: WAYPOINT=0, PATROL=1
        one_hot_state = torch.nn.functional.one_hot(
            self.robot_state, num_classes=2
        ).float()

        # Concatenate the selected observations into a single tensor.
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    target_unit_vector,
                    target_distance,
                    next_target_unit_vector,
                    next_target_distance,
                    self.is_contact,
                    # self._commands,
                    self._robot.data.joint_pos[:, self.dof_idx] - self._robot.data.default_joint_pos[:, self.dof_idx],
                    self._robot.data.joint_vel[:, self.dof_idx],
                    self._actions,
                    # --- New observations for Game AI ---
                    map_output.far_staleness,    # (num_envs, 8)
                    can_see,          # (num_envs, 1)
                    one_hot_state,    # (num_envs, 2)
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {
            "observations": obs,
            "height_data": map_output.height_data,
            "bev_data": map_output.bev_data,
            "nav_data": map_output.nav_data.squeeze(1)  # (num_envs, 1, 33, 33) -> (num_envs, 33, 33)
        }
    
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        observations = self._get_observations_impl()
        # Need to clone because of torch.compile
        observations = {"policy": observations}# for rl_games, "critic": observations.clone()}
        return observations

   #@torch.compile(mode="reduce-overhead")    
    def _get_rewards_impl(self, reached_target_ids) -> torch.Tensor:
        target_unit_vector, target_distance = self._get_relative_target_info(self._desired_pos)

        # - Rewards -
        # Reward for progress towards the target
        sum_valid = torch.nansum(self._distance_buffer, dim=1)
        count_valid = torch.clamp(torch.sum(~torch.isnan(self._distance_buffer), dim=1), min=1.0)
        previous_buffered_distance = sum_valid / count_valid
        difference = (previous_buffered_distance - target_distance.squeeze(-1)) * (1 + 0.5 * (count_valid-1))
        progress_reward = torch.sign(difference) * torch.pow(torch.abs(difference), self.cfg.progress_pow)
        #progress_reward = difference
        progress_reward *= (self._targets_reached * 0.5) + 1

        # torch.exp(self._targets_reached / self.cfg.progress_target_divisor)
        # Multipliy progress reward by distance to target (closer = more reward)
        # progress_reward *= (5 / torch.maximum(target_distance, torch.tensor(1, device=self.device)))
        
        # Velocity alignment the target
        velocity_alignment_reward = torch.nn.functional.cosine_similarity(
           self._robot.data.root_lin_vel_w[:, :2], target_unit_vector[:, :2], dim=1
        )
        # velocity_alignment_reward = torch.pow(velocity_alignment_reward, 3.0)  # emphasize forward movement

        # VAR * 2 if reached target
        # *2 for positive,*1/2 for negative
        # hit_target = self._targets_reached > 0
        # multiplier = torch.minimum(1 + self._targets_reached, torch.tensor(3.5, device=self.device))
        # hit_and_positive = hit_target & (velocity_alignment_reward > 0)
        # velocity_alignment_reward[hit_and_positive] *= multiplier[hit_and_positive]
        # hit_and_negative = hit_target & (velocity_alignment_reward < 0)
        # velocity_alignment_reward[hit_and_negative] /= multiplier[hit_and_negative] * 2.0
        # print("--- vel:", self._robot.data.root_lin_vel_w[:, :2], "\nunit_vector_to_target:", unit_vector_to_target,
        #    "\nvelocity_alignment_reward", velocity_alignment_reward)

        # Bonus for get_ting to target (1 reward for first target, 1.5 for second, 2 for third, etc...)
        target_reward = torch.zeros(self.num_envs, device=self.device)
        target_reward[reached_target_ids] = (
            0.5 + self._targets_reached[reached_target_ids] * 0.5
        )

        # time_penalty = torch.ones(self.num_envs, device=self.device)

        # net_contact_forces = self._contact_sensor.data.net_forces_w_history

        body_angular_velocity = torch.linalg.norm(self._robot.data.body_link_ang_vel_w[:, self.body_ids, :], dim=(-1, 1))

        # died if gravity is near positive (flipped over)
        died = self._robot.data.projected_gravity_b[:, 2] > 0.0
        base_contact_time = self._contact_sensor.data.current_contact_time[:, self.contact_sensor_base_ids].squeeze(-1)
        on_ground = base_contact_time > self.cfg.base_on_ground_time
        death_penalty = died.float() + on_ground.float()

        # Reward for just moving forward in the robot's frame (sqrt)
        # forward_vel_reward = torch.sign(self._robot.data.root_lin_vel_b[:, 0]) \
        #                   * torch.pow(torch.abs(self._robot.data.root_lin_vel_b[:, 0]), 1.0/2.0)
        # forward_vel_reward = torch.sign(self._robot.data.root_lin_vel_b[:, 0]) \
        #                   * torch.square(torch.abs(self._robot.data.root_lin_vel_b[:, 0]))
        # forward_vel_reward = self._robot.data.root_lin_vel_b[:, 0]

        
        # Get horizontal speed

        v_w = self._robot.data.body_link_lin_vel_w[:, self.body_ids]
        dot = (v_w * self._up_dir).sum(dim=-1, keepdim=True)
        v_horizontal = v_w - dot * self._up_dir
        horizontal_speed = v_horizontal.norm(dim=-1).squeeze(1)
    
        # # Combine linear and angular speed.
        # motion_metric = - (horizontal_speed - self.cfg.still_threshold) * self.cfg.motion_metric_pow
        # still_penalty = torch.exp(motion_metric)

        # linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        # z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # # angular velocity x/y
        # ang_vel_error = torch.sum(
        #     torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        # )
        # joint torques
        joint_num = len(self._robot.data.joint_acc[0])
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque / joint_num), dim=1)

        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc / joint_num), dim=1)

        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        # joint velocity
        # dof_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self.contact_sensor_feet_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self.contact_sensor_feet_ids]
        
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)

        contact_time = self._contact_sensor.data.current_contact_time[:, self.contact_sensor_feet_ids]
        in_contact = (contact_time > 0.0).to(contact_time.dtype) 

        feet_ground_max_time = torch.max(contact_time * in_contact, dim=1).values

        
        # body vertical acceleration
        acc_lin_w = self._robot.data.body_com_lin_acc_w
        body_vertical_acceleration = torch.mean(acc_lin_w[:, self.body_ids, 2], dim=1)
        body_vertical_acceleration = torch.pow(body_vertical_acceleration, 2.0).clip(max=20.0)

        # check jump (all feet in the air at the same time)
        current_air_times = self._contact_sensor.data.current_air_time[:, self.contact_sensor_feet_ids]
        air_feet_per_agent = (current_air_times > 0).float().sum(dim=1)
        # jump_penalty = (current_air_times > 0).all(dim=1).float()
        total_foot_num = float(len(self.contact_sensor_feet_ids))
        normalized_air_feet = air_feet_per_agent / total_foot_num # normalize to [0, 1]
        normalized_air_feet[normalized_air_feet < 0.0000001] = 2.0 / total_foot_num  # If no feet on air, equal to two foot on air
        jump_penalty = normalized_air_feet * normalized_air_feet * normalized_air_feet  # cubic penalty

        # force of feet in contact with ground
        feet_contact_forces = self._contact_sensor.data.net_forces_w[:, self.contact_sensor_feet_ids]
        feet_contact_forces_mean = torch.mean(torch.norm(feet_contact_forces, dim=-1), dim=1) ** 2.0

        # undesired contacts
        undesired_contacts = torch.sum(self.is_contact[:, self.undesired_contact_body_ids], dim=1)
        # If 3 or more feet are in contact, consider it stable
        # stable_contact = torch.clamp(torch.sum(self.is_contact[:, self.contact_sensor_feet_ids], dim=1), max=4.0)

        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        # Wall Proximity Penalty
        lidar_hits_w = self._lidar_sensor.data.ray_hits_w  # (batch_size, num_points, 3)
        lidar_hits_w = torch.nan_to_num(lidar_hits_w, nan=0.0, posinf=1000.0, neginf=-1000.0)
        rel_hits_w = lidar_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)  # (batch_size, num_points, 3)

        # Flatten for robust broadcasting in TorchScript
        batch_size, num_points, _ = rel_hits_w.shape
        rel_hits_w_flat = rel_hits_w.view(-1, 3)
        
        quat_w_expanded = self._lidar_sensor.data.quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
        
        rel_hits_b_flat = math_utils.quat_apply_inverse(quat_w_expanded, rel_hits_w_flat)
        rel_hits_b = rel_hits_b_flat.view(batch_size, num_points, 3)

        dists = torch.norm(rel_hits_b, dim=-1)  # (batch_size, num_points)
        is_close = dists < self.cfg.wall_close_threshold
        is_obstacle = rel_hits_b[:, :, 2] > self.cfg.wall_height_threshold

        valid_wall_hits= is_close & is_obstacle

        wall_proximity_score = torch.sum(
            (self.cfg.wall_close_threshold - dists) * valid_wall_hits.float(), 
            dim=1
        )

        wall_proximity_score = wall_proximity_score / float(num_points)
        wall_proximity_score = wall_proximity_score * wall_proximity_score * torch.sign(wall_proximity_score)


        # Feet height penalty
        # base_height = self._robot.data.body_pos_w[:, self.contact_sensor_base_ids, 2]  # [envs, 1]
        # feet_height = self._robot.data.body_pos_w[:, self.contact_sensor_feet_ids, 2] # [envs, num_feet]

        # Compute positive difference (only if feet are above base)
        # feet_above_base = torch.clamp(feet_height - base_height, min=0.0)

        # Mean or sum over feet (you can adjust depending on desired strength)
        # feet_height_penalty = torch.mean(feet_above_base, dim=1)  # [envs]

        # Lower leg upright penalty
        # lower_leg_quats_w = self._robot.data.body_quat_w[:, self.lower_leg_ids]
        # local_leg_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 1, 3)
        # expanded_leg_axis = local_leg_axis.expand(lower_leg_quats_w.shape[0], lower_leg_quats_w.shape[1], 3)

        # world_leg_axis = math_utils.quat_apply(lower_leg_quats_w, expanded_leg_axis)
        # cos_angle = torch.sum(world_leg_axis * self._up_dir, dim=-1)
        # angle_from_vertical_rad = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        # lower_leg_penalty = torch.mean(angle_from_vertical_rad, dim=1)

        # --- Patrol Rewards ---
        patrol_mask = (self.robot_state == RobotState.PATROL).float()

        # Exploration reward (cleared staleness)
        patrol_exploration = patrol_mask * self.last_exploration_bonus

        # Boundary penalty (distance outside patrol zone)
        rel_pos = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        dist_from_center = torch.norm(rel_pos, dim=1)
        outside_patrol = torch.clamp(dist_from_center - (self.cfg.patrol_size / 2), min=0.0)
        patrol_boundary = patrol_mask * outside_patrol

        return {
            # "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "life_time_reward": self._life_time * self.cfg.life_time_reward_scale * self.step_dt,
            "progress_reward": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            # "time_penalty": time_penalty * self.cfg.time_penalty_scale * self.step_dt,
            "velocity_alignment_reward": velocity_alignment_reward * self.cfg.velocity_alignment_reward_scale * self.step_dt,
            "reach_target_reward": target_reward * self.cfg.reach_target_reward_scale,
            "death_penalty": death_penalty * self.cfg.death_penalty_scale,
            "feet_ground_time": feet_ground_max_time * self.cfg.feet_ground_time_penalty_scale * self.step_dt,
            "jump_penalty": jump_penalty * self.cfg.jump_penalty_scale * self.step_dt,
            # "forward_vel": forward_vel_reward * self.cfg.forward_vel_reward_scale * self.step_dt,
            "body_angular_velocity_penalty": body_angular_velocity * self.cfg.body_angular_velocity_penalty_scale * self.step_dt,
            # "still_penalty": still_penalty * self.cfg.still_penalty_scale * self.step_dt,
            "speed_reward": horizontal_speed * self.cfg.speed_reward_scale * self.step_dt,
            # "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_penalty_scale * self.step_dt,
            # "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "body_vertical_acceleration": body_vertical_acceleration * self.cfg.body_vertical_acceleration_penalty_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            # "dof_vel_l2": dof_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "feet_contact_force": feet_contact_forces_mean * self.cfg.feet_contact_penalty_scale * self.step_dt,
            #"desired_contacts": stable_contact * self.cfg.stable_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            #"feet_height_penalty": feet_height_penalty * self.cfg.feet_height_penalty_scale * self.step_dt,
            #"lower_leg_penalty": lower_leg_penalty * self.cfg.lower_leg_penalty_scale * self.step_dt,
            "wall_proximity_penalty": wall_proximity_score * self.cfg.wall_proximity_penalty_scale * self.step_dt,
            # --- Patrol Rewards ---
            "patrol_exploration": patrol_exploration * self.cfg.patrol_exploration_reward_scale * self.step_dt,
            "patrol_boundary": patrol_boundary * self.cfg.patrol_boundary_penalty_scale * self.step_dt,
        } # type: ignore


    def _get_rewards(self) -> torch.Tensor:

        # Check if distance is within tolerance
        target_distance = torch.linalg.norm(self._desired_pos - self._robot.data.root_pos_w, dim=1)
        reached_target = target_distance.squeeze(-1) < self.cfg.success_tolerance
        reached_target_ids = reached_target.nonzero(as_tuple=False).squeeze(-1)

        # Generate a new target immediately for the environments that reached theirs
        if len(reached_target_ids) > 0:
            self._move_next_targets(reached_target_ids)

        rewards = self._get_rewards_impl(reached_target_ids)

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        if self.cfg.log:
            self._log_data("Episode_Reward/total_reward", torch.mean(reward))
            for key, value in rewards.items():
                episodic_sum_avg = torch.mean(value)
                self._log_data(f"Episode_Reward/{key}", episodic_sum_avg)
        
            # Add the average and max amount of targets reached to log
            self._log_data(
                "Episode_Info/targets_reached_avg",
                self._last_targets_reached.float().mean(),
            )
            self._log_data(
                "Episode_Info/targets_reached_max", self._last_targets_reached.max()
            )
            # Counts for thresholds 1 through 8 in one go
            thresholds = torch.arange(1, self.cfg.log_targets_reached_max, self.cfg.log_targets_reached_step, device=self.device)
            counts = (self._last_targets_reached.unsqueeze(-1) >= thresholds).sum(dim=0) / self.num_envs

            for _, (t, c) in enumerate(zip(thresholds, counts), start=1):
                self._log_data(f"Episode_Info/targets_reached_{t}", c)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        full_time_out = self.episode_length_buf >= self.max_episode_length - 1

        self._life_time += self.step_dt
        self._time_since_target += self.step_dt
        timed_out = self._time_since_target > self._time_outs
        # change it so seperate gtime_outs
        
        died = self._robot.data.projected_gravity_b[:, 2] > 0.0
        base_contact_time = self._contact_sensor.data.current_contact_time[:, self.contact_sensor_base_ids].squeeze(-1)
        on_ground = base_contact_time > self.cfg.base_on_ground_time
        
        # Logging deaths/time outs per second
        if self.cfg.log:
            self._log_data("Episode_Termination/full_time_out", torch.count_nonzero(full_time_out) / self.num_envs)
            self._log_data("Episode_Termination/time_out", torch.count_nonzero(timed_out) / self.num_envs)
            self._log_data("Episode_Termination/died", torch.count_nonzero(died) / self.num_envs)
            self._log_data("Episode_Termination/on_ground", torch.count_nonzero(on_ground) / self.num_envs)

        self.terminated = died | on_ground
        self.truncated = timed_out | full_time_out
        return self.terminated, self.truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES  # type: Sequence[int]

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Reset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._last_targets_reached[env_ids] = self._targets_reached[env_ids].clone()


        self._time_outs[env_ids] = self.cfg.time_out_per_target
        self._life_time[env_ids] = 0.0
        self._targets_reached[env_ids] = 0

        # Reset Game AI state
        self.robot_state[env_ids] = RobotState.PATROL
        self.map_manager.reset(env_ids)
        self.last_exploration_bonus[env_ids] = 0.0

        if self.scene.terrain is None:
            print("Terrain not found, cannot sample spawn positions.")
            return
        
        # Setup respawn position using flat patches
        # patches = self.scene.terrain.flat_patches["robot_spawn"]   # shape: (rows, cols, num_patches, 3)
        # patches = patches.reshape(-1, 3)   # patches.reshape(-1, 3)

        # Sample from pre-computed valid spawn points
        num_resets = len(env_ids)
        # Randomly select indices from the available spawn points
        spawn_indices = torch.randint(0, len(self._spawn_points), (num_resets,), device=self.device)
        patches = self._spawn_points[spawn_indices]

        n = len(env_ids)
        idx = torch.randperm(patches.shape[0])[:n]
        pos_w = patches[idx].clone()  
        pos_w[:, 2] += 1.0 # slight offset above ground

        # Random yaw
        yaw = (torch.rand(n, device=pos_w.device) - 0.5) * 0.5
        qw = torch.cos(0.5 * yaw)
        qz = torch.sin(0.5 * yaw)
        quat_w = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)

        root_pose_w = torch.cat([pos_w, quat_w], dim=-1) 
        self._robot.write_root_pose_to_sim(root_pose_w, env_ids)

        # Set Next target positions to be reset position
        self._next_desired_pos[env_ids] = self._robot.data.root_pos_w[env_ids].clone()
        self._move_next_targets(env_ids)
        self._move_next_targets(
            env_ids
        )  # call twice to initialize both current and next target positions
        if len(env_ids) == self.num_envs:
            # For initial randomize the initial timeout
            self._time_since_target[:] = (-self.cfg.time_out_per_target + 
                torch.rand(self.num_envs, device=self.device) * self.cfg.time_out_per_target)

        self._time_outs[env_ids] = self.cfg.time_out_per_target
        self._targets_reached[env_ids] = 0

    
    def _log_data(self, key, data) -> None:
        self.extras["log"][key] = data

    def _reached_target(self, env_ids):
        # Get robot and target positions (only x, y)
        robot_pos = self._robot.data.root_pos_w[env_ids, :2]
        target_pos = self._desired_pos[env_ids]
        # Calculate distance
        dist = torch.linalg.norm(robot_pos - target_pos, dim=1)

        # Check if distance is within tolerance
        return dist < self.cfg.success_tolerance

    def _check_obstacle_collision(self, positions_xy: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """
        Check if positions collide with any obstacle.
        Args:
            positions_xy: (N, 2) tensor of XY positions to check
            margin: additional clearance around obstacles
        Returns:
            (N,) bool tensor, True if position collides with any obstacle
        """
        if self._obstacle_circles is None:
            return torch.zeros(positions_xy.shape[0], dtype=torch.bool, device=self.device)

        # positions_xy: (N, 2), obstacles: (M, 3) where [:, :2] is xy, [:, 2] is radius
        obs_xy = self._obstacle_circles[:, :2]      # (M, 2)
        obs_radii = self._obstacle_circles[:, 2]    # (M,)

        # Compute distances: (N, M)
        diff = positions_xy.unsqueeze(1) - obs_xy.unsqueeze(0)  # (N, 1, 2) - (1, M, 2) = (N, M, 2)
        distances = torch.linalg.norm(diff, dim=2)               # (N, M)

        # Check collision with any obstacle
        collides = (distances < (obs_radii + margin)).any(dim=1)  # (N,)
        return collides

    def _move_next_targets(self, env_ids: Sequence[int]):
        # Update current position
        self._desired_pos[env_ids] = self._next_desired_pos[env_ids].clone()

        num_resets = len(env_ids)
        env_ids_t: torch.Tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # Generate new target positions with obstacle avoidance
        remaining_mask = torch.ones(num_resets, dtype=torch.bool, device=self.device)
        new_target_pos_xy = torch.zeros(num_resets, 2, device=self.device)

        for _ in range(self.cfg.target_sample_attempts):
            if not remaining_mask.any():
                break

            n_remaining = int(remaining_mask.sum().item())
            remaining_local_idx = remaining_mask.nonzero(as_tuple=True)[0]

            radius = self.cfg.point_max_distance + (
                self.cfg.point_min_distance - self.cfg.point_max_distance
            ) * torch.rand(n_remaining, device=self.device)
            angle = 2 * math.pi * torch.rand(n_remaining, device=self.device)

            # Get positions for remaining environments
            remaining_env_ids = env_ids_t[remaining_local_idx]
            candidate_xy = self._desired_pos[remaining_env_ids, :2] + torch.stack(
                [radius * torch.cos(angle), radius * torch.sin(angle)], dim=1
            )

            collides = self._check_obstacle_collision(candidate_xy, margin=self.cfg.target_obstacle_margin)

            # Accept non-colliding positions
            valid_mask = ~collides
            valid_local_idx = remaining_local_idx[valid_mask]
            new_target_pos_xy[valid_local_idx] = candidate_xy[valid_mask]

            # Update remaining for next iteration
            remaining_mask[valid_local_idx] = False

        # For any remaining (couldn't find valid), use last candidate anyway
        if remaining_mask.any():
            remaining_local_idx = remaining_mask.nonzero(as_tuple=True)[0]
            n_remaining = int(remaining_mask.sum().item())
            radius = self.cfg.point_max_distance + (
                self.cfg.point_min_distance - self.cfg.point_max_distance
            ) * torch.rand(n_remaining, device=self.device)
            angle = 2 * math.pi * torch.rand(n_remaining, device=self.device)
            remaining_env_ids = env_ids_t[remaining_local_idx]
            fallback_xy = self._desired_pos[remaining_env_ids, :2] + torch.stack(
                [radius * torch.cos(angle), radius * torch.sin(angle)], dim=1
            )
            new_target_pos_xy[remaining_local_idx] = fallback_xy

        new_target_pos_z = self._get_terrain_height(new_target_pos_xy).unsqueeze(-1) + 0.25
        new_target_pos = torch.cat([new_target_pos_xy, new_target_pos_z], dim=-1)

        # Update current and next desired positions
        self._next_desired_pos[env_ids] = new_target_pos

        # set _distance_buffer to nan and add current distance as element
        self._distance_buffer[env_ids] = torch.nan
        index = self._sim_step_counter % self.cfg.distance_lookback
        distance = torch.linalg.norm(
            self._robot.data.root_pos_w[env_ids] - self._desired_pos[env_ids], dim=1
        )
        self._distance_buffer[env_ids, index] = distance

        # Reset timer
        self._time_since_target[env_ids] = 0.0
        self._targets_reached[env_ids] += 1
        self._time_outs[env_ids] = (
            self.cfg.time_out_per_target
            - self._targets_reached[env_ids] * self.cfg.time_out_decrease_per_target
        )

    def _get_random_colors(self, num_colors: int) -> list[tuple[float, float, float]]:
        colors = []
        for i in range(num_colors):
            # Evenly space hues for maximum color distinction
            hue = (i / min(19.0, num_colors)) % 1.0
            saturation = 0.9
            value = (i % 3) / 3.0 * 0.5 + 0.5  # Vary brightness
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb_color)
        return colors

    def _create_sphere_markers(
        self, num_markers: int, radius: float, prim_path: str, opacity: float = 1
    ) -> VisualizationMarkers:
        colors = self._get_random_colors(num_markers)
        markers = {}
        for i, color in enumerate(colors):
            marker_key = f"sphere{i}"
            markers[marker_key] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, opacity=opacity
                ),
            )
        marker_cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        return VisualizationMarkers(marker_cfg)

    def _create_arrow_markers(
        self, num_markers: int, prim_path: str
    ) -> VisualizationMarkers:
        colors = self._get_random_colors(num_markers)
        markers = {}
        for i, color in enumerate(colors):
            marker_key = f"arrow{i}"
            # Load the arrow mesh provided by Isaac Lab
            markers[marker_key] = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        marker_cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        return VisualizationMarkers(marker_cfg)

    def _visualize_markers_impl(self, target_distance):
        marker_indices = (
            torch.arange(self.num_envs, device=self.device) % self.cfg.marker_colors
        )

        # --- Identifier Arrow Marker ---
        relative_target_pos = self._desired_pos - self._robot.data.root_pos_w
        yaw = torch.atan2(relative_target_pos[:, 1], relative_target_pos[:, 0])
        orientations = math_utils.quat_from_angle_axis(yaw, self._up_dir)

        # Linearly interpolate scale from 0.15 down to 0 as the robot gets closer
        # Clamp between 0.0 and 1.0 to handle cases where the robot is farther than max_dist
        # The maximum distance should be `self.cfg.point_max_distance`
        scale_factor = torch.clamp(
            target_distance / self.cfg.point_max_distance, 0.0, 1.0
        )

        # Base scale for the arrow's width and height
        arrow_thickness_scale = 0.15 * scale_factor.unsqueeze(1) + 0.05
        # Make the arrow's length (x-axis) a bit longer for better visibility, relative to thickness
        scales = torch.cat(
            [2 * arrow_thickness_scale, arrow_thickness_scale, arrow_thickness_scale],
            dim=1,
        ).squeeze(-1)

        arrow_positions = self._robot.data.root_pos_w + torch.tensor(
            [0.0, 0.0, 0.5], device=self.device
        )

        return self._desired_pos, marker_indices, arrow_positions, orientations, scales

    def _visualize_markers(self, target_distance):
        (
            desired_pos_3d,
            marker_indices,
            arrow_positions,
            orientations,
            scales,
        ) = self._visualize_markers_impl(target_distance)
        self.goal_pos_visualizer.visualize(
            desired_pos_3d, marker_indices=marker_indices
        )
        self.identifier_visualizer.visualize(
            translations=arrow_positions,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )

    def _get_terrain_height(self, positions_xy: torch.Tensor) -> torch.Tensor:
        """Delegate to MapManager for terrain height lookup."""
        if self.scene.terrain is None:
            raise RuntimeError("Terrain not found, cannot get height information.")
        return self.map_manager.get_terrain_height(positions_xy)
    
    def switch_mode(self, env_ids, new_state: RobotState):
        self.robot_state[env_ids] = new_state
        if new_state == RobotState.WAYPOINT:
            self._move_next_targets(env_ids) # set new target
        elif new_state == RobotState.PATROL:
            self.map_manager.reset(env_ids) # reset exploration map



