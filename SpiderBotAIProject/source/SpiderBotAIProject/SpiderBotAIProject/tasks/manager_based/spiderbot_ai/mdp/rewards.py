# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for SpiderBotAIProject.

Each function computes its value inline from sensors/robot data.
No ensure_updated() or FeatureCacheCommandTerm dependency.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def _mode_scale(env, reward_name: str) -> torch.Tensor:
    """Per-env reward scale factor based on current mode.

    Returns (num_envs,) tensor. Falls back to 1.0 if reward_name
    is not in env.cfg.mode_reward_scales.
    """
    mode_term = env.command_manager.get_term("mode")
    scales_tuple = env.cfg.mode_reward_scales.get(reward_name)
    if scales_tuple is None:
        return torch.ones(env.num_envs, device=env.device)
    scales_tensor = torch.tensor(scales_tuple, device=env.device, dtype=torch.float32)
    return scales_tensor[mode_term.current_mode]


def life_time_reward(env) -> torch.Tensor:
    # life_time was: accumulator += step_dt each step = episode_length_buf * step_dt
    return env.episode_length_buf.float() * env.step_dt * env.step_dt


def progress_reward(env) -> torch.Tensor:
    waypoint = env.command_manager.get_term("waypoint")
    robot = env.scene.articulations["robot"]

    target_distance = torch.linalg.norm(waypoint.desired_pos - robot.data.root_pos_w, dim=1)
    previous_distance = waypoint.get_previous_distance()

    # Instantaneous step-to-step delta: positive = approaching target
    delta = previous_distance - target_distance
    delta = torch.nan_to_num(delta, nan=0.0)  # First step after reset → 0
    # Zero out on target-transition step to avoid spike from mismatched targets
    delta = torch.where(waypoint.reached_target, torch.zeros_like(delta), delta)

    progress = delta * ((waypoint.targets_reached * 0.25) + 1.0)

    return progress * _mode_scale(env, "progress") * env.step_dt


def velocity_alignment_reward(env) -> torch.Tensor:
    waypoint = env.command_manager.get_term("waypoint")
    robot = env.scene.articulations["robot"]

    relative_target_pos_w = waypoint.desired_pos - robot.data.root_pos_w
    distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)
    target_unit_vector_w = relative_target_pos_w / (distance + 1e-6)
    velocity_alignment = torch.nn.functional.cosine_similarity(
        robot.data.root_lin_vel_w[:, :2], target_unit_vector_w[:, :2], dim=1
    )

    return velocity_alignment * _mode_scale(env, "velocity_alignment") * env.step_dt


def reach_target_reward(env) -> torch.Tensor:
    waypoint = env.command_manager.get_term("waypoint")

    reward = (0.5 + waypoint.targets_reached * 0.5) * waypoint.reached_target.to(dtype=waypoint.targets_reached.dtype)
    return reward * _mode_scale(env, "reach_target")


def death_penalty(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    died = robot.data.projected_gravity_b[:, 2] > 0.0
    on_ground = env._base_contact_time > float(env.cfg.base_on_ground_time)
    return died.to(torch.float32) + on_ground.to(torch.float32)


def feet_ground_time_penalty(env) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_sensor"]
    contact_time = contact_sensor.data.current_contact_time[:, env.robot_idx.contact_sensor_feet_ids]
    in_contact = (contact_time > 0.0).to(contact_time.dtype)
    return torch.max(contact_time * in_contact, dim=1).values * env.step_dt


def jump_penalty(env) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_sensor"]
    current_air_times = contact_sensor.data.current_air_time[:, env.robot_idx.contact_sensor_feet_ids]
    air_feet_per_agent = (current_air_times > 0).float().sum(dim=1)
    total_foot_num = float(len(env.robot_idx.contact_sensor_feet_ids))
    normalized_air_feet = air_feet_per_agent / total_foot_num
    normalized_air_feet[normalized_air_feet < 1e-7] = 2.0 / total_foot_num
    return (normalized_air_feet ** 3) * env.step_dt


def body_angular_velocity_penalty(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    return torch.linalg.norm(
        robot.data.body_link_ang_vel_w[:, env.robot_idx.body_ids, :], dim=(-1, 1)
    ) * env.step_dt


def speed_reward(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    up_dir = torch.tensor([0.0, 0.0, 1.0], device=env.device).view(1, 1, 3)
    v_w = robot.data.body_link_lin_vel_w[:, env.robot_idx.body_ids]
    dot = (v_w * up_dir).sum(dim=-1, keepdim=True)
    v_horizontal = v_w - dot * up_dir
    return v_horizontal.norm(dim=-1).squeeze(1) * _mode_scale(env, "speed") * env.step_dt


def body_vertical_acceleration_penalty(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    body_vertical_acc = torch.mean(robot.data.body_com_lin_acc_w[:, env.robot_idx.body_ids, 2], dim=1)
    return torch.pow(body_vertical_acc, 2.0).clip(max=20.0) * env.step_dt


def dof_torques_l2(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    joint_num = float(robot.data.joint_acc.shape[1])
    return torch.sum(torch.square(robot.data.applied_torque / joint_num), dim=1) * env.step_dt


def dof_acc_l2(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    joint_num = float(robot.data.joint_acc.shape[1])
    return torch.sum(torch.square(robot.data.joint_acc / joint_num), dim=1) * env.step_dt


def action_rate_l2(env) -> torch.Tensor:
    delta_action = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta_action), dim=1) * env.step_dt


def feet_air_time_reward(env) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_sensor"]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, env.robot_idx.contact_sensor_feet_ids]
    last_air_time = contact_sensor.data.last_air_time[:, env.robot_idx.contact_sensor_feet_ids]
    return torch.sum((1.0 - 4.0 * (last_air_time - 0.5) ** 2) * first_contact, dim=1) * env.step_dt


def undesired_contacts_penalty(env) -> torch.Tensor:
    return torch.sum(env._is_contact[:, env.robot_idx.undesired_contact_body_ids], dim=1) * env.step_dt


def feet_contact_force_penalty(env) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_sensor"]
    feet_forces = contact_sensor.data.net_forces_w[:, env.robot_idx.contact_sensor_feet_ids]
    return (torch.mean(torch.norm(feet_forces, dim=-1), dim=1) ** 2.0) * env.step_dt


def flat_orientation_l2(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1) * env.step_dt


def wall_proximity_penalty(env) -> torch.Tensor:
    lidar_sensor = env.scene.sensors["lidar_sensor"]

    lidar_hits_w = lidar_sensor.data.ray_hits_w
    lidar_hits_w = torch.nan_to_num(lidar_hits_w, nan=0.0, posinf=1000.0, neginf=-1000.0)
    rel_hits_w = lidar_hits_w - lidar_sensor.data.pos_w.unsqueeze(1)

    batch_size, num_points, _ = rel_hits_w.shape
    rel_hits_w_flat = rel_hits_w.view(-1, 3)
    quat_w_expanded = lidar_sensor.data.quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)

    rel_hits_b_flat = math_utils.quat_apply_inverse(quat_w_expanded, rel_hits_w_flat)
    rel_hits_b = rel_hits_b_flat.view(batch_size, num_points, 3)

    dists = torch.norm(rel_hits_b, dim=-1)
    is_close = dists < float(env.cfg.wall_close_threshold)
    is_obstacle = rel_hits_b[:, :, 2] > float(env.cfg.wall_height_threshold)
    valid_wall_hits = is_close & is_obstacle

    wall_score = torch.sum((float(env.cfg.wall_close_threshold) - dists) * valid_wall_hits.float(), dim=1)
    wall_score = wall_score / float(num_points)
    return (wall_score * wall_score * torch.sign(wall_score)) * env.step_dt


def patrol_exploration_reward(env) -> torch.Tensor:
    return env._map_output.exploration_bonus * _mode_scale(env, "patrol_exploration") * env.step_dt


def patrol_boundary_penalty(env) -> torch.Tensor:
    robot = env.scene.articulations["robot"]
    rel_pos = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dist_from_center = torch.norm(rel_pos, dim=1)
    outside_patrol = torch.clamp(dist_from_center - (float(env.cfg.patrol_size) / 2), min=0.0)
    return outside_patrol * _mode_scale(env, "patrol_boundary") * env.step_dt


def chase_proximity_reward(env) -> torch.Tensor:
    """Continuous reward for staying close to the moving target in CHASE mode."""
    waypoint = env.command_manager.get_term("waypoint")
    robot = env.scene.articulations["robot"]
    dist = torch.linalg.norm(waypoint.desired_pos - robot.data.root_pos_w, dim=1)
    proximity = 1.0 / (dist + 0.5)
    return proximity * _mode_scale(env, "chase_proximity") * env.step_dt
