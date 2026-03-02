# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


def spawn_robot(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
    """Event function: spawn robot at terrain-sampled positions on reset.

    Runs during ``event_manager.apply(mode="reset")`` which executes BEFORE
    ``command_manager.reset()``, ensuring ``env.spawn_pos_w`` is available
    for ``WaypointCommandTerm.reset()``.
    """
    robot = env.scene.articulations["robot"]

    if env_ids is None:
        env_ids_t = robot._ALL_INDICES
    else:
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    # Sample spawn positions from terrain data
    env_origins = env.scene.env_origins[env_ids_t]
    spawn_xy = env.terrain_data.sample_spawn(env_origins, patrol_size=float(env.cfg.patrol_size))

    # Add Z at terrain height + offset
    z = env.terrain_data.height_at_xy(spawn_xy) + float(env.cfg.spawn_z_offset)
    spawn_pos_w = torch.cat([spawn_xy, z.unsqueeze(1)], dim=1)

    # Set robot pose
    default_root_state = robot.data.default_root_state[env_ids_t].clone()
    default_root_state[:, :3] = spawn_pos_w

    # Random yaw
    yaw = (torch.rand(env_ids_t.numel(), device=env.device) - 0.5) * float(env.cfg.spawn_yaw_range)
    qw = torch.cos(0.5 * yaw)
    qz = torch.sin(0.5 * yaw)
    quat_w = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)
    default_root_state[:, 3:7] = quat_w

    # Write to simulation
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_t)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_t)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids_t],
        robot.data.default_joint_vel[env_ids_t],
        None,
        env_ids_t,
    )
    # Reset PhysX PD controller targets (prevents stale targets from previous episode)
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids_t], env_ids=env_ids_t)
    robot.set_joint_velocity_target(robot.data.default_joint_vel[env_ids_t], env_ids=env_ids_t)

    # Store spawn position for WaypointCommandTerm to reference during its reset
    env.spawn_pos_w[env_ids_t] = spawn_pos_w
