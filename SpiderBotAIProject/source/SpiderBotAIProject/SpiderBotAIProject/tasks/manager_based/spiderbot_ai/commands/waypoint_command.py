# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG


class WaypointCommandTerm(CommandTerm):
    """Waypoint/timeout/progress state machine owned by CommandsManager."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        self.robot = self._env.scene.articulations["robot"]

        self.desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.next_desired_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.time_since_target = torch.zeros(self.num_envs, device=self.device)
        self.time_outs = torch.full((self.num_envs,), float(self._env.cfg.time_out_per_target), device=self.device)
        self.targets_reached = torch.zeros(self.num_envs, device=self.device)

        self.reached_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.per_target_timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._chase_heading = torch.zeros(self.num_envs, device=self.device)

        self._previous_distance = torch.full((self.num_envs,), float('nan'), device=self.device)
        self._previous_distance_snapshot = torch.full((self.num_envs,), float('nan'), device=self.device)
        self._step_counter = 0
        self._last_update_step = -1
        self._debug_vis_z_offset = torch.tensor([0.0, 0.0, 0.15], device=self.device).view(1, 3)

        # Patrol staleness targeting
        self._patrol_time_since_update = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.desired_pos

    def ensure_updated(self) -> None:
        """Updates waypoint state once per environment step.

        This is intentionally callable from reward/termination functions to ensure the state-machine
        runs before rewards/dones are computed (matching direct env semantics).
        """
        step = int(self._env.common_step_counter)
        if self._last_update_step == step:
            return
        self._last_update_step = step

        # Pull-based mode transition signals.
        mode_term = self._env.command_manager.get_term("mode")
        mode_term.ensure_updated()
        is_waypoint = mode_term.is_waypoint
        is_chase = mode_term.is_chase
        is_patrol = mode_term.is_patrol
        is_target_active = is_waypoint | is_chase | is_patrol

        # When entering waypoint or chase mode, reset per-target timeout bookkeeping.
        entered_target_mode = mode_term.entered_waypoint | mode_term.entered_chase
        if torch.any(entered_target_mode):
            entered_ids = entered_target_mode.nonzero(as_tuple=False).squeeze(-1)
            self.time_since_target[entered_ids] = 0.0
            self.reached_target[entered_ids] = False
            self.per_target_timed_out[entered_ids] = False
            self._previous_distance[entered_ids] = float('nan')
            n_entered = entered_ids.numel()
            self._chase_heading[entered_ids] = 2.0 * torch.pi * torch.rand(n_entered, device=self.device)

        # When entering patrol mode, force immediate staleness target update.
        if torch.any(mode_term.entered_patrol):
            patrol_enter_ids = mode_term.entered_patrol.nonzero(as_tuple=False).squeeze(-1)
            self._patrol_time_since_update[patrol_enter_ids] = float('inf')
            self._previous_distance[patrol_enter_ids] = float('nan')

        self._step_counter += 1
        dt = float(self._env.step_dt)
        self.time_since_target += dt * is_target_active.to(dtype=self.time_since_target.dtype)

        # Snapshot previous distance for progress reward, then update.
        target_distance = torch.linalg.norm(self.desired_pos - self.robot.data.root_pos_w, dim=1)
        self._previous_distance_snapshot = self._previous_distance.clone()
        self._previous_distance[:] = target_distance

        # Determine whether the target has been reached.
        tolerance = torch.where(
            is_chase,
            torch.tensor(float(self._env.cfg.chase_success_tolerance), device=self.device),
            torch.where(
                is_patrol,
                torch.tensor(float(self._env.cfg.patrol_target_tolerance), device=self.device),
                torch.tensor(float(self._env.cfg.success_tolerance), device=self.device),
            ),
        )
        self.reached_target = (target_distance < tolerance) & is_target_active

        reached_ids = self.reached_target.nonzero(as_tuple=False).squeeze(-1)
        if reached_ids.numel() > 0:
            self._on_reached_target(reached_ids)

        # Timeouts (per-target, only for waypoint mode).
        self.per_target_timed_out = (self.time_since_target > self.time_outs) & is_waypoint

        # Move chase targets periodically.
        chase_ids = is_chase.nonzero(as_tuple=False).squeeze(-1)
        if chase_ids.numel() > 0:
            self._update_chase_target(chase_ids, dt)

        # Update patrol targets from staleness peaks.
        patrol_ids = is_patrol.nonzero(as_tuple=False).squeeze(-1)
        if patrol_ids.numel() > 0:
            self._update_patrol_target(patrol_ids, dt)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        # Episode logging (values before reset).
        if isinstance(env_ids, slice):
            tr = self.targets_reached
        else:
            tr = self.targets_reached[env_ids]
        extras = {
            "Episode_Info/targets_reached_avg": float(tr.float().mean().item()),
            "Episode_Info/targets_reached_max": float(tr.float().max().item()),
        }

        if isinstance(env_ids, slice):
            # Spread out initial timeouts to avoid synchronized resets (matches direct env semantics).
            t = float(self._env.cfg.time_out_per_target)
            self.time_since_target[:] = (-t + torch.rand(self.num_envs, device=self.device) * t)
        else:
            self.time_since_target[env_ids] = 0.0
        self.targets_reached[env_ids] = 0.0
        self.time_outs[env_ids] = float(self._env.cfg.time_out_per_target)
        self.reached_target[env_ids] = False
        self.per_target_timed_out[env_ids] = False
        self._previous_distance[env_ids] = float('nan')
        n_reset = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._chase_heading[env_ids] = 2.0 * torch.pi * torch.rand(n_reset, device=self.device)
        self._patrol_time_since_update[env_ids] = float('inf')  # force update on first step

        self._resample_targets(env_ids)
        super().reset(env_ids=env_ids)
        return extras

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        self.ensure_updated()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):
                goal_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Commands/waypoint_goal")
                goal_cfg.markers["sphere"].radius = 0.12
                goal_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                self._goal_visualizer = VisualizationMarkers(goal_cfg)

                next_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Commands/waypoint_next_goal")
                next_cfg.markers["sphere"].radius = 0.08
                next_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0))
                self._next_goal_visualizer = VisualizationMarkers(next_cfg)

            self._goal_visualizer.set_visibility(True)
            self._next_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_goal_visualizer"):
                self._goal_visualizer.set_visibility(False)
                self._next_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self._goal_visualizer.visualize(translations=self.desired_pos + self._debug_vis_z_offset)
        self._next_goal_visualizer.visualize(translations=self.next_desired_pos + self._debug_vis_z_offset)

    def get_previous_distance(self) -> torch.Tensor:
        """Return the distance snapshot from the previous step (NaN if unavailable)."""
        return self._previous_distance_snapshot

    def _on_reached_target(self, env_ids: torch.Tensor):
        self.targets_reached[env_ids] += 1.0
        self.desired_pos[env_ids] = self.next_desired_pos[env_ids].clone()
        self.next_desired_pos[env_ids] = self._sample_target_positions(self.desired_pos[env_ids])

        self.time_since_target[env_ids] = 0.0
        new_time_outs = float(self._env.cfg.time_out_per_target) - self.targets_reached[env_ids] * float(
            self._env.cfg.time_out_decrease_per_target
        )
        self.time_outs[env_ids] = torch.clamp(new_time_outs, min=float(self._env.cfg.min_time_out))

        self._previous_distance[env_ids] = float('nan')

    def _update_chase_target(self, chase_ids: torch.Tensor, dt: float):
        """Wander the chase target smoothly each step."""
        n = chase_ids.numel()
        cfg = self._env.cfg
        td = self._env.terrain_data

        # 1. Random heading perturbation.
        wander_rate = float(cfg.chase_target_wander_rate)
        noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * wander_rate * dt
        self._chase_heading[chase_ids] += noise

        # 2. Boundary repulsion steering.
        margin = float(cfg.chase_target_boundary_margin)
        cur_xy = self.desired_pos[chase_ids, :2]
        x_min = -td.size_x / 2.0 + float(cfg.spawn_padding)
        x_max = td.size_x / 2.0 - float(cfg.spawn_padding)
        y_min = -td.size_y / 2.0 + float(cfg.spawn_padding)
        y_max = td.size_y / 2.0 - float(cfg.spawn_padding)

        repulsion = torch.zeros(n, 2, device=self.device)
        repulsion[:, 0] += (cur_xy[:, 0] < x_min + margin).float()
        repulsion[:, 0] -= (cur_xy[:, 0] > x_max - margin).float()
        repulsion[:, 1] += (cur_xy[:, 1] < y_min + margin).float()
        repulsion[:, 1] -= (cur_xy[:, 1] > y_max - margin).float()

        has_repulsion = repulsion.norm(dim=1) > 0
        if torch.any(has_repulsion):
            rep_ids = has_repulsion.nonzero(as_tuple=False).squeeze(-1)
            repulsion_angle = torch.atan2(repulsion[rep_ids, 1], repulsion[rep_ids, 0])
            self._chase_heading[chase_ids[rep_ids]] = repulsion_angle

        # 3. Forward movement.
        speed = float(cfg.chase_target_speed)
        heading = self._chase_heading[chase_ids]
        dx = speed * dt * torch.cos(heading)
        dy = speed * dt * torch.sin(heading)
        new_xy = cur_xy + torch.stack([dx, dy], dim=1)

        new_xy[:, 0].clamp_(x_min, x_max)
        new_xy[:, 1].clamp_(y_min, y_max)

        # 4. Obstacle avoidance — flip heading for colliding envs.
        collides = td.collides(new_xy, margin=float(cfg.target_obstacle_margin))
        if torch.any(collides):
            self._chase_heading[chase_ids[collides]] += torch.pi

        # 5. Apply valid moves.
        valid = ~collides
        valid_ids = chase_ids[valid]
        if valid_ids.numel() > 0:
            valid_xy = new_xy[valid]
            new_z = td.height_at_xy(valid_xy) + float(cfg.target_z_offset)
            self.desired_pos[valid_ids, :2] = valid_xy
            self.desired_pos[valid_ids, 2] = new_z

    def _update_patrol_target(self, patrol_ids: torch.Tensor, dt: float) -> None:
        """Set desired_pos for patrol envs to the highest-staleness region."""
        if patrol_ids.numel() == 0:
            return

        cfg = self._env.cfg
        map_output = self._env._map_output

        # Increment patrol update timer
        self._patrol_time_since_update[patrol_ids] += dt

        # Get current peak data for patrol envs
        peak_xy = map_output.staleness_peak_world[patrol_ids]       # (M, 2)
        peak_val = map_output.staleness_peak_value[patrol_ids]      # (M,)

        # Current distance to target
        robot_xy = self.robot.data.root_pos_w[patrol_ids, :2]
        target_xy = self.desired_pos[patrol_ids, :2]
        dist_to_target = torch.linalg.norm(target_xy - robot_xy, dim=1)

        # Hybrid update conditions
        reached = dist_to_target < float(cfg.patrol_target_tolerance)
        timed_out = self._patrol_time_since_update[patrol_ids] > float(cfg.patrol_target_update_interval)
        has_meaningful_peak = peak_val >= float(cfg.patrol_target_min_staleness)

        # Only update when there IS a meaningful peak to navigate toward
        needs_update = (reached | timed_out) & has_meaningful_peak

        update_ids = patrol_ids[needs_update]
        if update_ids.numel() == 0:
            return

        # Enforce minimum distance: don't target a cell right under the robot
        update_peak_xy = peak_xy[needs_update]
        update_robot_xy = robot_xy[needs_update]
        peak_dist = torch.linalg.norm(update_peak_xy - update_robot_xy, dim=1)
        far_enough = peak_dist > float(cfg.patrol_target_min_distance)

        final_ids = update_ids[far_enough]
        final_xy = update_peak_xy[far_enough]

        if final_ids.numel() == 0:
            return

        # Compute Z from terrain height
        z = self._env.terrain_data.height_at_xy(final_xy) + float(cfg.target_z_offset)

        # Set desired_pos to staleness peak
        self.desired_pos[final_ids, 0] = final_xy[:, 0]
        self.desired_pos[final_ids, 1] = final_xy[:, 1]
        self.desired_pos[final_ids, 2] = z

        # No lookahead for patrol — next target is dynamically recomputed on reach
        self.next_desired_pos[final_ids] = self.desired_pos[final_ids].clone()

        # Reset timers and distance tracking to avoid spurious progress reward spike
        self._patrol_time_since_update[final_ids] = 0.0
        self._previous_distance[final_ids] = float('nan')

    def _resample_targets(self, env_ids):
        if isinstance(env_ids, slice):
            anchor = self._env.spawn_pos_w
            self.desired_pos[:] = self._sample_target_positions(anchor)
            self.next_desired_pos[:] = self._sample_target_positions(self.desired_pos)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            anchor = self._env.spawn_pos_w[env_ids_t]
            self.desired_pos[env_ids_t] = self._sample_target_positions(anchor)
            self.next_desired_pos[env_ids_t] = self._sample_target_positions(self.desired_pos[env_ids_t])

    def _sample_target_positions(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        """Sample obstacle-avoiding 3D target positions around anchors."""
        return self._env.terrain_data.sample_target(anchor_pos_w, self._env.cfg)
