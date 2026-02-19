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

        self._distance_buffer = torch.full(
            (self.num_envs, int(self._env.cfg.distance_lookback)), torch.nan, device=self.device
        )
        self._step_counter = 0
        self._last_update_step = -1
        self._debug_vis_z_offset = torch.tensor([0.0, 0.0, 0.15], device=self.device).view(1, 3)

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

        # When entering waypoint mode, reset per-target timeout bookkeeping so patrol time doesn't count.
        entered_waypoint = mode_term.entered_waypoint
        if torch.any(entered_waypoint):
            entered_ids = entered_waypoint.nonzero(as_tuple=False).squeeze(-1)
            self.time_since_target[entered_ids] = 0.0
            self.reached_target[entered_ids] = False
            self.per_target_timed_out[entered_ids] = False
            self._distance_buffer[entered_ids] = torch.nan

        self._step_counter += 1
        dt = float(self._env.step_dt)
        self.time_since_target += dt * is_waypoint.to(dtype=self.time_since_target.dtype)

        # Determine whether the target has been reached.
        target_distance = torch.linalg.norm(self.desired_pos - self.robot.data.root_pos_w, dim=1)
        self.reached_target = (target_distance < float(self._env.cfg.success_tolerance)) & is_waypoint

        reached_ids = self.reached_target.nonzero(as_tuple=False).squeeze(-1)
        if reached_ids.numel() > 0:
            self._on_reached_target(reached_ids)

        # Timeouts (per-target).
        self.per_target_timed_out = (self.time_since_target > self.time_outs) & is_waypoint

        # Update progress buffer (lookback distance).
        active_ids = is_waypoint.nonzero(as_tuple=False).squeeze(-1)
        if active_ids.numel() > 0:
            self._write_distance_buffer(active_ids, target_distance)

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
        self._distance_buffer[env_ids] = torch.nan

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

    def get_distance_buffer(self) -> torch.Tensor:
        return self._distance_buffer

    def _write_distance_buffer(self, env_ids: torch.Tensor, target_distance: torch.Tensor):
        """Write target distance into the cyclic lookback buffer for a subset of environments."""
        index = self._step_counter % int(self._env.cfg.distance_lookback)
        self._distance_buffer[env_ids, index] = target_distance[env_ids]

    def _on_reached_target(self, env_ids: torch.Tensor):
        self.targets_reached[env_ids] += 1.0
        self.desired_pos[env_ids] = self.next_desired_pos[env_ids].clone()
        self.next_desired_pos[env_ids] = self._sample_target_positions(self.desired_pos[env_ids])

        self.time_since_target[env_ids] = 0.0
        new_time_outs = float(self._env.cfg.time_out_per_target) - self.targets_reached[env_ids] * float(
            self._env.cfg.time_out_decrease_per_target
        )
        self.time_outs[env_ids] = torch.clamp(new_time_outs, min=float(self._env.cfg.min_time_out))

        self._distance_buffer[env_ids] = torch.nan

    def _resample_targets(self, env_ids):
        lookback = int(self._env.cfg.distance_lookback)
        index = self._step_counter % lookback

        if isinstance(env_ids, slice):
            anchor = self._env.spawn_pos_w
            self.desired_pos[:] = self._sample_target_positions(anchor)
            self.next_desired_pos[:] = self._sample_target_positions(self.desired_pos)

            distance = torch.linalg.norm(anchor - self.desired_pos, dim=1)
            self._distance_buffer[:, index] = distance
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            anchor = self._env.spawn_pos_w[env_ids_t]
            self.desired_pos[env_ids_t] = self._sample_target_positions(anchor)
            self.next_desired_pos[env_ids_t] = self._sample_target_positions(self.desired_pos[env_ids_t])

            distance = torch.linalg.norm(anchor - self.desired_pos[env_ids_t], dim=1)
            self._distance_buffer[env_ids_t, index] = distance

    def _sample_target_positions(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        """Sample obstacle-avoiding 3D target positions around anchors."""
        return self._env.terrain_data.sample_target(anchor_pos_w, self._env.cfg)
