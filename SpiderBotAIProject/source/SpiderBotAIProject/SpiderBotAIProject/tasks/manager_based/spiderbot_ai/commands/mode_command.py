# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg


class ModeCommandTerm(CommandTerm):
    """Robot mode state owned by CommandsManager.

    Modes:
        0 = WAYPOINT
        1 = PATROL
        2 = CHASE
    """

    WAYPOINT = 0
    PATROL = 1
    CHASE = 2
    NUM_MODES = 3

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        # Desired/assigned mode (can be set by other terms/systems).
        self.robot_mode = torch.full((self.num_envs,), self.WAYPOINT, dtype=torch.int64, device=self.device)

        # Permanent mode assignment per env (computed once on first reset).
        self._assigned_mode = torch.full((self.num_envs,), self.WAYPOINT, dtype=torch.int64, device=self.device)
        self._assigned_mode_initialized = False

        # Cached mode state for pull-based "events" (updated once per sim step).
        self.current_mode = self.robot_mode.clone()
        self.prev_mode = self.robot_mode.clone()
        self.mode_changed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.entered_waypoint = torch.zeros_like(self.mode_changed)
        self.exited_waypoint = torch.zeros_like(self.mode_changed)
        self.entered_chase = torch.zeros_like(self.mode_changed)
        self.exited_chase = torch.zeros_like(self.mode_changed)

        self._one_hot = torch.zeros(self.num_envs, self.NUM_MODES, device=self.device, dtype=torch.float32)
        self._last_update_step = -1

    @property
    def command(self) -> torch.Tensor:
        return self._one_hot

    @property
    def is_waypoint(self) -> torch.Tensor:
        return self.current_mode == self.WAYPOINT

    @property
    def is_patrol(self) -> torch.Tensor:
        return self.current_mode == self.PATROL

    @property
    def is_chase(self) -> torch.Tensor:
        return self.current_mode == self.CHASE

    def ensure_updated(self) -> None:
        """Updates cached one-hot mode once per environment step."""
        step = int(self._env.common_step_counter)
        if self._last_update_step == step:
            return
        self._last_update_step = step

        # Cache transition state (pull-based "events").
        self.prev_mode.copy_(self.current_mode)
        self.current_mode.copy_(torch.clamp(self.robot_mode, min=0, max=self.NUM_MODES - 1).to(dtype=torch.int64))

        self.mode_changed[:] = self.current_mode != self.prev_mode
        self.entered_waypoint[:] = self.mode_changed & (self.current_mode == self.WAYPOINT)
        self.exited_waypoint[:] = self.mode_changed & (self.prev_mode == self.WAYPOINT)
        self.entered_chase[:] = self.mode_changed & (self.current_mode == self.CHASE)
        self.exited_chase[:] = self.mode_changed & (self.prev_mode == self.CHASE)

        idx = self.current_mode.unsqueeze(1)
        self._one_hot.zero_()
        self._one_hot.scatter_(1, idx, 1.0)

    def _compute_assigned_modes(self):
        """Compute permanent mode assignment per env based on spawn_mode config."""
        if self._env.cfg.spawn_mode == "mixed":
            # env 0->WAYPOINT, env 1->PATROL, env 2->CHASE, env 3->WAYPOINT, ...
            self._assigned_mode = torch.arange(self.num_envs, device=self.device, dtype=torch.int64) % self.NUM_MODES
        else:
            self._assigned_mode[:] = self.WAYPOINT
        self._assigned_mode_initialized = True

    def reset(self, env_ids: Sequence[int] | None = None):
        if not self._assigned_mode_initialized:
            self._compute_assigned_modes()

        if env_ids is None:
            mode = self._assigned_mode
            self.robot_mode[:] = mode
            self.current_mode[:] = mode
            self.prev_mode[:] = mode
            self.mode_changed[:] = False
            self.entered_waypoint[:] = False
            self.exited_waypoint[:] = False
            self.entered_chase[:] = False
            self.exited_chase[:] = False
            self._one_hot.zero_()
            self._one_hot.scatter_(1, mode.unsqueeze(1), 1.0)
        else:
            mode = self._assigned_mode[env_ids]
            self.robot_mode[env_ids] = mode
            self.current_mode[env_ids] = mode
            self.prev_mode[env_ids] = mode
            self.mode_changed[env_ids] = False
            self.entered_waypoint[env_ids] = False
            self.exited_waypoint[env_ids] = False
            self.entered_chase[env_ids] = False
            self.exited_chase[env_ids] = False
            one_hot = torch.zeros(len(env_ids), self.NUM_MODES, device=self.device, dtype=torch.float32)
            one_hot.scatter_(1, mode.unsqueeze(1), 1.0)
            self._one_hot[env_ids] = one_hot

        return super().reset(env_ids=env_ids)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        self.ensure_updated()

