# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.common import VecEnvStepReturn

from .environment.map_manager import MapManager, MapManagerOutput
from .environment.robot_indices import RobotIndices
from .environment.terrain_data import TerrainData


class SpiderBotAIEnv(ManagerBasedRLEnv):
    """SpiderBotAIProject environment (manager-based).

    Design:
        Core data providers live as env attributes. CommandTerms are only used
        for legitimate commands (waypoint, mode). Sensor-derived per-step data
        is computed in ``_compute_step_data()`` which is inserted into the
        ``step()`` loop between physics and MDP computation.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # These must exist BEFORE super().__init__() which calls load_managers().
        # load_managers() creates CommandTerms that may reference these during init.
        self.terrain_data = TerrainData(cfg, device=cfg.sim.device)
        self.spawn_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)

        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        # Robot body/joint/contact indices (resolved after scene is created)
        self.robot_idx = RobotIndices.from_scene(self.scene, self.cfg)

        # Map manager (owns staleness maps, computes BEV/height/nav)
        self._map_manager = MapManager(
            config=self.cfg,
            num_envs=self.num_envs,
            device=self.device,
            height_scanner=self.scene.sensors["height_scanner"],
            lidar_sensor=self.scene.sensors["lidar_sensor"],
        )

        # Per-step output buffers (written by _compute_step_data)
        self._map_output = MapManagerOutput(
            nav_data=torch.zeros(self.num_envs, 1, self.cfg.nav_dim, self.cfg.nav_dim, device=self.device),
            bev_data=torch.zeros(self.num_envs, 3, 64, 64, device=self.device),
            height_data=torch.zeros(self.num_envs, 64, 64, device=self.device),
            far_staleness=torch.zeros(self.num_envs, 8, device=self.device),
            exploration_bonus=torch.zeros(self.num_envs, device=self.device),
            staleness_peak_world=torch.zeros(self.num_envs, 2, device=self.device),
            staleness_peak_value=torch.zeros(self.num_envs, device=self.device),
        )

        contact_sensor = self.scene.sensors["contact_sensor"]
        self._is_contact = torch.zeros(
            self.num_envs,
            contact_sensor.data.net_forces_w_history.shape[-2],
            device=self.device,
            dtype=torch.float32,
        )
        self._base_contact_time = torch.zeros(self.num_envs, device=self.device)

        # Create ObservationManager, TerminationManager, RewardManager etc.
        super().load_managers()

    # ------------------------------------------------------------------
    # step() override — based on IsaacLab 2.3.2 ManagerBasedRLEnv.step()
    #
    # The ONLY change is the insertion of self._compute_step_data() between
    # the counter update and termination/reward computation.
    # If upgrading IsaacLab, verify the parent step() hasn't changed.
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # post-step: update env counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # *** INSERTED: compute shared per-step data before MDP ***
        self._compute_step_data()

        # check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # reset envs that terminated/timed-out
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # update commands
        self.command_manager.compute(dt=self.step_dt)
        # step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # Per-step reward logging (mean across all envs)
        log = self.extras.get("log")
        if not isinstance(log, dict):
            log = {}
            self.extras["log"] = log
        rm = self.reward_manager
        for idx, name in enumerate(rm._term_names):
            log[f"Step_Reward/{name}"] = rm._step_reward[:, idx].mean()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ------------------------------------------------------------------
    # Per-step computation
    # ------------------------------------------------------------------

    def _compute_step_data(self):
        """Compute shared per-step values after physics, before MDP managers.

        This replaces the scattered ``ensure_updated()`` calls that were spread
        across 5 fake CommandTerms and 20+ reward/obs/termination functions.
        """
        robot = self.scene.articulations["robot"]
        contact_sensor = self.scene.sensors["contact_sensor"]

        # 1. Mode state transitions (legitimate command — updates one-hot encoding)
        self.command_manager.get_term("mode").ensure_updated()

        # 2. Map manager (BEV, height scan, staleness, exploration bonus, staleness peak)
        #    Runs before waypoint so patrol mode can read staleness_peak_world.
        self._map_manager.update_into(
            self._map_output,
            env_origins=self.scene.env_origins,
            robot_pos_w=robot.data.root_pos_w,
            robot_yaw_w=robot.data.heading_w.unsqueeze(-1),
            dt=self.step_dt,
        )

        # 3. Waypoint state machine (legitimate command — target reached, timeouts, patrol target)
        self.command_manager.get_term("waypoint").ensure_updated()

        # 4. Contact state (used by observations + undesired_contacts reward)
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > float(self.cfg.contact_threshold)
        self._is_contact[:] = is_contact.to(dtype=torch.float32)

        # 5. Base contact time (used by death_penalty reward + on_ground termination)
        self._base_contact_time[:] = contact_sensor.data.current_contact_time[
            :, self.robot_idx.contact_sensor_base_ids
        ].squeeze(-1)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        # Reset cached buffers (will be recomputed on next _compute_step_data)
        self._map_manager.reset(env_ids)
        self._map_output.nav_data[env_ids] = 0.0
        self._map_output.bev_data[env_ids] = 0.0
        self._map_output.height_data[env_ids] = 0.0
        self._map_output.far_staleness[env_ids] = 0.0
        self._map_output.exploration_bonus[env_ids] = 0.0
        self._map_output.staleness_peak_world[env_ids] = 0.0
        self._map_output.staleness_peak_value[env_ids] = 0.0
        self._is_contact[env_ids] = 0.0
        self._base_contact_time[env_ids] = 0.0

        # Parent reset: scene.reset → spawn event (writes spawn_pos_w) → command_manager.reset
        super()._reset_idx(env_ids)

        # Promote waypoint metrics to Episode_Info keys (without manager prefixes)
        log = self.extras.get("log")
        if not isinstance(log, dict):
            return

        waypoint_prefix = "Metrics/waypoint/"
        for key in ("Episode_Info/targets_reached_avg", "Episode_Info/targets_reached_max"):
            prefixed = waypoint_prefix + key
            if prefixed in log and key not in log:
                log[key] = log.pop(prefixed)
