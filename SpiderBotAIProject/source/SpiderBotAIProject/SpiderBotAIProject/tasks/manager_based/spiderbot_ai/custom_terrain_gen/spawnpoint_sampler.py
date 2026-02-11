# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from .custom_terrain_config import CustomTerrainCfg
from .height_sampling import sample_height_np
from .obstacles import obstacle_radii


def spawn_point_sampler(
    height_map: np.ndarray, obstacle_placement: dict[str, dict[str, np.ndarray]] | None, cfg: CustomTerrainCfg
) -> np.ndarray:
    """Sample valid spawn points across the terrain (N, 3)."""
    rng = np.random.default_rng(cfg.seed)

    obstacle_circles: list[list[float]] = []
    if obstacle_placement and cfg.obstacles:
        obs_cfg_map = {obs.type: obs for obs in cfg.obstacles}
        for obs_type, data in obstacle_placement.items():
            positions = data.get("positions")
            scales = data.get("scales")
            if positions is None or scales is None:
                continue
            base_radius = obs_cfg_map.get(obs_type).radius if obs_type in obs_cfg_map else None
            radii = obstacle_radii(obs_type, scales, base_radius)
            for i in range(len(positions)):
                obstacle_circles.append([positions[i, 0], positions[i, 1], float(radii[i])])

    obs_data = np.asarray(obstacle_circles, dtype=np.float32)  # (M, 3)

    valid_points: list[list[float]] = []

    x_min = -cfg.size[0] / 2 + cfg.margin
    x_max = cfg.size[0] / 2 - cfg.margin
    y_min = -cfg.size[1] / 2 + cfg.margin
    y_max = cfg.size[1] / 2 - cfg.margin

    rows, cols = height_map.shape

    for _ in range(int(cfg.num_points)):
        found = False
        for _attempt in range(int(cfg.max_attempts)):
            rx = float(rng.uniform(x_min, x_max))
            ry = float(rng.uniform(y_min, y_max))

            if obs_data.size:
                dists = np.linalg.norm(obs_data[:, :2] - np.array([rx, ry], dtype=np.float32), axis=1)
                min_dists = obs_data[:, 2] + cfg.sample_radius + 0.2
                if np.any(dists < min_dists):
                    continue

            grid_x = int((rx + cfg.size[0] / 2) / cfg.meter_per_grid)
            grid_y = int((ry + cfg.size[1] / 2) / cfg.meter_per_grid)
            grid_r = int(np.ceil(cfg.sample_radius / cfg.meter_per_grid))

            gx_min = max(0, grid_x - grid_r)
            gx_max = min(cols, grid_x + grid_r + 1)
            gy_min = max(0, grid_y - grid_r)
            gy_max = min(rows, grid_y + grid_r + 1)

            patch = height_map[gy_min:gy_max, gx_min:gx_max]
            if patch.size == 0:
                continue

            if (float(np.max(patch)) - float(np.min(patch))) > float(cfg.flatness_threshold):
                continue

            rz = float(sample_height_np(height_map, cfg.meter_per_grid, np.array([[rx, ry]], dtype=np.float32))[0])
            valid_points.append([rx, ry, rz + 0.2])
            found = True
            break

        if not found:
            valid_points.append([0.0, 0.0, 1.0])

    return np.asarray(valid_points, dtype=np.float32)

