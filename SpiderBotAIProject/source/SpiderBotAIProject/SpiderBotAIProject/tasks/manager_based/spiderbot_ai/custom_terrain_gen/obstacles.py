# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from .custom_terrain_config import CustomTerrainCfg, ObstacleType
from .height_sampling import sample_height_np


def obstacle_radii(obstacle_type: str, scales: np.ndarray, base_radius: float | None = None) -> np.ndarray:
    """Approximate obstacle radius from type and (N, 3) scales."""
    s_xy = np.max(scales[:, :2], axis=1)
    if obstacle_type == ObstacleType.CUBE:
        return 0.5 * s_xy
    if obstacle_type == ObstacleType.SPHERE:
        return 0.5 * s_xy
    if obstacle_type == ObstacleType.CUSTOM_MESH:
        if base_radius is None:
            base_radius = 1.0
        return base_radius * s_xy
    return 0.5 * s_xy


def mesh_placer(cfg: CustomTerrainCfg, height_map: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Randomly sample obstacle placements (positions + scales)."""
    if cfg.obstacles is None:
        return {}

    rng = np.random.default_rng(cfg.seed)

    x_range = (-cfg.size[0] / 2, cfg.size[0] / 2)
    y_range = (-cfg.size[1] / 2, cfg.size[1] / 2)

    placements: dict[str, dict[str, np.ndarray]] = {}
    for obstacle in cfg.obstacles:
        num = int(obstacle.num_instances)
        coords = np.zeros((num, 2), dtype=np.float32)
        coords[:, 0] = rng.uniform(x_range[0], x_range[1], size=num)
        coords[:, 1] = rng.uniform(y_range[0], y_range[1], size=num)

        z = sample_height_np(height_map, cfg.meter_per_grid, coords)
        coords_3d = np.concatenate([coords, z.reshape(-1, 1)], axis=1).astype(np.float32)

        scales = rng.uniform(obstacle.scale_range[0], obstacle.scale_range[1], size=(num, 3)).astype(np.float32)
        placements[obstacle.type] = {"positions": coords_3d, "scales": scales}

    return placements

