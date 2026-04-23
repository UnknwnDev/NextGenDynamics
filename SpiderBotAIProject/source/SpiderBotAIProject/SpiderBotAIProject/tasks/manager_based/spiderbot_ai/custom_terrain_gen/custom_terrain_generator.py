# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np

from .custom_terrain_config import CustomTerrainCfg
from .height_map_generator import generate_hills, generate_roughness
from .height_map_to_usd import save_height_map_to_usd
from .obstacles import mesh_placer
from .spawnpoint_sampler import spawn_point_sampler
from .terracing import apply_terracing


class CustomTerrainGenerator:
    """Generate terrain artifacts (height-map + spawn points + optional obstacles) and export a USD."""

    def __init__(self, cfg: CustomTerrainCfg):
        self.cfg = cfg
        self.height_map: np.ndarray | None = None
        self.obstacle_placement: dict[str, dict[str, np.ndarray]] | None = None
        self.spawn_points: np.ndarray | None = None

    def initialize(self, *, export_usd: bool = True, force_export: bool = False) -> Path:
        # 1. Hills only  2. Terrace  3. Roughness on top
        self.height_map = generate_hills(self.cfg)
        apply_terracing(self.height_map, self.cfg)
        self.height_map += generate_roughness(self.cfg)

        self.obstacle_placement = mesh_placer(self.cfg, self.height_map)
        self.spawn_points = spawn_point_sampler(self.height_map, self.obstacle_placement, self.cfg)

        usd_path = Path(self.cfg.usd_path)
        if export_usd and (force_export or not usd_path.exists()):
            save_height_map_to_usd(self.height_map, self.cfg, self.obstacle_placement, spawn_points=self.spawn_points)
        return usd_path
