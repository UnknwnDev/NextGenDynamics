# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..paths import CUSTOM_TERRAIN_USD_PATH


class ObstacleType:
    CUSTOM_MESH = "custom_mesh"
    CUBE = "cube"
    SPHERE = "sphere"


@dataclass
class Obstacle:
    type: str
    path: None | str = None
    scale_range: tuple[float, float] = (0.5, 2.0)
    num_instances: int = 100
    radius: float | None = None


@dataclass
class TerracedZone:
    """Define a rectangular region whose heights are quantized into discrete steps."""

    # Centre of the zone in world coordinates (metres).
    center: tuple[float, float] = (0.0, 0.0)
    # Zone extents (along heading, perpendicular) in metres.
    size: tuple[float, float] = (20.0, 20.0)
    # Counter-clockwise rotation in degrees (0 = +X axis).
    heading_deg: float = 0.0
    # Height quantisation interval (metres) – the "ledge" height.
    step_height: float = 0.15


@dataclass
class CustomTerrainCfg:
    """Configuration for custom terrain generation.

    This generator is responsible for creating the *actual* terrain USD that the environment loads.
    """

    # Terrain size in meters (x, y).
    size: tuple[float, float] = (196.0, 196.0)
    # Grid resolution in meters.
    meter_per_grid: float = 0.15
    # Output USD path.
    usd_path: Path = CUSTOM_TERRAIN_USD_PATH
    # Random seed.
    seed: int = 42

    # Height-map synthesis parameters.
    roughness: float = 0.1
    hill_scale: float = 768.0
    hill_height: float = 8.0 # 8.0
    hill_noise_lacunarity: float = 2.5
    hill_noise_persistence: float = 0.5
    hill_noise_octaves: int = 32

    # Obstacles (optional).
    obstacles: tuple[Obstacle, ...] | None = None

    # Terraced zones (optional, manual placement).
    terraced_zones: tuple[TerracedZone, ...] | None = None
    # Random terraced zone generation.
    random_terraced_count: int = 10
    random_terraced_size_range: tuple[float, float] = (10.0, 30.0)
    random_terraced_step_height_range: tuple[float, float] = (0.02, 0.10)

    # Spawn sampling.
    num_points: int = 1024
    sample_radius: float = 0.5
    flatness_threshold: float = 0.5
    max_attempts: int = 1024
    margin: float = 32.0
    # Include debug visualization points in exported USD.
    include_spawn_debug_points: bool = False

    def __post_init__(self):
        self.grid_size = (
            int(self.size[1] / self.meter_per_grid),
            int(self.size[0] / self.meter_per_grid),
        )

    def config_hash(self) -> str:
        """Return a SHA-256 hex digest of all generation-relevant parameters."""
        d = dataclasses.asdict(self)
        d.pop("usd_path", None)
        d.pop("grid_size", None)
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
