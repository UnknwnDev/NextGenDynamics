# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from .custom_terrain_config import CustomTerrainCfg
import opensimplex

def generate_hills(cfg: CustomTerrainCfg) -> np.ndarray:
    """Generate the hill layer using OpenSimplex-based fractal noise (no roughness)."""

    opensimplex.seed(int(cfg.seed))
    hills = np.zeros(cfg.grid_size, dtype=np.float32)

    xs = np.arange(cfg.grid_size[1], dtype=np.float32)
    ys = np.arange(cfg.grid_size[0], dtype=np.float32)

    frequency = 1.0 / float(cfg.hill_scale)
    amplitude = 1.0
    for _ in range(int(cfg.hill_noise_octaves)):
        nx = xs * frequency
        ny = ys * frequency
        noise = opensimplex.noise2array(nx, ny).astype(np.float32)
        hills += noise * amplitude

        frequency *= float(cfg.hill_noise_lacunarity)
        amplitude *= float(cfg.hill_noise_persistence)

    hills *= float(cfg.hill_height)
    return hills


def generate_roughness(cfg: CustomTerrainCfg) -> np.ndarray:
    """Generate a small random roughness layer."""

    rng = np.random.default_rng(cfg.seed)
    return rng.random(cfg.grid_size, dtype=np.float32) * float(cfg.roughness)


def generate_height_map(cfg: CustomTerrainCfg) -> np.ndarray:
    """Generate a deterministic height-map using OpenSimplex-based fractal noise."""

    return generate_hills(cfg) + generate_roughness(cfg)

