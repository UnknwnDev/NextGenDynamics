# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantise height-map regions into discrete terraced steps."""

from __future__ import annotations

import numpy as np

from .custom_terrain_config import CustomTerrainCfg, TerracedZone


def apply_terracing(height_map: np.ndarray, cfg: CustomTerrainCfg) -> np.ndarray:
    """Terrace all configured zones on *height_map* (mutates in place).

    Combines manually specified ``cfg.terraced_zones`` with randomly generated
    zones (controlled by ``cfg.random_terraced_count``).
    """
    zones: list[TerracedZone] = []

    if cfg.terraced_zones is not None:
        zones.extend(cfg.terraced_zones)

    if cfg.random_terraced_count > 0:
        rng = np.random.default_rng(cfg.seed + 777)
        zones.extend(_generate_random_zones(cfg, rng))

    for zone in zones:
        _terrace_zone(height_map, cfg, zone)

    return height_map


def _generate_random_zones(cfg: CustomTerrainCfg, rng: np.random.Generator) -> list[TerracedZone]:
    """Create randomly placed terraced zones within the terrain margins."""
    x_min = -cfg.size[0] / 2 + cfg.margin
    x_max = cfg.size[0] / 2 - cfg.margin
    y_min = -cfg.size[1] / 2 + cfg.margin
    y_max = cfg.size[1] / 2 - cfg.margin

    zones: list[TerracedZone] = []
    for _ in range(cfg.random_terraced_count):
        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))
        s_lo, s_hi = cfg.random_terraced_size_range
        sx = float(rng.uniform(s_lo, s_hi))
        sy = float(rng.uniform(s_lo, s_hi))
        h_lo, h_hi = cfg.random_terraced_step_height_range
        sh = float(rng.uniform(h_lo, h_hi))
        heading = float(rng.uniform(0.0, 360.0))
        zones.append(TerracedZone(center=(cx, cy), size=(sx, sy), heading_deg=heading, step_height=sh))
    return zones


def _terrace_zone(height_map: np.ndarray, cfg: CustomTerrainCfg, zone: TerracedZone) -> None:
    """Quantise heights inside one rotated rectangular zone."""
    rows, cols = height_map.shape
    mpg = cfg.meter_per_grid

    half_x = zone.size[0] / 2.0
    half_y = zone.size[1] / 2.0

    # Pre-compute rotation.
    angle_rad = np.radians(zone.heading_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Four corners of the zone rectangle in world space.
    corners_local = np.array([
        [-half_x, -half_y],
        [-half_x, half_y],
        [half_x, -half_y],
        [half_x, half_y],
    ], dtype=np.float64)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners_world = corners_local @ rot.T + np.array(zone.center)

    # Axis-aligned bounding box in grid space.
    corners_gc = corners_world[:, 0] / mpg + 0.5 * (cols - 1)
    corners_gr = corners_world[:, 1] / mpg + 0.5 * (rows - 1)

    gc_min = max(0, int(np.floor(corners_gc.min())) - 1)
    gc_max = min(cols - 1, int(np.ceil(corners_gc.max())) + 1)
    gr_min = max(0, int(np.floor(corners_gr.min())) - 1)
    gr_max = min(rows - 1, int(np.ceil(corners_gr.max())) + 1)

    # Build grid of column/row indices within the AABB.
    col_idx = np.arange(gc_min, gc_max + 1, dtype=np.float64)
    row_idx = np.arange(gr_min, gr_max + 1, dtype=np.float64)
    grid_c, grid_r = np.meshgrid(col_idx, row_idx, indexing="xy")

    # Grid → world coordinates.
    world_x = (grid_c - 0.5 * (cols - 1)) * mpg
    world_y = (grid_r - 0.5 * (rows - 1)) * mpg

    # World → zone-local coordinates (inverse rotation).
    dx = world_x - zone.center[0]
    dy = world_y - zone.center[1]
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a

    # Mask: cells inside the zone rectangle.
    inside = (
        (local_x >= -half_x) & (local_x <= half_x)
        & (local_y >= -half_y) & (local_y <= half_y)
    )

    # Integer indices for the cells that need updating.
    r_int = grid_r.astype(np.intp)
    c_int = grid_c.astype(np.intp)
    mask_r = r_int[inside]
    mask_c = c_int[inside]

    # Quantise: floor(h / step) * step.
    original = height_map[mask_r, mask_c]
    quantised = np.floor(original / zone.step_height) * zone.step_height
    height_map[mask_r, mask_c] = quantised.astype(np.float32)
