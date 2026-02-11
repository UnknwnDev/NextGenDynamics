# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch


def sample_height_np(height_map: np.ndarray, meter_per_grid: float, coords_xy: np.ndarray) -> np.ndarray:
    """Bilinear height sampling for a 2D height-map centered at (0, 0)."""
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError(f"coords_xy must have shape (N, 2). Received: {coords_xy.shape}")

    rows, cols = height_map.shape
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    grid_x = x / meter_per_grid + 0.5 * (cols - 1)
    grid_y = y / meter_per_grid + 0.5 * (rows - 1)

    x0 = np.floor(grid_x).astype(np.int64)
    y0 = np.floor(grid_y).astype(np.int64)
    x0 = np.clip(x0, 0, cols - 1)
    y0 = np.clip(y0, 0, rows - 1)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)

    sx = np.clip(grid_x - x0, 0.0, 1.0)
    sy = np.clip(grid_y - y0, 0.0, 1.0)

    h00 = height_map[y0, x0]
    h10 = height_map[y0, x1]
    h01 = height_map[y1, x0]
    h11 = height_map[y1, x1]

    h0 = h00 * (1.0 - sx) + h10 * sx
    h1 = h01 * (1.0 - sx) + h11 * sx
    return h0 * (1.0 - sy) + h1 * sy


def sample_height_torch(height_map: torch.Tensor, meter_per_grid: float, coords_xy: torch.Tensor) -> torch.Tensor:
    """Bilinear height sampling for a 2D height-map centered at (0, 0)."""
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError(f"coords_xy must have shape (N, 2). Received: {tuple(coords_xy.shape)}")

    rows, cols = int(height_map.shape[0]), int(height_map.shape[1])
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    grid_x = x / float(meter_per_grid) + 0.5 * float(cols - 1)
    grid_y = y / float(meter_per_grid) + 0.5 * float(rows - 1)

    x0 = torch.floor(grid_x).to(dtype=torch.int64).clamp(0, cols - 1)
    y0 = torch.floor(grid_y).to(dtype=torch.int64).clamp(0, rows - 1)
    x1 = (x0 + 1).clamp(0, cols - 1)
    y1 = (y0 + 1).clamp(0, rows - 1)

    sx = (grid_x - x0.to(dtype=grid_x.dtype)).clamp(0.0, 1.0)
    sy = (grid_y - y0.to(dtype=grid_y.dtype)).clamp(0.0, 1.0)

    h00 = height_map[y0, x0]
    h10 = height_map[y0, x1]
    h01 = height_map[y1, x0]
    h11 = height_map[y1, x1]

    h0 = h00 * (1.0 - sx) + h10 * sx
    h1 = h01 * (1.0 - sx) + h11 * sx
    return h0 * (1.0 - sy) + h1 * sy
