# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from .custom_terrain_config import CustomTerrainCfg, ObstacleType
from .height_sampling import sample_height_np


def euler_to_rotation_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """Convert (3,) XYZ euler angles in degrees to a (3,3) rotation matrix.

    Applies Rx(x) @ Ry(y) @ Rz(z).
    """
    rad = np.deg2rad(euler_deg)
    cx, cy, cz = np.cos(rad)
    sx, sy, sz = np.sin(rad)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


# Unit cube vertices (centred at origin, side length 1).
_CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)


def obstacle_radii(
    obstacle_type: str,
    scales: np.ndarray,
    rotations: np.ndarray | None = None,
    base_radius: float | None = None,
) -> np.ndarray:
    """Compute XY bounding radius per obstacle, accounting for rotation.

    For cubes the 8 corners are scaled and rotated, then the maximum XY
    distance from the origin gives the bounding radius.  For spheres
    rotation is irrelevant so the radius is simply ``max(sx, sy) / 2``.
    """
    n = scales.shape[0]

    if obstacle_type == ObstacleType.SPHERE:
        if rotations is None:
            # Axis-aligned ellipsoid: XY radius = max semi-axis in XY
            return 0.5 * np.max(scales[:, :2], axis=1)
        # Rotated ellipsoid: conservative bound = half of largest scale axis
        # (bounding sphere of the ellipsoid projected onto XY)
        radii = np.empty(n, dtype=np.float64)
        for i in range(n):
            semi = 0.5 * scales[i]  # semi-axes (a, b, c)
            R = euler_to_rotation_matrix(rotations[i])
            # Max XY extent: ||R[:2,:] @ diag(semi)||_op = sqrt(max eig of M @ M^T)
            M = R[:2, :] * semi  # (2, 3) broadcast
            MtM = M @ M.T  # (2, 2)
            radii[i] = np.sqrt(np.max(np.linalg.eigvalsh(MtM)))
        return radii.astype(np.float32)

    if obstacle_type == ObstacleType.CUBE:
        if rotations is None:
            return 0.5 * np.max(scales[:, :2], axis=1)
        radii = np.empty(n, dtype=np.float64)
        for i in range(n):
            corners = _CUBE_VERTS * scales[i]
            R = euler_to_rotation_matrix(rotations[i])
            rotated = corners @ R.T
            radii[i] = np.max(np.linalg.norm(rotated[:, :2], axis=1))
        return radii.astype(np.float32)

    if obstacle_type == ObstacleType.CUSTOM_MESH:
        if base_radius is None:
            base_radius = 1.0
        return base_radius * np.max(scales[:, :2], axis=1)

    return 0.5 * np.max(scales[:, :2], axis=1)


def _parse_range_3d(range_spec: tuple, rng: np.random.Generator, num: int) -> np.ndarray:
    """Parse a range into (num, 3) per-axis random values.

    Accepts either ``(min, max)`` for uniform range across all axes, or
    ``((min_x, max_x), (min_y, max_y), (min_z, max_z))`` for per-axis ranges.
    """
    if isinstance(range_spec[0], (tuple, list)):
        vx = rng.uniform(range_spec[0][0], range_spec[0][1], size=num)
        vy = rng.uniform(range_spec[1][0], range_spec[1][1], size=num)
        vz = rng.uniform(range_spec[2][0], range_spec[2][1], size=num)
    else:
        lo, hi = float(range_spec[0]), float(range_spec[1])
        vx = rng.uniform(lo, hi, size=num)
        vy = rng.uniform(lo, hi, size=num)
        vz = rng.uniform(lo, hi, size=num)
    return np.column_stack([vx, vy, vz]).astype(np.float32)


def compute_obstacle_circles(
    placements: dict[str, dict[str, np.ndarray]],
    cfg: CustomTerrainCfg,
) -> np.ndarray:
    """Convert obstacle placement dict to (M, 3) array of [x, y, radius] circles."""
    if not placements or cfg.obstacles is None:
        return np.zeros((0, 3), dtype=np.float32)

    obs_cfg_map = {obs.type: obs for obs in cfg.obstacles}
    circles: list[list[float]] = []
    for obs_type, data in placements.items():
        positions = data.get("positions")
        scales = data.get("scales")
        if positions is None or scales is None:
            continue
        rotations = data.get("rotations")
        base_radius = obs_cfg_map[obs_type].radius if obs_type in obs_cfg_map else None
        radii = obstacle_radii(obs_type, scales, rotations, base_radius)
        for i in range(len(positions)):
            circles.append([positions[i, 0], positions[i, 1], float(radii[i])])
    return np.asarray(circles, dtype=np.float32) if circles else np.zeros((0, 3), dtype=np.float32)


def mesh_placer(cfg: CustomTerrainCfg, height_map: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Randomly sample obstacle placements (positions, scales, rotations)."""
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

        scales = _parse_range_3d(obstacle.scale_range, rng, num)
        rotations = _parse_range_3d(obstacle.rotation_range, rng, num)

        placements[obstacle.type] = {
            "positions": coords_3d,
            "scales": scales,
            "rotations": rotations,
        }

    return placements
