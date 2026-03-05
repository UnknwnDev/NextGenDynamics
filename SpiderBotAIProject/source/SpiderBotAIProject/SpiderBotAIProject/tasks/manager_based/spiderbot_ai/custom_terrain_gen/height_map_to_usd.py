# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .custom_terrain_config import CustomTerrainCfg, ObstacleType
from .obstacles import euler_to_rotation_matrix


# ---------------------------------------------------------------------------
# Tessellation helpers
# ---------------------------------------------------------------------------

def _make_cube_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit cube [-0.5, 0.5]³ as a triangle mesh.

    Returns (verts (8,3), face_vertex_indices (36,), face_vertex_counts (12,)).
    """
    verts = np.array(
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
    # 6 faces × 2 triangles = 12 triangles (CCW winding when viewed from outside)
    tris = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # -Z face
            [4, 5, 6], [4, 6, 7],  # +Z face
            [0, 1, 5], [0, 5, 4],  # -Y face
            [2, 3, 7], [2, 7, 6],  # +Y face
            [0, 4, 7], [0, 7, 3],  # -X face
            [1, 2, 6], [1, 6, 5],  # +X face
        ],
        dtype=np.int32,
    )
    counts = np.full(12, 3, dtype=np.int32)
    return verts, tris.ravel(), counts


def _make_sphere_mesh(
    radius: float = 1.0, segments: int = 16, rings: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV sphere as a triangle mesh.

    Returns (verts, face_vertex_indices, face_vertex_counts).
    """
    verts = []
    # Top pole
    verts.append([0.0, 0.0, radius])
    # Intermediate rings
    for i in range(1, rings):
        phi = np.pi * i / rings
        sp, cp = np.sin(phi), np.cos(phi)
        for j in range(segments):
            theta = 2.0 * np.pi * j / segments
            verts.append([radius * sp * np.cos(theta), radius * sp * np.sin(theta), radius * cp])
    # Bottom pole
    verts.append([0.0, 0.0, -radius])
    verts = np.array(verts, dtype=np.float64)

    tris = []
    # Top cap
    for j in range(segments):
        j_next = (j + 1) % segments
        tris.append([0, 1 + j, 1 + j_next])
    # Middle quads (split into 2 tris each)
    for i in range(rings - 2):
        row0 = 1 + i * segments
        row1 = 1 + (i + 1) * segments
        for j in range(segments):
            j_next = (j + 1) % segments
            tris.append([row0 + j, row1 + j, row1 + j_next])
            tris.append([row0 + j, row1 + j_next, row0 + j_next])
    # Bottom cap
    bottom = len(verts) - 1
    last_row = 1 + (rings - 2) * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        tris.append([bottom, last_row + j_next, last_row + j])

    tris = np.array(tris, dtype=np.int32)
    counts = np.full(len(tris), 3, dtype=np.int32)
    return verts, tris.ravel(), counts


# ---------------------------------------------------------------------------
# Height-map mesh builder (unchanged)
# ---------------------------------------------------------------------------

def _build_height_mesh(height_map: np.ndarray, cfg: CustomTerrainCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = height_map.shape

    x_length = (cols - 1) * cfg.meter_per_grid
    y_length = (rows - 1) * cfg.meter_per_grid
    x = np.linspace(-x_length / 2, x_length / 2, cols, dtype=np.float32)
    y = np.linspace(-y_length / 2, y_length / 2, rows, dtype=np.float32)
    xv, yv = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xv.ravel(), yv.ravel(), height_map.ravel())).astype(np.float32)

    row_idx = np.arange(rows - 1, dtype=np.int32)
    col_idx = np.arange(cols - 1, dtype=np.int32)
    cv, rv = np.meshgrid(col_idx, row_idx, indexing="xy")
    p_idxs = rv * cols + cv

    t1 = np.column_stack((p_idxs.ravel(), p_idxs.ravel() + 1, p_idxs.ravel() + cols)).astype(np.int32)
    t2 = np.column_stack((p_idxs.ravel() + 1, p_idxs.ravel() + cols + 1, p_idxs.ravel() + cols)).astype(np.int32)

    face_vertex_indices = np.vstack((t1, t2)).reshape(-1).astype(np.int32)
    face_vertex_counts = np.full(len(face_vertex_indices) // 3, 3, dtype=np.int32)
    return points, face_vertex_indices, face_vertex_counts


def _spawn_spawn_points(stage: Usd.Stage, spawn_points: np.ndarray | None) -> None:
    if spawn_points is None:
        return

    points_prim = UsdGeom.Points.Define(stage, "/World/debug/spawn_points")
    points_np = np.asarray(spawn_points, dtype=np.float32)
    points_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points_np))

    widths = np.full(points_np.shape[0], 0.2, dtype=np.float32)
    points_prim.GetWidthsAttr().Set(Vt.FloatArray(widths.tolist()))
    points_prim.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])


# ---------------------------------------------------------------------------
# Obstacle mesh builder — merged into /World/terrain
# ---------------------------------------------------------------------------

def _build_obstacle_meshes(
    cfg: CustomTerrainCfg,
    placements: dict[str, dict[str, np.ndarray]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build merged obstacle mesh data (points, face_vertex_indices, face_vertex_counts).

    Returns None if there are no obstacles.  The caller is responsible for
    offsetting ``face_vertex_indices`` by the terrain vertex count before
    concatenation.
    """
    if cfg.obstacles is None or not placements:
        return None

    all_points: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    vert_offset = 0

    for obstacle in cfg.obstacles:
        data = placements.get(obstacle.type)
        if data is None:
            continue

        positions = data.get("positions")
        scales = data.get("scales")
        if positions is None or scales is None:
            continue
        rotations = data.get("rotations")

        # Pre-generate base mesh once per obstacle type
        if obstacle.type == ObstacleType.CUBE:
            base_verts, base_indices, base_counts = _make_cube_mesh()
        elif obstacle.type == ObstacleType.SPHERE:
            base_verts, base_indices, base_counts = _make_sphere_mesh(radius=1.0)
        elif obstacle.type == ObstacleType.CUSTOM_MESH:
            raise NotImplementedError("CUSTOM_MESH obstacles are not supported yet.")
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle.type}")

        for i in range(int(positions.shape[0])):
            # Start from base mesh
            if obstacle.type == ObstacleType.SPHERE:
                r = 0.5 * max(float(scales[i, 0]), float(scales[i, 1]))
                verts = base_verts * r  # uniform radius scale
            else:
                verts = base_verts * scales[i]  # per-axis scale

            # Rotate (XYZ euler)
            if rotations is not None:
                R = euler_to_rotation_matrix(rotations[i])
                verts = verts @ R.T

            # Translate
            verts = verts + positions[i]

            all_points.append(verts.astype(np.float32))
            all_indices.append(base_indices + vert_offset)
            all_counts.append(base_counts)
            vert_offset += len(verts)

    if not all_points:
        return None

    return (
        np.concatenate(all_points),
        np.concatenate(all_indices),
        np.concatenate(all_counts),
    )


def save_height_map_to_usd(
    height_map: np.ndarray,
    cfg: CustomTerrainCfg,
    obstacle_placement: dict[str, dict[str, np.ndarray]] | None = None,
    spawn_points: np.ndarray | None = None,
) -> Path:
    """Save height-map (and optional obstacles) to a USD file at cfg.usd_path."""
    usd_path = Path(cfg.usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)

    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/terrain")

    points, face_vertex_indices, face_vertex_counts = _build_height_mesh(height_map, cfg)

    # Merge obstacle meshes into the terrain mesh so RayCaster sees everything.
    obs = _build_obstacle_meshes(cfg, obstacle_placement)
    if obs is not None:
        obs_pts, obs_idx, obs_cnt = obs
        obs_idx = obs_idx + len(points)  # offset indices past terrain verts
        points = np.concatenate([points, obs_pts])
        face_vertex_indices = np.concatenate([face_vertex_indices, obs_idx])
        face_vertex_counts = np.concatenate([face_vertex_counts, obs_cnt])

    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_vertex_counts))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_vertex_indices))
    mesh_prim.GetExtentAttr().Set(mesh_prim.ComputeExtent(mesh_prim.GetPointsAttr().Get()))

    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api.CreateApproximationAttr().Set("none")

    mat_path = "/World/material/terrain"
    material = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(0.5)
    phys_mat.CreateDynamicFrictionAttr().Set(0.5)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(material)

    mesh_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.06, 0.025)])

    if cfg.include_spawn_debug_points:
        _spawn_spawn_points(stage, spawn_points)

    stage.GetRootLayer().Save()
    return usd_path
