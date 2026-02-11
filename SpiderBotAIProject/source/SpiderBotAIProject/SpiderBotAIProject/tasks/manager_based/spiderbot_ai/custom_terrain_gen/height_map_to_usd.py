# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .custom_terrain_config import CustomTerrainCfg, ObstacleType


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


def _spawn_obstacles(stage: Usd.Stage, cfg: CustomTerrainCfg, placements: dict[str, dict[str, np.ndarray]] | None) -> None:
    if cfg.obstacles is None or not placements:
        return

    root = UsdGeom.Scope.Define(stage, "/World/obstacles")
    _ = root

    for obstacle in cfg.obstacles:
        data = placements.get(obstacle.type)
        if data is None:
            continue

        positions = data.get("positions")
        scales = data.get("scales")
        if positions is None or scales is None:
            continue

        for i in range(int(positions.shape[0])):
            prim_path = f"/World/obstacles/{obstacle.type}_{i:04d}"
            if obstacle.type == ObstacleType.CUBE:
                prim = UsdGeom.Cube.Define(stage, prim_path)
                prim.AddTranslateOp().Set(Gf.Vec3d(float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
                prim.AddScaleOp().Set(Gf.Vec3f(float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2])))
                UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            elif obstacle.type == ObstacleType.SPHERE:
                prim = UsdGeom.Sphere.Define(stage, prim_path)
                prim.AddTranslateOp().Set(Gf.Vec3d(float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
                # Sphere uses radius; take max XY as radius scale.
                prim.GetRadiusAttr().Set(float(0.5 * max(scales[i, 0], scales[i, 1])))
                UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            elif obstacle.type == ObstacleType.CUSTOM_MESH:
                raise NotImplementedError("CUSTOM_MESH obstacles are not supported in SpiderBotAIProject terrain USD yet.")
            else:
                raise ValueError(f"Unknown obstacle type: {obstacle.type}")


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

    _spawn_obstacles(stage, cfg, obstacle_placement)
    _spawn_spawn_points(stage, spawn_points)

    stage.GetRootLayer().Save()
    return usd_path
