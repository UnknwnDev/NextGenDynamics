# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pxr import Usd, UsdGeom

from ..custom_terrain_gen.custom_terrain_config import CustomTerrainCfg
from ..custom_terrain_gen.height_sampling import sample_height_torch
from ..custom_terrain_gen.obstacles import compute_obstacle_circles, mesh_placer
from ..custom_terrain_gen.spawnpoint_sampler import spawn_point_sampler


class TerrainData:
    """Static terrain data provider (height map, spawn points, obstacle collision).

    Loaded once at environment init time. Replaces TerrainCommandTerm which
    was abusing the CommandTerm pattern for static data storage.
    """

    def __init__(self, cfg, device: torch.device | str):
        seed = int(getattr(cfg, "seed", 42))
        terrain_cfg = CustomTerrainCfg(
            size=(float(cfg.height_map_size_x), float(cfg.height_map_size_y)),
            meter_per_grid=float(cfg.height_map_meter_per_grid),
            seed=seed,
        )

        usd_path = Path(terrain_cfg.usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Missing terrain USD: {usd_path}. "
                "Run `scripts/skrlcustom/train.py` or `scripts/skrlcustom/play.py` once to generate it."
            )

        stage = Usd.Stage.Open(str(usd_path))
        mesh = UsdGeom.Mesh.Get(stage, "/World/terrain")
        if mesh is None or not mesh.GetPrim().IsValid():
            raise RuntimeError(f"Terrain USD is missing prim '/World/terrain': {usd_path}")

        points = mesh.GetPointsAttr().Get()
        points_np = np.array(points, dtype=np.float32, copy=True)
        rows, cols = terrain_cfg.grid_size
        if points_np.shape[0] != rows * cols:
            raise RuntimeError(
                f"Terrain mesh points count mismatch. Expected {rows * cols}, got {points_np.shape[0]} ({usd_path})."
            )

        height_map_np = points_np[:, 2].reshape(rows, cols)
        self.height_map = torch.from_numpy(height_map_np).to(device=device, dtype=torch.float32)

        self.device = device
        self.meter_per_grid = float(terrain_cfg.meter_per_grid)
        self.size_x = float(terrain_cfg.size[0])
        self.size_y = float(terrain_cfg.size[1])
        self.origin_xy = torch.tensor([0.0, 0.0], device=device)

        # Recompute obstacle placement (deterministic: same seed + terrain size)
        obstacle_placement = mesh_placer(terrain_cfg, height_map_np)
        obstacle_circles_np = compute_obstacle_circles(obstacle_placement, terrain_cfg)

        spawn_points_np = None
        spawn_prim = UsdGeom.Points.Get(stage, "/World/debug/spawn_points")
        if spawn_prim is not None and spawn_prim.GetPrim().IsValid():
            spawn_points = spawn_prim.GetPointsAttr().Get()
            spawn_points_np = np.array(spawn_points, dtype=np.float32, copy=True)

        if spawn_points_np is None or spawn_points_np.size == 0:
            spawn_points_np = spawn_point_sampler(height_map_np, obstacle_placement=obstacle_placement, cfg=terrain_cfg)

        self.spawn_points = torch.from_numpy(spawn_points_np).to(device=device, dtype=torch.float32)
        self.obstacle_circles = torch.from_numpy(obstacle_circles_np).to(device=device, dtype=torch.float32)

    def height_at_xy(self, xy_w: torch.Tensor) -> torch.Tensor:
        if xy_w.shape[-1] != 2:
            xy_w = xy_w[..., :2]
        return sample_height_torch(self.height_map, self.meter_per_grid, xy_w)

    def collides(self, xy_w: torch.Tensor, margin: float) -> torch.Tensor:
        if xy_w.shape[-1] != 2:
            xy_w = xy_w[..., :2]
        if self.obstacle_circles.numel() == 0:
            return torch.zeros(xy_w.shape[0], device=self.device, dtype=torch.bool)

        obs_xy = self.obstacle_circles[:, :2]
        obs_r = self.obstacle_circles[:, 2]
        diff = xy_w[:, None, :] - obs_xy[None, :, :]
        dist2 = torch.sum(diff * diff, dim=-1)
        thresh2 = torch.square(obs_r[None, :] + float(margin))
        return torch.any(dist2 < thresh2, dim=1)

    def sample_spawn(self, env_origins: torch.Tensor, patrol_size: float) -> torch.Tensor:
        """Sample spawn XY positions near environment origins.

        Returns:
            (N, 2) XY world positions (caller adds Z).
        """
        n = env_origins.shape[0]
        env_origins_xy = env_origins[:, :2]
        patrol_r = patrol_size / 2.0
        spawn_xy = self.spawn_points[:, :2]
        out_xy = torch.zeros(n, 2, device=self.device, dtype=torch.float32)

        for i in range(n):
            center = env_origins_xy[i]
            d = torch.norm(spawn_xy - center, dim=1)
            candidates = (d < patrol_r).nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                out_xy[i] = center
            else:
                idx = candidates[torch.randint(0, candidates.numel(), (1,), device=self.device)]
                out_xy[i] = spawn_xy[idx].squeeze(0)

        return out_xy

    def sample_target(self, anchor_pos_w: torch.Tensor, cfg) -> torch.Tensor:
        """Sample obstacle-avoiding target positions around anchors.

        Returns:
            (N, 3) target positions with Z at terrain height + offset.
        """
        n = anchor_pos_w.shape[0]
        anchor_xy = anchor_pos_w[:, :2]

        x_min = -self.size_x / 2.0 + float(cfg.spawn_padding)
        x_max = self.size_x / 2.0 - float(cfg.spawn_padding)
        y_min = -self.size_y / 2.0 + float(cfg.spawn_padding)
        y_max = self.size_y / 2.0 - float(cfg.spawn_padding)

        out_xy = anchor_xy.clone()
        valid = torch.zeros(n, device=self.device, dtype=torch.bool)

        attempts = int(cfg.target_sample_attempts)
        for _ in range(attempts):
            remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if remaining.numel() == 0:
                break

            m = remaining.numel()
            r = float(cfg.point_max_distance) + (
                float(cfg.point_min_distance) - float(cfg.point_max_distance)
            ) * torch.rand(m, device=self.device)
            a = 2.0 * torch.pi * torch.rand(m, device=self.device)
            cand_xy = anchor_xy[remaining] + torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)

            in_bounds = (
                (cand_xy[:, 0] >= x_min)
                & (cand_xy[:, 0] <= x_max)
                & (cand_xy[:, 1] >= y_min)
                & (cand_xy[:, 1] <= y_max)
            )
            not_collide = ~self.collides(cand_xy, margin=float(cfg.target_obstacle_margin))
            ok = in_bounds & not_collide

            ok_ids = remaining[ok]
            out_xy[ok_ids] = cand_xy[ok]
            valid[ok_ids] = True

        z = self.height_at_xy(out_xy) + float(cfg.target_z_offset)
        return torch.cat([out_xy, z.unsqueeze(1)], dim=1)
