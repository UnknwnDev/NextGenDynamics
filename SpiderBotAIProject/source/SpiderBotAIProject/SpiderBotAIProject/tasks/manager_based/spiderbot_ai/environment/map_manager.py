"""
MapManager: Manages exploration/staleness maps and generates nav_data for patrol mode.

Ported from the direct env implementation and adapted for manager-based usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.sensors import RayCaster

from ..utils.cloudpoint_to_bev import BEVWorkspaceBuilder, build_bev_inplace, transform_world_to_ego_inplace


@dataclass
class MapManagerOutput:
    """Return type for MapManager.update()."""

    nav_data: torch.Tensor  # (N, 1, nav_dim, nav_dim)
    bev_data: torch.Tensor  # (N, 3, 64, 64)
    height_data: torch.Tensor  # (N, 64, 64)
    far_staleness: torch.Tensor  # (N, 8)
    exploration_bonus: torch.Tensor  # (N,)
    staleness_peak_world: torch.Tensor  # (N, 2) — world XY of highest-staleness region
    staleness_peak_value: torch.Tensor  # (N,) — staleness value at peak


@dataclass
class MapManagerWorkspace:
    """Reusable tensors for staleness/nav updates."""

    flat_cell_idx: torch.Tensor
    valid_mask: torch.Tensor
    valid_mask_tmp: torch.Tensor
    hit_map_flat: torch.Tensor
    inverse_hit_map: torch.Tensor
    exploration_bonus: torch.Tensor
    far_out: torch.Tensor
    far_counts: torch.Tensor
    far_ones: torch.Tensor
    theta: torch.Tensor
    batch_offsets: torch.Tensor
    base_gx: torch.Tensor
    base_gy: torch.Tensor
    rel: torch.Tensor
    col_idx: torch.Tensor


class MapManager:
    def __init__(self, config, num_envs, device, height_scanner: RayCaster, lidar_sensor: RayCaster):
        self.device = device
        self.num_envs = num_envs
        self.config = config

        self._height_scanner = height_scanner
        self._lidar_sensor = lidar_sensor

        x = torch.linspace(-1, 1, self.config.staleness_dim, device=device, dtype=torch.float32)
        y = torch.linspace(-1, 1, self.config.staleness_dim, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(grid_x**2 + grid_y**2)

        fade_width = 0.15
        self.patrol_mask = torch.clamp((1.0 - dist) / fade_width, min=0.0, max=1.0)
        self.patrol_mask = self.patrol_mask.view(1, 1, self.config.staleness_dim, self.config.staleness_dim)

        self.staleness_maps = torch.zeros(
            self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim, device=device
        )

        lidar_dtype = self._lidar_sensor.data.ray_hits_w.dtype
        self._num_lidar_points = int(self._lidar_sensor.data.ray_hits_w.shape[1])
        self._workspace = self._create_workspace(self._num_lidar_points)
        self._bev_builder = BEVWorkspaceBuilder(
            batch_size=self.num_envs,
            num_points=self._num_lidar_points,
            device=self.device,
            dtype=lidar_dtype,
            channels=("max_height", "mean_height", "density"),
        )
        # Only allocate a default output if callers use the legacy `update()` API.
        # Training uses `update_into` via MapCommandTerm, so eager allocation is wasted VRAM.
        self._default_output: MapManagerOutput | None = None

    def update(self, env_origins, robot_pos_w, robot_yaw_w, dt) -> MapManagerOutput:
        if self._default_output is None:
            self._default_output = MapManagerOutput(
                nav_data=torch.zeros(self.num_envs, 1, self.config.nav_dim, self.config.nav_dim, device=self.device),
                bev_data=torch.zeros(self.num_envs, 3, 64, 64, device=self.device),
                height_data=torch.zeros(self.num_envs, 64, 64, device=self.device),
                far_staleness=torch.zeros(self.num_envs, 8, device=self.device),
                exploration_bonus=torch.zeros(self.num_envs, device=self.device),
                staleness_peak_world=torch.zeros(self.num_envs, 2, device=self.device),
                staleness_peak_value=torch.zeros(self.num_envs, device=self.device),
            )

        self.update_into(self._default_output, env_origins, robot_pos_w, robot_yaw_w, dt)
        return self._default_output

    def update_into(self, output: MapManagerOutput, env_origins, robot_pos_w, robot_yaw_w, dt) -> None:
        lidar_hits_w = self._lidar_sensor.data.ray_hits_w
        self._ensure_workspace(int(lidar_hits_w.shape[1]), lidar_hits_w.dtype)

        self._update_staleness_map(lidar_hits_w, env_origins, dt, output.exploration_bonus)
        self._get_far_staleness(robot_pos_w, robot_yaw_w, env_origins, output.far_staleness)
        self._sample_egocentric_maps(robot_pos_w, robot_yaw_w, env_origins, output.nav_data)
        self._compute_height_data(output.height_data)
        self._compute_bev_data(output.bev_data)
        self._compute_staleness_peak(env_origins, output)

    def _create_workspace(self, num_points: int) -> MapManagerWorkspace:
        dim = int(self.config.staleness_dim)
        cell_count = dim * dim
        base = torch.linspace(-1.0, 1.0, dim, device=self.device, dtype=torch.float32)
        base_gy, base_gx = torch.meshgrid(base, base, indexing="ij")

        return MapManagerWorkspace(
            flat_cell_idx=torch.zeros((self.num_envs, num_points), device=self.device, dtype=torch.long),
            valid_mask=torch.zeros((self.num_envs, num_points), device=self.device, dtype=torch.bool),
            valid_mask_tmp=torch.zeros((self.num_envs, num_points), device=self.device, dtype=torch.bool),
            hit_map_flat=torch.zeros((self.num_envs, cell_count), device=self.device, dtype=torch.float32),
            inverse_hit_map=torch.zeros((self.num_envs, cell_count), device=self.device, dtype=torch.float32),
            exploration_bonus=torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32),
            far_out=torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.float32),
            far_counts=torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.float32),
            far_ones=torch.ones((self.num_envs, cell_count), device=self.device, dtype=torch.float32),
            theta=torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=torch.float32),
            batch_offsets=torch.arange(self.num_envs, device=self.device, dtype=torch.long).view(self.num_envs, 1),
            base_gx=base_gx.unsqueeze(0).contiguous(),
            base_gy=base_gy.unsqueeze(0).contiguous(),
            rel=torch.zeros((self.num_envs, num_points), device=self.device, dtype=torch.float32),
            col_idx=torch.zeros((self.num_envs, num_points), device=self.device, dtype=torch.long),
        )

    def _ensure_workspace(self, num_points: int, lidar_dtype: torch.dtype) -> None:
        if num_points != self._num_lidar_points:
            self._num_lidar_points = num_points
            self._workspace = self._create_workspace(num_points)
        self._bev_builder.ensure_shape(
            batch_size=self.num_envs,
            num_points=num_points,
            device=self.device,
            dtype=lidar_dtype,
        )

    def _update_staleness_map(self, lidar_hits_w, env_origins, dt, output_exploration_bonus: torch.Tensor) -> None:
        """Update staleness (decay + clear seen areas) and write exploration reward."""
        ws = self._workspace
        dim = int(self.config.staleness_dim)
        half_size = float(self.config.patrol_size) / 2.0
        index_scale = dim / float(self.config.patrol_size)

        self.staleness_maps.add_(float(dt) * float(self.config.staleness_decay_rate))
        torch.minimum(self.staleness_maps, self.patrol_mask, out=self.staleness_maps)

        # X pass: compute col indices and the first part of the valid mask.
        torch.sub(lidar_hits_w[..., 0], env_origins[:, 0].unsqueeze(1), out=ws.rel)
        torch.ge(ws.rel, -half_size, out=ws.valid_mask)
        torch.lt(ws.rel, half_size, out=ws.valid_mask_tmp)
        ws.valid_mask.logical_and_(ws.valid_mask_tmp)

        torch.nan_to_num(ws.rel, nan=0.0, posinf=0.0, neginf=0.0, out=ws.rel)
        ws.rel.add_(half_size)
        ws.rel.mul_(index_scale)
        ws.rel.floor_()
        ws.col_idx.copy_(ws.rel)
        ws.col_idx.clamp_(0, dim - 1)

        # Y pass: finish the valid mask and compute flat cell indices.
        torch.sub(lidar_hits_w[..., 1], env_origins[:, 1].unsqueeze(1), out=ws.rel)
        torch.ge(ws.rel, -half_size, out=ws.valid_mask_tmp)
        ws.valid_mask.logical_and_(ws.valid_mask_tmp)
        torch.lt(ws.rel, half_size, out=ws.valid_mask_tmp)
        ws.valid_mask.logical_and_(ws.valid_mask_tmp)

        torch.nan_to_num(ws.rel, nan=0.0, posinf=0.0, neginf=0.0, out=ws.rel)
        ws.rel.add_(half_size)
        ws.rel.mul_(index_scale)
        ws.rel.floor_()

        ws.flat_cell_idx.copy_(ws.rel)
        ws.flat_cell_idx.clamp_(0, dim - 1)
        ws.flat_cell_idx.mul_(dim)
        ws.flat_cell_idx.add_(ws.col_idx)

        # Use the scratch buffer as the hit indicator source (0/1) to avoid per-step allocations.
        ws.rel.zero_()
        ws.rel.masked_fill_(ws.valid_mask, 1.0)
        ws.hit_map_flat.zero_()
        ws.hit_map_flat.scatter_reduce_(
            1, ws.flat_cell_idx, ws.rel, reduce="amax", include_self=False
        )

        staleness_flat = self.staleness_maps.view(self.num_envs, dim * dim)
        torch.mul(staleness_flat, ws.hit_map_flat, out=ws.inverse_hit_map)
        ws.exploration_bonus.copy_(ws.inverse_hit_map.sum(dim=1))
        output_exploration_bonus.copy_(ws.exploration_bonus)

        ws.inverse_hit_map.copy_(ws.hit_map_flat)
        ws.inverse_hit_map.neg_()
        ws.inverse_hit_map.add_(1.0)
        staleness_flat.mul_(ws.inverse_hit_map)

    def _get_far_staleness(self, robot_pos_w, robot_yaw_w, env_origins, output_far: torch.Tensor) -> None:
        """Get average staleness in 8 cardinal directions around the robot."""
        ws = self._workspace
        b = self.num_envs
        h = w = self.config.staleness_dim
        half = float(self.config.patrol_size) / 2.0

        grid_x = ws.base_gx.expand(b, -1, -1)
        grid_y = ws.base_gy.expand(b, -1, -1)
        pixel_x = grid_x * half + env_origins[:, 0].view(b, 1, 1)
        pixel_y = grid_y * half + env_origins[:, 1].view(b, 1, 1)

        dx = pixel_x - robot_pos_w[:, 0].view(b, 1, 1)
        dy = pixel_y - robot_pos_w[:, 1].view(b, 1, 1)

        cos = torch.cos(robot_yaw_w).view(b, 1, 1)
        sin = torch.sin(robot_yaw_w).view(b, 1, 1)

        rx = dx * cos + dy * sin
        ry = -dx * sin + dy * cos

        angles = torch.atan2(ry, rx)
        angles.add_(2 * np.pi)
        angles.remainder_(2 * np.pi)
        oct_idx = (angles / (np.pi / 4)).long() % 8

        flat_map = self.staleness_maps.view(b, h * w)
        flat_idx = oct_idx.view(b, h * w)

        ws.far_out.zero_()
        ws.far_counts.zero_()
        ws.far_out.scatter_add_(1, flat_idx, flat_map)
        ws.far_counts.scatter_add_(1, flat_idx, ws.far_ones)

        output_far.copy_(ws.far_out)
        output_far.div_(ws.far_counts + 1e-6)

    def _sample_egocentric_maps(self, robot_pos_w, robot_yaw_w, env_origins, output_nav: torch.Tensor) -> None:
        """Generate egocentric nav observation (1 channel: staleness)."""
        ws = self._workspace
        cos = torch.cos(robot_yaw_w).squeeze(-1)
        sin = torch.sin(robot_yaw_w).squeeze(-1)

        half_patrol = float(self.config.patrol_size) / 2.0
        rel_x = (robot_pos_w[:, 0] - env_origins[:, 0]) / half_patrol
        rel_y = (robot_pos_w[:, 1] - env_origins[:, 1]) / half_patrol

        tx = -rel_x * cos - rel_y * sin
        ty = rel_x * sin - rel_y * cos

        scale = float(self.config.nav_size) / float(self.config.patrol_size)
        ws.theta.zero_()
        ws.theta[:, 0, 0] = scale * cos
        ws.theta[:, 0, 1] = -scale * sin
        ws.theta[:, 0, 2] = tx
        ws.theta[:, 1, 0] = scale * sin
        ws.theta[:, 1, 1] = scale * cos
        ws.theta[:, 1, 2] = ty

        grid_s = F.affine_grid(
            ws.theta,
            [self.num_envs, 1, self.config.nav_dim, self.config.nav_dim],
            align_corners=False,
        )
        nav_staleness = F.grid_sample(self.staleness_maps, grid_s, align_corners=False, padding_mode="border")
        output_nav.copy_(nav_staleness)

    def _compute_staleness_peak(self, env_origins: torch.Tensor, output: MapManagerOutput) -> None:
        """Find the highest-staleness region and write world XY + value to output."""
        dim = int(self.config.staleness_dim)
        half_size = float(self.config.patrol_size) / 2.0
        cell_size = float(self.config.patrol_size) / dim  # meters per cell

        # 3x3 avg_pool to smooth single-cell noise into region peaks
        smoothed = F.avg_pool2d(self.staleness_maps, kernel_size=3, stride=1, padding=1)
        flat = smoothed.view(self.num_envs, -1)
        peak_flat_idx = torch.argmax(flat, dim=1)

        peak_row = peak_flat_idx // dim
        peak_col = peak_flat_idx % dim

        # Grid cell center → world coordinates
        output.staleness_peak_world[:, 0] = (peak_col.float() + 0.5) * cell_size - half_size + env_origins[:, 0]
        output.staleness_peak_world[:, 1] = (peak_row.float() + 0.5) * cell_size - half_size + env_origins[:, 1]

        # Report unsmoothed staleness at the peak cell
        staleness_flat = self.staleness_maps.view(self.num_envs, -1)
        output.staleness_peak_value[:] = staleness_flat.gather(1, peak_flat_idx.unsqueeze(1)).squeeze(1)

    def reset(self, env_ids):
        """Reset staleness maps for specific environments."""
        self.staleness_maps[env_ids] = 0.0

    def _compute_height_data(self, output_height: torch.Tensor) -> None:
        """Compute height scanner observation from internal sensor reference."""
        height_scanner_pos_z = self._height_scanner.data.pos_w[:, 2].view(self.num_envs, 1, 1)
        height_scanner_ray_hits_z = self._height_scanner.data.ray_hits_w[..., 2]

        output_height.copy_(height_scanner_ray_hits_z.view(self.num_envs, 64, 64))
        output_height.mul_(-1.0)
        output_height.add_(height_scanner_pos_z)
        output_height.sub_(0.5)
        output_height.clamp_(-1.0, 1.0)

    def _compute_bev_data(self, output_bev: torch.Tensor) -> None:
        """Compute BEV observation from internal LiDAR sensor reference."""
        world_points = self._lidar_sensor.data.ray_hits_w
        sensor_position_world = self._lidar_sensor.data.pos_w
        sensor_quaternion_world_wxyz = self._lidar_sensor.data.quat_w

        torch.nan_to_num(world_points, nan=0.0, posinf=0.0, neginf=0.0, out=self._bev_builder.ego_points)
        transform_world_to_ego_inplace(
            self._bev_builder.ego_points,
            sensor_position_world,
            sensor_quaternion_world_wxyz,
            out=self._bev_builder.ego_points,
            rotation_world_to_sensor=self._bev_builder.rotation_matrix,
            relative_points=self._bev_builder.relative_points,
        )
        build_bev_inplace(self._bev_builder, self._bev_builder.ego_points, output_bev)

