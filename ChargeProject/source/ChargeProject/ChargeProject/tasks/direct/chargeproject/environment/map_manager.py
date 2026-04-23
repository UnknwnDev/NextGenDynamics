"""
MapManager: Manages exploration/staleness maps and generates nav_data for patrol mode.

Ported from spiderbot-ai branch and modified to work with custom_terrain_generator.
"""
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from isaaclab.sensors import RayCaster

from ChargeProject.tasks.direct.chargeproject.utils.cloudpoint_to_bev import (
    build_bev, transform_world_to_ego
)
from ChargeProject.tasks.direct.chargeproject.environment.terrain_gen.custom_terrain_generator import terrain_generator


@dataclass
class MapManagerOutput:
    """Return type for MapManager.update()"""
    nav_data: torch.Tensor       # (num_envs, 1, nav_dim, nav_dim)
    bev_data: torch.Tensor       # (num_envs, 3, 64, 64)
    height_data: torch.Tensor    # (num_envs, 64, 64)
    far_staleness: torch.Tensor  # (num_envs, 8)
    exploration_bonus: torch.Tensor  # (num_envs,)


class MapManager:
    def __init__(self, config, num_envs, device, height_scanner: RayCaster, lidar_sensor: RayCaster):
        """
        Initialize MapManager.

        Args:
            config: Environment config containing patrol_size, staleness_dim, nav_size, nav_dim, etc.
            num_envs: Number of parallel environments
            device: torch device
            height_scanner: Height scanner RayCaster sensor reference
            lidar_sensor: LiDAR RayCaster sensor reference
        """
        self.device = device
        self.num_envs = num_envs
        self.config = config

        # Store sensor references
        self._height_scanner = height_scanner
        self._lidar_sensor = lidar_sensor

        # Load height_map directly from terrain_generator
        self._terrain_height_map = torch.from_numpy(terrain_generator.height_map).to(
            dtype=torch.float32, device=device
        )

        # --- Generate Circular Patrol Mask ---
        # Coordinates -1 to 1
        x = torch.linspace(-1, 1, self.config.staleness_dim, device=device)
        y = torch.linspace(-1, 1, self.config.staleness_dim, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        # Distance from center (Normalized 0 to 1)
        dist = torch.sqrt(grid_x**2 + grid_y**2)

        # Fade width: 15% of radius
        fade_width = 0.15

        self.patrol_mask = torch.clamp((1.0 - dist) / fade_width, min=0.0, max=1.0)

        # Reshape to (1, 1, H, W) for broadcasting
        self.patrol_mask = self.patrol_mask.view(1, 1, self.config.staleness_dim, self.config.staleness_dim)

        # Initialize per-env staleness (0.0 = Clean, increases over time)
        self.staleness_maps = torch.zeros(self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim, device=device)

        # Pre-calculate 8 cardinal relative offsets (Radius = 12.0m)
        # Angles: 0 (Front), 45, 90 (Left), 135, 180 (Back), etc.
        scan_radius = 12.0
        angles = torch.arange(0, 2*np.pi, 2*np.pi/8, device=device)
        self.far_sensor_offsets = torch.stack([
            scan_radius * torch.cos(angles),
            scan_radius * torch.sin(angles)
        ], dim=1)  # (8, 2)

    def update(self, env_origins, robot_pos_w, robot_yaw_w, dt) -> MapManagerOutput:
        """
        Update all map-related data and generate observations.

        Args:
            env_origins: (num_envs, 3) Environment origin positions
            robot_pos_w: (num_envs, 3) Robot world positions
            robot_yaw_w: (num_envs, 1) Robot yaw angles
            dt: Time step

        Returns:
            MapManagerOutput containing nav_data, bev_data, height_data,
            far_staleness, and exploration_bonus
        """
        # Get lidar hits from internal sensor
        lidar_hits_w = self._lidar_sensor.data.ray_hits_w

        # --- Existing staleness updates ---
        exploration_bonus = self._update_staleness_map(lidar_hits_w, env_origins, dt)
        far_staleness = self._get_far_staleness(robot_pos_w, robot_yaw_w, env_origins)
        nav_data = self._sample_egocentric_maps(robot_pos_w, robot_yaw_w, env_origins)

        # --- New data generation ---
        height_data = self._compute_height_data()
        bev_data = self._compute_bev_data()

        return MapManagerOutput(
            nav_data=nav_data,
            bev_data=bev_data,
            height_data=height_data,
            far_staleness=far_staleness,
            exploration_bonus=exploration_bonus
        )

    def _update_staleness_map(self, lidar_hits_w, env_origins, dt):
        """Update staleness (decay + clear seen areas) and return exploration reward."""
        # Decay (Everything gets dusty over time)
        self.staleness_maps += dt * self.config.staleness_decay_rate
        self.staleness_maps = torch.minimum(self.staleness_maps, self.patrol_mask)

        # Calculate hits relative to Env Origin
        rel_hits = lidar_hits_w - env_origins.unsqueeze(1)

        # Map to Pixel Coordinates
        half_size = self.config.patrol_size / 2
        col = ((rel_hits[..., 0] + half_size) / self.config.patrol_size * self.config.staleness_dim).long()
        row = ((rel_hits[..., 1] + half_size) / self.config.patrol_size * self.config.staleness_dim).long()

        # Filter valid hits within patrol zone
        mask = (col >= 0) & (col < self.config.staleness_dim) & (row >= 0) & (row < self.config.staleness_dim)

        # Identify indices to clear
        batch_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(col)
        # Linear index for flatten: B * (H*W) + Row * W + Col
        flat_idx = batch_ids * (self.config.staleness_dim**2) + row * self.config.staleness_dim + col

        valid_idx = flat_idx[mask]

        total_cleared_value = torch.zeros(self.num_envs, device=self.device)

        if valid_idx.numel() > 0:
            flat_map = self.staleness_maps.view(-1)

            # --- Exploration Reward Calculation ---
            # Get values before we clear them
            # Note: Multiple rays might hit the same cell. We should only count unique cells
            unique_idx, _ = torch.unique(valid_idx, return_inverse=True)

            # Map unique indices back to environment IDs
            env_ids_for_unique = torch.div(unique_idx, (self.config.staleness_dim**2), rounding_mode='floor')

            values_to_clear = flat_map[unique_idx]

            # Sum values per environment
            total_cleared_value.index_add_(0, env_ids_for_unique, values_to_clear)

            # Clear map
            flat_map[valid_idx] = 0.0
            self.staleness_maps = flat_map.view(self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim)

        return total_cleared_value

    def _get_far_staleness(self, robot_pos_w, robot_yaw_w, env_origins):
        """Get average staleness in 8 cardinal directions around the robot."""
        B = self.num_envs
        H = W = self.config.staleness_dim
        device = self.device

        # Build pixel coordinate grid for the staleness map
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        grid = torch.stack([gx, gy], dim=-1)
        grid = grid.view(1, H, W, 2).expand(B, -1, -1, -1)

        # Convert to metric world coordinates inside the patrol zone
        half = self.config.patrol_size / 2
        pixel_x = grid[..., 0] * half + env_origins[:, 0].view(B, 1, 1)
        pixel_y = grid[..., 1] * half + env_origins[:, 1].view(B, 1, 1)

        # Vector from robot -> each pixel
        dx = pixel_x - robot_pos_w[:, 0].view(B, 1, 1)
        dy = pixel_y - robot_pos_w[:, 1].view(B, 1, 1)

        # Angle of each pixel in robot's rotated frame
        cos = torch.cos(robot_yaw_w).view(B, 1, 1)
        sin = torch.sin(robot_yaw_w).view(B, 1, 1)

        # Rotate world vectors into robot frame
        rx = dx * cos + dy * sin
        ry = -dx * sin + dy * cos

        # Angle in [-pi, pi]
        angles = torch.atan2(ry, rx)

        # Convert to [0, 2*pi)
        angles = (angles + 2*np.pi) % (2*np.pi)

        # Compute octant index 0..7 for each pixel
        oct_idx = (angles / (np.pi/4)).long() % 8

        # Average staleness inside each octant
        flat_map = self.staleness_maps.view(B, H*W)
        flat_idx = oct_idx.view(B, H*W)

        out = torch.zeros(B, 8, device=device)
        counts = torch.zeros(B, 8, device=device)

        out.scatter_add_(1, flat_idx, flat_map)
        counts.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))

        out = out / (counts + 1e-6)

        return out

    def _sample_egocentric_maps(self, robot_pos_w, robot_yaw_w, env_origins):
        """Generate egocentric nav observation (1 channel: staleness)."""
        cos = torch.cos(robot_yaw_w).squeeze(-1)
        sin = torch.sin(robot_yaw_w).squeeze(-1)

        # --- Prepare Affine Data for Local Staleness ---
        # Normals relative to Patrol Zone [-1, 1]
        rel_pos = robot_pos_w - env_origins
        tx_s = rel_pos[:, 0] / (self.config.patrol_size / 2)
        ty_s = rel_pos[:, 1] / (self.config.patrol_size / 2)

        # Zoom: nav_size view inside patrol_size map
        zoom_s = self.config.nav_size / self.config.patrol_size
        theta_s = self._get_affine_matrix(zoom_s, cos, sin, tx_s, ty_s)
        grid_s = F.affine_grid(theta_s, [self.num_envs, 1, self.config.nav_dim, self.config.nav_dim], align_corners=False)
        nav_staleness = F.grid_sample(self.staleness_maps, grid_s, align_corners=False, padding_mode='border')

        return nav_staleness  # (N, 1, nav_dim, nav_dim)

    def _get_affine_matrix(self, scale, cos, sin, tx, ty):
        """Generate 2x3 affine transformation matrix for grid_sample."""
        theta = torch.zeros(self.num_envs, 2, 3, device=self.device)
        theta[:, 0, 0] = scale * cos
        theta[:, 0, 1] = -scale * sin
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = scale * sin
        theta[:, 1, 1] = scale * cos
        theta[:, 1, 2] = ty
        return theta

    def reset(self, env_ids):
        """Reset staleness maps for specific environments."""
        self.staleness_maps[env_ids] = 0.0

    def get_terrain_height(self, positions_xy: torch.Tensor) -> torch.Tensor:
        """
        Get terrain height at given XY positions using bilinear interpolation.

        Args:
            positions_xy: (N, 2) tensor of XY positions in world coordinates

        Returns:
            heights: (N,) tensor of terrain heights
        """
        meter_per_grid = self.config.height_map_meter_per_grid
        terrain_height_map = self._terrain_height_map

        # Convert world coordinates to grid coordinates
        grid_pos = positions_xy / meter_per_grid
        num_rows, num_cols = terrain_height_map.shape
        grid_pos[:, 0] += num_cols / 2  # X -> column offset
        grid_pos[:, 1] += num_rows / 2  # Y -> row offset

        # Get integer coordinates
        col0 = torch.floor(grid_pos[:, 0]).long()
        row0 = torch.floor(grid_pos[:, 1]).long()
        col1 = col0 + 1
        row1 = row0 + 1

        # Clamp coordinates to map bounds
        col0 = torch.clamp(col0, 0, num_cols - 1)
        col1 = torch.clamp(col1, 0, num_cols - 1)
        row0 = torch.clamp(row0, 0, num_rows - 1)
        row1 = torch.clamp(row1, 0, num_rows - 1)

        # Get heights at grid points
        h00 = terrain_height_map[row0, col0]
        h10 = terrain_height_map[row0, col1]
        h01 = terrain_height_map[row1, col0]
        h11 = terrain_height_map[row1, col1]

        # Calculate weights
        wx = grid_pos[:, 0] - col0.float()
        wy = grid_pos[:, 1] - row0.float()

        # Bilinear interpolation
        hx0 = h00 * (1 - wx) + h10 * wx
        hx1 = h01 * (1 - wx) + h11 * wx
        height = hx0 * (1 - wy) + hx1 * wy

        return height

    def _compute_height_data(self) -> torch.Tensor:
        """
        Compute height scanner observation from internal sensor reference.

        Returns:
            height_data: (num_envs, 64, 64) normalized height differences
        """
        height_scanner_pos_z = self._height_scanner.data.pos_w[:, 2]
        height_scanner_ray_hits_z = self._height_scanner.data.ray_hits_w[..., 2]

        height_data = (
            height_scanner_pos_z.unsqueeze(1) - height_scanner_ray_hits_z - 0.5
        ).clip(-1.0, 1.0)

        return height_data.view(self.num_envs, 64, 64)

    def _compute_bev_data(self) -> torch.Tensor:
        """
        Compute BEV observation from internal LiDAR sensor reference.

        Returns:
            bev_data: (num_envs, 3, 64, 64) BEV channels (max_height, mean_height, density)
        """
        world_points = self._lidar_sensor.data.ray_hits_w
        sensor_position_world = self._lidar_sensor.data.pos_w
        sensor_quaternion_world_wxyz = self._lidar_sensor.data.quat_w

        ego_points = transform_world_to_ego(
            world_points,
            sensor_position_world,
            sensor_quaternion_world_wxyz
        )

        bev_data = build_bev(
            ego_points,
            channels=("max_height", "mean_height", "density"),
        )

        return bev_data
