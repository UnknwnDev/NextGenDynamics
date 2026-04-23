import torch
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

# -------------------------------
# Geometry helpers
# -------------------------------

def quaternion_to_rotation_matrix_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation matrices.
    Args:
        quaternion_wxyz: (batch_size, 4) in [w, x, y, z] order.
    Returns:
        rotation_matrix: (batch_size, 3, 3)
    """
    w, x, y, z = quaternion_wxyz.unbind(-1)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotation_matrix = torch.empty(
        (quaternion_wxyz.shape[0], 3, 3),
        device=quaternion_wxyz.device,
        dtype=quaternion_wxyz.dtype
    )

    rotation_matrix[:, 0, 0] = 1 - 2 * (y2 + z2)
    rotation_matrix[:, 0, 1] = 2 * (xy - wz)
    rotation_matrix[:, 0, 2] = 2 * (xz + wy)

    rotation_matrix[:, 1, 0] = 2 * (xy + wz)
    rotation_matrix[:, 1, 1] = 1 - 2 * (x2 + z2)
    rotation_matrix[:, 1, 2] = 2 * (yz - wx)

    rotation_matrix[:, 2, 0] = 2 * (xz - wy)
    rotation_matrix[:, 2, 1] = 2 * (yz + wx)
    rotation_matrix[:, 2, 2] = 1 - 2 * (x2 + y2)
    return rotation_matrix


def transform_world_to_ego(
    world_points: torch.Tensor,                 # (batch_size, num_points, 3)
    sensor_position_world: torch.Tensor,        # (batch_size, 3)
    sensor_quaternion_world_wxyz: torch.Tensor  # (batch_size, 4) [w,x,y,z]
) -> torch.Tensor:
    """
    Transform points from world frame into the ego (sensor/base) frame.
    Returns:
        ego_points: (batch_size, num_points, 3)
    """
    rotation_world_to_sensor = quaternion_to_rotation_matrix_wxyz(
        sensor_quaternion_world_wxyz
    )  # (B, 3, 3)

    # Translate, then rotate by R^T to go into sensor frame
    relative_points = world_points - sensor_position_world[:, None, :]   # (B, P, 3)
    ego_points = torch.einsum(
        "bij,bpj->bpi", rotation_world_to_sensor.transpose(1, 2), relative_points
    )
    return ego_points

# -------------------------------
# BEV builder
# -------------------------------

@torch.no_grad()
def build_bev(
    ego_points: torch.Tensor,                          # (batch_size, num_points, 3) -> [x, y, z]
    intensities: Optional[torch.Tensor] = None,        # (batch_size, num_points) or None
    x_limits_meters: Tuple[float, float] = (-12.8, 12.8),
    y_limits_meters: Tuple[float, float] = (-12.8, 12.8),
    z_limits_meters: Tuple[float, float] = (-3.0, 2.0),
    resolution_meters: float = 0.40, # grid will be 64x64 for these limits
    density_normalization: float = 10.0,
    channels: Sequence[str] = ("max_height", "mean_height", "mean_intensity", "density"),
) -> torch.Tensor:
    """
    Create a BEV tensor suitable for 2D CNN input.

    Args:
        ego_points: (B, P, 3) points in ego frame [x, y, z].
        intensities: (B, P) optional reflectance/intensity values in [0, 1].
        x_limits_meters, y_limits_meters, z_limits_meters: (min, max) per axis.
        resolution_meters: grid cell size in meters.
        density_normalization: divisor to scale density to ~[0, 1].
        channels: list from {"max_height", "mean_height", "mean_intensity", "density"}.

    Returns:
        bev: (B, C, H, W) float32, channels in the provided order,
             laid out as channels_last memory format for better Conv2d kernels.
    """
    device = ego_points.device
    dtype = ego_points.dtype
    batch_size, num_points, _ = ego_points.shape

    x_min, x_max = x_limits_meters
    y_min, y_max = y_limits_meters
    z_min, z_max = z_limits_meters

    width_cells  = int((x_max - x_min) / resolution_meters)   # along +x (columns)
    height_cells = int((y_max - y_min) / resolution_meters)   # along +y (rows)
    total_cells_per_sample = height_cells * width_cells
    total_cells_all_batches = batch_size * total_cells_per_sample

    x_values, y_values, z_values = ego_points.unbind(-1)

    # Mask points that fall within the configured region of interest
    valid_mask = (
        (x_values >= x_min) & (x_values < x_max) &
        (y_values >= y_min) & (y_values < y_max) &
        (z_values >= z_min) & (z_values <= z_max)
    )

    # Map (x, y) to integer grid indices (column, row)
    grid_x = torch.clamp(((x_values - x_min) / resolution_meters).floor().long(), 0, width_cells  - 1)
    grid_y = torch.clamp(((y_values - y_min) / resolution_meters).floor().long(), 0, height_cells - 1)

    # Keep only valid points
    batch_indices, point_indices = torch.where(valid_mask)
    if batch_indices.numel() == 0:
        output = torch.zeros(
            (batch_size, len(channels), height_cells, width_cells),
            device=device, dtype=torch.float32
        )
        return output.contiguous(memory_format=torch.channels_last)

    grid_x_valid = grid_x[batch_indices, point_indices]
    grid_y_valid = grid_y[batch_indices, point_indices]
    linear_index_within_sample = grid_y_valid * width_cells + grid_x_valid              # (N_valid,)

    z_valid = z_values[batch_indices, point_indices]

    # Convert (sample_id, cell_id) into a single global linear index for vectorized scatter
    global_linear_index = linear_index_within_sample + batch_indices * total_cells_per_sample
    global_size = total_cells_all_batches

    channel_outputs = []
    z_range_span = max(1e-6, (z_max - z_min))  # safe normalization denominator

    for channel_name in channels:
        if channel_name == "max_height":
            # Per-cell max z, normalized to [0, 1] using [z_min, z_max]
            max_buffer = torch.full((global_size,), z_min, device=device, dtype=dtype)
            max_buffer.scatter_reduce_(0, global_linear_index, z_valid, reduce="amax")
            max_height_channel = max_buffer.view(batch_size, height_cells, width_cells)
            max_height_channel = (max_height_channel - z_min) / z_range_span
            channel_outputs.append(max_height_channel)

        elif channel_name == "mean_height":
            # Per-cell mean z; empty cells become z_min; then normalize to [0, 1]
            sum_z = torch.zeros((global_size,), device=device, dtype=dtype)
            count_points = torch.zeros((global_size,), device=device, dtype=dtype)
            sum_z.scatter_add_(0, global_linear_index, z_valid)
            count_points.scatter_add_(0, global_linear_index, torch.ones_like(z_valid))

            mean_z = sum_z / torch.clamp(count_points, min=1.0)
            mean_z = torch.where(count_points > 0, mean_z, torch.full_like(mean_z, z_min))
            mean_height_channel = (mean_z.view(batch_size, height_cells, width_cells) - z_min) / z_range_span
            channel_outputs.append(mean_height_channel)

        elif channel_name == "mean_intensity":
            if intensities is None:
                intensity_values = torch.zeros((batch_size, num_points), device=device, dtype=dtype)
            else:
                intensity_values = intensities

            intensity_valid = intensity_values[batch_indices, point_indices]

            # Per-cell mean intensity in [0, 1]
            sum_intensity = torch.zeros((global_size,), device=device, dtype=dtype)
            count_points = torch.zeros((global_size,), device=device, dtype=dtype)
            sum_intensity.scatter_add_(0, global_linear_index, intensity_valid)
            count_points.scatter_add_(0, global_linear_index, torch.ones_like(intensity_valid))
            mean_intensity_channel = (sum_intensity / torch.clamp(count_points, min=1.0)).view(
                batch_size, height_cells, width_cells
            )
            channel_outputs.append(mean_intensity_channel)

        elif channel_name == "density":
            # Per-cell point counts scaled to ~[0,1]
            counts = torch.bincount(global_linear_index, minlength=global_size).to(dtype)
            density_channel = (counts / max(1.0, density_normalization)).clamp(max=1.0).view(
                batch_size, height_cells, width_cells
            )
            channel_outputs.append(density_channel)

        else:
            raise ValueError(f"Unknown channel name: {channel_name}")

    bev_tensor = torch.stack(channel_outputs, dim=1).to(torch.float32)  # (B, C, H, W)
    # Keep NCHW semantics but set NHWC (channels_last) strides for faster Conv2d kernels
    bev_tensor = bev_tensor.contiguous(memory_format=torch.channels_last)
    return bev_tensor


# -------------------------------
# Visualization helpers
# -------------------------------

def visualize_bev(
    bev_tensor: torch.Tensor,
    batch_idx: int = 0,
    channel_names: Sequence[str] = ("max_height", "mean_height", "mean_intensity", "density"),
    x_limits: Tuple[float, float] = (-12.8, 12.8),
    y_limits: Tuple[float, float] = (-12.8, 12.8),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize BEV tensor channels in a 2x2 subplot grid.

    Args:
        bev_tensor: (B, C, H, W) BEV tensor from build_bev()
        batch_idx: which batch sample to visualize
        channel_names: names for each channel
        x_limits: (x_min, x_max) in meters
        y_limits: (y_min, y_max) in meters
        save_path: if provided, save figure to this path instead of showing
    """
    bev_np = bev_tensor[batch_idx].cpu().numpy()  # (C, H, W)
    num_channels = bev_np.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(num_channels, 4)):
        ax = axes[i]
        im = ax.imshow(
            bev_np[i],
            origin='lower',
            extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
            cmap='viridis',
            vmin=0, vmax=1
        )
        ax.set_title(channel_names[i] if i < len(channel_names) else f"Channel {i}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(num_channels, 4):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_bev_with_pointcloud(
    ego_points: torch.Tensor,
    bev_tensor: torch.Tensor,
    batch_idx: int = 0,
    x_limits: Tuple[float, float] = (-12.8, 12.8),
    y_limits: Tuple[float, float] = (-12.8, 12.8),
    z_limits: Tuple[float, float] = (-3.0, 2.0),
    save_path: Optional[str] = None
) -> None:
    """
    Side-by-side comparison of pointcloud (top-down) and BEV max_height channel.

    Args:
        ego_points: (B, P, 3) points in ego frame
        bev_tensor: (B, C, H, W) BEV tensor
        batch_idx: which batch sample to visualize
        x_limits, y_limits, z_limits: spatial limits in meters
        save_path: if provided, save figure to this path
    """
    points_np = ego_points[batch_idx].cpu().numpy()  # (P, 3)
    bev_np = bev_tensor[batch_idx, 0].cpu().numpy()  # max_height channel (H, W)

    # Filter points within limits for visualization
    mask = (
        (points_np[:, 0] >= x_limits[0]) & (points_np[:, 0] < x_limits[1]) &
        (points_np[:, 1] >= y_limits[0]) & (points_np[:, 1] < y_limits[1]) &
        (points_np[:, 2] >= z_limits[0]) & (points_np[:, 2] <= z_limits[1])
    )
    filtered_points = points_np[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pointcloud top-down view
    ax0 = axes[0]
    if len(filtered_points) > 0:
        z_norm = (filtered_points[:, 2] - z_limits[0]) / (z_limits[1] - z_limits[0])
        scatter = ax0.scatter(
            filtered_points[:, 0], filtered_points[:, 1],
            c=z_norm, cmap='viridis', s=1, vmin=0, vmax=1
        )
        fig.colorbar(scatter, ax=ax0, label='Height (normalized)')
    ax0.set_xlim(x_limits)
    ax0.set_ylim(y_limits)
    ax0.set_aspect('equal')
    ax0.set_title(f"Pointcloud Top-Down ({len(filtered_points)} pts)")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")

    # Right: BEV max_height
    ax1 = axes[1]
    im = ax1.imshow(
        bev_np,
        origin='lower',
        extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
        cmap='viridis',
        vmin=0, vmax=1
    )
    ax1.set_title("BEV max_height")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    fig.colorbar(im, ax=ax1, label='Height (normalized)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


class BEVDebugVisualizer:
    """
    Real-time BEV visualizer for debugging during training/inference.
    Uses matplotlib interactive mode for non-blocking updates.
    """

    def __init__(
        self,
        x_limits: Tuple[float, float] = (-12.8, 12.8),
        y_limits: Tuple[float, float] = (-12.8, 12.8),
        z_limits: Tuple[float, float] = (-3.0, 2.0),
        channel_names: Sequence[str] = ("max_height", "mean_height", "mean_intensity", "density"),
    ):
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.channel_names = channel_names

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.axes = self.axes.flatten()
        self.images = [None] * 6
        self.scatter = None
        self.initialized = False

    def update(
        self,
        bev_tensor: torch.Tensor,
        ego_points: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> None:
        """
        Update the visualization with new data (non-blocking).

        Args:
            bev_tensor: (B, C, H, W) BEV tensor
            ego_points: (B, P, 3) optional pointcloud for comparison
            batch_idx: which batch to visualize
        """
        bev_np = bev_tensor[batch_idx].cpu().numpy()

        if not self.initialized:
            self._init_plots(bev_np, ego_points, batch_idx)
            self.initialized = True
        else:
            self._update_plots(bev_np, ego_points, batch_idx)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _init_plots(self, bev_np, ego_points, batch_idx):
        # BEV channels (first 4 subplots)
        for i in range(min(4, bev_np.shape[0])):
            ax = self.axes[i]
            self.images[i] = ax.imshow(
                bev_np[i],
                origin='lower',
                extent=[self.x_limits[0], self.x_limits[1], self.y_limits[0], self.y_limits[1]],
                cmap='viridis', vmin=0, vmax=1
            )
            title = self.channel_names[i] if i < len(self.channel_names) else f"Channel {i}"
            ax.set_title(title)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

        # Pointcloud (5th subplot)
        ax_pc = self.axes[4]
        if ego_points is not None:
            points_np = ego_points[batch_idx].cpu().numpy()
            mask = self._get_valid_mask(points_np)
            filtered = points_np[mask]
            if len(filtered) > 0:
                z_norm = (filtered[:, 2] - self.z_limits[0]) / (self.z_limits[1] - self.z_limits[0])
                self.scatter = ax_pc.scatter(filtered[:, 0], filtered[:, 1], c=z_norm, cmap='viridis', s=1, vmin=0, vmax=1)
        ax_pc.set_xlim(self.x_limits)
        ax_pc.set_ylim(self.y_limits)
        ax_pc.set_aspect('equal')
        ax_pc.set_title("Pointcloud")
        ax_pc.set_xlabel("X (m)")
        ax_pc.set_ylabel("Y (m)")

        # Hide 6th subplot
        self.axes[5].axis('off')
        plt.tight_layout()

    def _update_plots(self, bev_np, ego_points, batch_idx):
        for i in range(min(4, bev_np.shape[0])):
            img = self.images[i]
            if img is not None:
                img.set_data(bev_np[i])

        if ego_points is not None:
            ax_pc = self.axes[4]
            points_np = ego_points[batch_idx].cpu().numpy()
            mask = self._get_valid_mask(points_np)
            filtered = points_np[mask]
            if self.scatter is not None:
                self.scatter.remove()
            if len(filtered) > 0:
                z_norm = (filtered[:, 2] - self.z_limits[0]) / (self.z_limits[1] - self.z_limits[0])
                self.scatter = ax_pc.scatter(filtered[:, 0], filtered[:, 1], c=z_norm, cmap='viridis', s=1, vmin=0, vmax=1)
            ax_pc.set_title(f"Pointcloud ({len(filtered)} pts)")

    def _get_valid_mask(self, points_np):
        return (
            (points_np[:, 0] >= self.x_limits[0]) & (points_np[:, 0] < self.x_limits[1]) &
            (points_np[:, 1] >= self.y_limits[0]) & (points_np[:, 1] < self.y_limits[1]) &
            (points_np[:, 2] >= self.z_limits[0]) & (points_np[:, 2] <= self.z_limits[1])
        )

    def save_snapshot(self, path: str) -> None:
        """Save current figure to file."""
        self.fig.savefig(path, dpi=150)

    def close(self) -> None:
        """Close the visualizer and disable interactive mode."""
        plt.ioff()
        plt.close(self.fig)
