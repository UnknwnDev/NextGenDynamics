import numpy as np
from .obstacles_generator import get_height_at_point
from .custom_terrain_config import CustomTerrainCfg
from .mesh_loader import get_obstacle_radius

def spawn_point_sampler(height_map, obstacles_placement, config: CustomTerrainCfg) -> np.ndarray:
    """
    Samples valid spawn points on the terrain, avoiding obstacles and rough terrain.
    
    Args:
        height_map (np.ndarray): The terrain height map.
        obstacles_placement (dict): Dictionary containing obstacle placement info (positions, scales).
        config (CustomTerrainCfg): Terrain configuration.

    Returns:
        np.ndarray: Array of spawn points (N, 3) -> (x, y, z).
    """
    
    # Load parameters from config
    num_points = config.num_points
    sample_radius = config.sample_radius
    flatness_threshold = config.flatness_threshold
    max_attempts = config.max_attempts
    margin = config.margin

    # 1. Preprocess obstacles for fast collision checking
    obstacle_circles = []
    if obstacles_placement and config.obstacles:
        # Map type to config for radius lookup
        obs_config_map = {obs.type: obs for obs in config.obstacles}

        for obs_type, data in obstacles_placement.items():
            positions = data.get("positions")
            scales = data.get("scales")
            
            if positions is None or scales is None:
                continue
                
            # Get base radius from config if available (for custom meshes)
            base_radius = None
            if obs_type in obs_config_map:
                base_radius = obs_config_map[obs_type].radius
            
            # Calculate radii using mesh_loader function
            radii = get_obstacle_radius(obs_type, scales, base_radius)
            
            # Store (x, y, radius)
            for i in range(len(positions)):
                obstacle_circles.append([positions[i, 0], positions[i, 1], radii[i]])
    
    obs_data = np.array(obstacle_circles) # (M, 3) -> x, y, r
    
    valid_points = []
    
    # Terrain bounds (with margin)
    x_min = -config.size[0] / 2 + margin
    x_max = config.size[0] / 2 - margin
    y_min = -config.size[1] / 2 + margin
    y_max = config.size[1] / 2 - margin
    
    rows, cols = height_map.shape
    meter_per_grid = config.meter_per_grid
    
    for _ in range(num_points):
        found = False
        for attempt in range(max_attempts):
            # Sample random (x, y)
            rx = np.random.uniform(x_min, x_max)
            ry = np.random.uniform(y_min, y_max)
            
            # 2. Obstacle Collision Check
            if len(obs_data) > 0:
                # Calculate distances to all obstacles
                # obs_data[:, :2] is (M, 2), point is (2,)
                dists = np.linalg.norm(obs_data[:, :2] - np.array([rx, ry]), axis=1)
                # Check if any distance is less than sum of radii + margin
                # obs_data[:, 2] is obstacle radii
                min_dists = obs_data[:, 2] + sample_radius + 0.2 # 0.2m margin
                if np.any(dists < min_dists):
                    continue # Collision detected
            
            # 3. Flatness Check
            # Convert world pos to grid indices
            # We assume the map is centered at (0,0)
            grid_x = int((rx + config.size[0] / 2) / meter_per_grid)
            grid_y = int((ry + config.size[1] / 2) / meter_per_grid)
            
            # Define window for robot footprint
            grid_r = int(np.ceil(sample_radius / meter_per_grid))
            
            gx_min = max(0, grid_x - grid_r)
            gx_max = min(cols, grid_x + grid_r + 1)
            gy_min = max(0, grid_y - grid_r)
            gy_max = min(rows, grid_y + grid_r + 1)
            
            patch = height_map[gy_min:gy_max, gx_min:gx_max]
            
            if patch.size == 0:
                continue
                
            h_min = np.min(patch)
            h_max = np.max(patch)
            
            if (h_max - h_min) > flatness_threshold:
                continue # Too rough
            
            # 4. Valid point found
            # Get exact height at point
            coords = np.array([[rx, ry]])
            rz = get_height_at_point(config, height_map, coords)[0]
            
            valid_points.append([rx, ry, rz + 0.2]) # Add slight offset to z for safety
            found = True
            break
        
        if not found:
            # Fallback if no point found
            print(f"Warning: Could not find valid spawn point after {max_attempts} attempts.")
            valid_points.append([0.0, 0.0, 1.0])
            
    return np.array(valid_points)
    