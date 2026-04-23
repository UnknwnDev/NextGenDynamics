from pathlib import Path
from dataclasses import dataclass

CURRENT_DIR = Path(__file__).resolve().parent

class ObstacleType:
    CUSTOM_MESH = "custom_mesh"
    CUBE = "cube"
    SPHERE = "sphere"

@dataclass
class Obstacle:
    type: str
    path: None | str = None  # Required if type is CUSTOM_MESH
    scale_range: tuple[float, float] = (0.5, 2.0)
    num_instances: int = 100
    radius: float | None = None  # Base radius for custom meshes (optional)


@dataclass
class CustomTerrainCfg:
    size: tuple[float, float] = (196.0, 196.0) # Terrain size in meters (x, y)
    meter_per_grid: float = 0.15 # Grid resolution in meters
    SAVE_PATH: Path = CURRENT_DIR / "terrains" / "custom_terrain.usd" # Path to save the generated heightmap
    seed: int = 42  # Random seed for heightmap generation


    roughness: float = 0.05  # Roughness factor for heightmap generation
    hill_scale: float = 768.0  # Scale factor for hill generation
    hill_height: float = 8.0  # Maximum height of hills
    hill_noise_lacunarity: float = 2.5  # Lacunarity of noise for hills
    hill_noise_persistence: float = 0.5  # Persistence of noise for hills
    hill_noise_octaves: int = 32  # Number of octaves for hill noise

    # obstacles: tuple[Obstacle, ...] | None = (
    #     Obstacle(type=ObstacleType.CUBE, scale_range=(2.0, 4.0), num_instances=200),
    #     Obstacle(type=ObstacleType.SPHERE, scale_range=(2.0, 4.0), num_instances=100),
    # )
    obstacles: tuple[Obstacle, ...] | None = None

    # Spawn sampling configuration
    num_points: int = 1024
    sample_radius: float = 0.5
    flatness_threshold: float = 0.5
    max_attempts: int = 1024
    margin: float = 32.0

    def __post_init__(self):
        self.grid_size = (
            int(self.size[1] / self.meter_per_grid),
            int(self.size[0] / self.meter_per_grid)
        )

