import sys
from pathlib import Path

from .obstacles_generator import mesh_placer

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))  # Adjust the path as needed

from .height_map_generator import generate_height_map
from .height_map_to_usd import save_height_map_to_usd
from .custom_terrain_config import CustomTerrainCfg
from .spawnpoint_sampler import spawn_point_sampler

is_initialized = False

class CustomTerrainGenerator:
    def __init__(self, config: CustomTerrainCfg):
        self.config = config
        self.height_map = None
        self.obstacle_placement = None
                
    def initialize(self):
        if self.config is None:
            raise ValueError("CustomTerrainGenerator not initialized.")
        
        global is_initialized

        if self.height_map is not None or is_initialized == True:
            return str(self.config.SAVE_PATH)       
        
        is_initialized = True

        print("Generating height_map...")
        self._generate_height_map()

        if self.config.obstacles is not None:
            print("placing obstacles...")
            self.obstacle_placement = mesh_placer(self.config, self.height_map)
        
        print("Sampling spawn points...")
        self.spawn_points = spawn_point_sampler(
            self.height_map,
            self.obstacle_placement,
            self.config
        )

        print("Saving height_map to USD...")
        self._export_height_map_usd()
        print("")

        print("Map generation complete.")

    def _generate_height_map(self):
        """Generates a flat height_map based on the configuration."""
        if self.config is None:
            raise ValueError("CustomTerrainGenerator not initialized. Call initialize() first.")
        
        height_map = generate_height_map(self.config)
        self.height_map = height_map

    def _export_height_map_usd(self):
        """Saves the generated height_map to a file."""
        if self.height_map is not None:
            spawn_points = getattr(self, 'spawn_points', None)
            save_height_map_to_usd(self.height_map, self.config, self.obstacle_placement) # , spawn_points)
        else:
            raise ValueError("height_map not generated yet. Call _generate_height_map() first.")

    def get_terrain_path(self) -> str:
        """Returns the path to the height_map file.
        Returns:
            str: Path to the generated height_map file.
        """
        return str(self.config.SAVE_PATH)


terrain_config = CustomTerrainCfg()
terrain_generator = CustomTerrainGenerator(config=terrain_config)

if __name__ == "__main__":
    terrain_generator.initialize()