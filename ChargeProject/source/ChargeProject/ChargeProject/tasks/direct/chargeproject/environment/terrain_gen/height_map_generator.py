import numpy as np
import sys
from pathlib import Path
import opensimplex

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))  # Adjust the path as needed

from terrain_gen.custom_terrain_config import CustomTerrainCfg

def generate_rough_map(config: CustomTerrainCfg) -> np.ndarray:
    """Generates a rough heightmap.
    Returns:
        np.ndarray: Generated heightmap as a 2D numpy array.
    """
    np.random.seed(config.seed)
    rough_map = np.random.random(config.grid_size) * config.roughness

    return rough_map


def generate_hills_map(config: CustomTerrainCfg) -> np.ndarray:
    """Generates a hills heightmap.
    Returns:
        np.ndarray: Generated heightmap as a 2D numpy array.
    """
    simplex = opensimplex.OpenSimplex(seed=config.seed)
    hills_map = np.zeros(config.grid_size)

    xs = np.arange(config.grid_size[1], dtype=np.float32)
    ys = np.arange(config.grid_size[0], dtype=np.float32)

    frequency = 1.0 / config.hill_scale
    amplitude = 1.0

    for _ in range(config.hill_noise_octaves):
        # 스케일 적용
        nx = xs * frequency
        ny = ys * frequency

        noise = opensimplex.noise2array(nx, ny)  # shape: (len(y), len(x))

        hills_map += noise.astype(np.float32) * amplitude

        frequency *= config.hill_noise_lacunarity
        amplitude *= config.hill_noise_persistence


    return hills_map * config.hill_height

def generate_height_map(config: CustomTerrainCfg) -> np.ndarray:
    """Generates a flat heightmap.
    Returns:
        np.ndarray: Generated heightmap as a 2D numpy array.
    """
    rough_map = generate_rough_map(config)
    hills_map = generate_hills_map(config)
    heightmap = rough_map + hills_map

    return heightmap
