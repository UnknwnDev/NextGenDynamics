import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf, UsdPhysics, UsdShade
import trimesh

from .custom_terrain_config import CustomTerrainCfg 


def get_height_at_point(config: CustomTerrainCfg, height_map: np.ndarray, coords_2d: np.ndarray) -> np.ndarray:
    """
    Get the height value from the heightmap at given (x, y) coordinates using bilinear interpolation.

    Args:
        config: CustomTerrainCfg
            Configuration containing heightmap parameters.
        height_map: np.ndarray
            2D numpy array representing the heightmap.
        coords_2d: (N, 2) np.ndarray
            Array of (x, y) coordinates in meters.
    Returns:
        Height values at the specified (x, y) locations.
    """
    meter_per_grid = config.meter_per_grid

    x = coords_2d[:, 0]
    y = coords_2d[:, 1]

    grid_x = x / meter_per_grid + 0.5 * (height_map.shape[1] - 1)
    grid_y = y / meter_per_grid + 0.5 * (height_map.shape[0] - 1)

    x0 = np.floor(grid_x).astype(int)
    x1 = np.minimum(x0 + 1, height_map.shape[1] - 1)
    y0 = np.floor(grid_y).astype(int)
    y1 = np.minimum(y0 + 1, height_map.shape[0] - 1)

    sx = grid_x - x0
    sy = grid_y - y0

    h00 = height_map[y0, x0]
    h10 = height_map[y0, x1]
    h01 = height_map[y1, x0]
    h11 = height_map[y1, x1]

    h0 = h00 * (1 - sx) + h10 * sx
    h1 = h01 * (1 - sx) + h11 * sx

    height = h0 * (1 - sy) + h1 * sy

    return height



def mesh_placer(config: CustomTerrainCfg, heightmap) -> dict[str, dict[str, np.ndarray]]:
    """
    Set up a random location and scale for multiple instances of a given mesh types.
    """
    obstacles = config.obstacles

    if obstacles is None:
        return {}
    
    x_range = (-config.size[0]/2, config.size[0]/2)
    y_range = (-config.size[1]/2, config.size[1]/2)

    
    mesh_placement = {}
    for obstacle in obstacles:
        scales = []

        num = obstacle.num_instances

        coords = np.zeros((num, 2))
        coords[:, 0] = np.random.uniform(x_range[0], x_range[1], size=num)  # x
        coords[:, 1] = np.random.uniform(y_range[0], y_range[1], size=num)  # y
        z = get_height_at_point(config, heightmap, coords)

        coords_3d = np.concatenate(
            [coords, z.reshape(-1, 1)], axis=1
        )

        scale_3d = np.zeros((num, 3))
        scale_3d[:, 0] = np.random.uniform(obstacle.scale_range[0], obstacle.scale_range[1], size=num)
        scale_3d[:, 1] = np.random.uniform(obstacle.scale_range[0], obstacle.scale_range[1], size=num)
        scale_3d[:, 2] = np.random.uniform(obstacle.scale_range[0], obstacle.scale_range[1], size=num)

        mesh_placement[obstacle.type] = {
            "positions": np.array(coords_3d, dtype=np.float32),
            "scales": np.array(scale_3d, dtype=np.float32),
        }

    return mesh_placement




# ----------------------------------------------------------------------
# Common helpers for appending trimesh meshes to the terrain mesh
# ----------------------------------------------------------------------

def append_trimesh_meshes(
    points: np.ndarray,
    face_vertex_indices: np.ndarray,
    face_vertex_counts: np.ndarray,
    meshes: list[trimesh.Trimesh],
):
    """
    Append multiple triangle meshes (Trimesh) to an existing triangle mesh.

    Args:
        points: (N, 3) float32 array of existing vertex positions.
        face_vertex_indices: (M,) int32 flat array of triangle indices.
        face_vertex_counts: (F,) int32 array of vertex counts per face (3 for triangles).
        meshes: list of Trimesh objects. Each mesh's vertices are assumed to be
                already in world coordinates.

    Returns:
        new_points, new_face_vertex_indices, new_face_vertex_counts
    """
    new_points = points.astype(np.float32, copy=True)
    new_face_vertex_indices = face_vertex_indices.astype(np.int32, copy=True)
    new_face_vertex_counts = face_vertex_counts.astype(np.int32, copy=True)

    current_vertex_count = new_points.shape[0]

    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int32)

        # Offset indices because we are appending vertices
        offset = current_vertex_count
        mesh_faces_offset = mesh_faces + offset

        # Append vertices and indices
        new_points = np.vstack([new_points, mesh_vertices])
        new_face_vertex_indices = np.concatenate(
            [new_face_vertex_indices, mesh_faces_offset.reshape(-1)]
        )
        new_face_vertex_counts = np.concatenate(
            [new_face_vertex_counts, np.full(mesh_faces.shape[0], 3, dtype=np.int32)]
        )

        current_vertex_count += mesh_vertices.shape[0]

    return new_points, new_face_vertex_indices, new_face_vertex_counts
