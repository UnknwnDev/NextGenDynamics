import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf, UsdPhysics, UsdShade
import trimesh


def _to_vec3_array(values) -> np.ndarray:
    """
    Normalize input to a (N, 3) float32 array.
    Accepts:
      - (3,)
      - (N, 3)
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"Transition must have shape (3,) or (N, 3). Got {arr.shape}.")
        arr = arr[None, :]  # (1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Transition must have shape (N, 3). Got {arr.shape}.")
    return arr


def _normalize_scale_to_xyz(scale, num_instances: int) -> np.ndarray:
    """
    Normalize various scale representations into (num_instances, 3) float32.

    Accepts:
      - scalar: same uniform xyz scale for all instances
      - (num_instances,): per-instance uniform xyz scale
      - (3,): one (sx, sy, sz) used for all instances
      - (1, 3): same as above
      - (num_instances, 3): per-instance (sx, sy, sz)
    """
    s = np.asarray(scale, dtype=np.float32)

    if s.ndim == 0:
        # single scalar -> broadcast to (N, 3)
        return np.full((num_instances, 3), float(s), dtype=np.float32)

    if s.ndim == 1:
        if s.size == num_instances:
            # per-instance scalar -> (N, 3) with repeated components
            return np.repeat(s[:, None], 3, axis=1)
        if s.size == 3:
            # one (sx, sy, sz) for all instances
            return np.repeat(s[None, :], num_instances, axis=0)
        raise ValueError(
            f"scale 1D array must have length 3 or num_instances ({num_instances}). "
            f"Got length {s.size}."
        )

    if s.ndim == 2:
        if s.shape == (num_instances, 3):
            return s
        if s.shape == (1, 3):
            return np.repeat(s, num_instances, axis=0)

    raise ValueError(
        f"scale must be scalar, (num_instances,), (3,), (1, 3), or (num_instances, 3). "
        f"Got shape {s.shape} with num_instances={num_instances}."
    )

def make_sphere_meshes(
    mesh_placement: dict,
    subdivisions: int = 2,
) -> list[trimesh.Trimesh]:
    """
    Create icosphere-based obstacles.

    transitions: (N, 3) or (3,)
    scale:   scalar, (N,), (3,), (1, 3) or (N, 3)
             -> normalized to (N, 3) as final radii along x, y, z.

    Result: for each center, an ellipsoid with radii given by scale row.
    """
    transitions = mesh_placement["positions"]
    scale = mesh_placement["scales"]

    transitions_arr = _to_vec3_array(transitions)
    num_instances = transitions_arr.shape[0]
    scale_arr = _normalize_scale_to_xyz(scale, num_instances)

    meshes: list[trimesh.Trimesh] = []

    for center, s_xyz in zip(transitions_arr, scale_arr):
        # Start from unit sphere (radius=1.0)
        sphere = trimesh.creation.icosphere(
            subdivisions=subdivisions,
            radius=1.0,
        )
        # Non-uniform scale to get radii = s_xyz
        sx, sy, sz = float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])
        scale_mat = np.diag([sx, sy, sz, 1.0])
        sphere.apply_transform(scale_mat)

        # Move to center
        sphere.apply_translation(center.astype(float))

        meshes.append(sphere)

    return meshes

def make_cube_meshes(
    mesh_placement: dict,
) -> list[trimesh.Trimesh]:
    """
    Create axis-aligned box (cuboid) obstacles.

    transitions: (N, 3) or (3,)
    scale:   scalar, (N,), (3,), (1, 3) or (N, 3)
             -> normalized to (N, 3) and interpreted as full extents
                (edge lengths along x, y, z) for each box.
    """
    transitions = mesh_placement["positions"]
    scale = mesh_placement["scales"]

    transitions_arr = _to_vec3_array(transitions)
    num_instances = transitions_arr.shape[0]
    scale_arr = _normalize_scale_to_xyz(scale, num_instances)

    meshes: list[trimesh.Trimesh] = []

    for center, extents in zip(transitions_arr, scale_arr):
        # Box with given extents centered at origin
        box = trimesh.creation.box(extents=extents.astype(float))
        # Move to center
        box.apply_translation(center.astype(float))
        meshes.append(box)

    return meshes


def make_custom_meshes(
    path: str,
    mesh_placement: dict,
) -> list[trimesh.Trimesh]:
    """
    Create multiple instances of the same OBJ mesh with per-instance
    translation and non-uniform scale.

    path:          OBJ file path.
    translations:  (N, 3) or (3,)
    scale:         scalar, (N,), (3,), (1, 3), or (N, 3)
                   -> normalized to (N, 3) and used as per-axis scale factors.
    """
    translations = mesh_placement["positions"]
    scale = mesh_placement["scales"]

    translations_arr = _to_vec3_array(translations)
    num_instances = translations_arr.shape[0]
    scale_arr = _normalize_scale_to_xyz(scale, num_instances)

    # Load OBJ once
    loaded = trimesh.load(path, force="mesh")

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"OBJ file '{path}' contains no geometry.")
        base_mesh: trimesh.Trimesh = trimesh.util.concatenate(
            list(loaded.geometry.values())
        )
    elif isinstance(loaded, trimesh.Trimesh):
        base_mesh = loaded
    else:
        raise TypeError(f"Unexpected type from trimesh.load: {type(loaded)}")

    meshes: list[trimesh.Trimesh] = []

    for t, s_xyz in zip(translations_arr, scale_arr):
        mesh_i: trimesh.Trimesh = base_mesh.copy()

        # Non-uniform scaling about the origin
        sx, sy, sz = float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])
        scale_mat = np.diag([sx, sy, sz, 1.0])
        mesh_i.apply_transform(scale_mat)

        # Then translate to target position
        mesh_i.apply_translation(t.astype(float))

        meshes.append(mesh_i)

    return meshes


def get_obstacle_radius(obstacle_type: str, scales: np.ndarray, base_radius: float | None = None) -> np.ndarray:
    """
    Calculate the collision radius for obstacles based on type and scale.
    
    Args:
        obstacle_type: "cube", "sphere", or custom type.
        scales: (N, 3) array of scales.
        base_radius: Optional base radius for custom types. If None, defaults to 1.0.
    """
    if obstacle_type == "cube":
        # Cube scale is full extent. Radius is distance from center to corner in XY plane.
        # extent/2 is half extent. norm(extent/2) = norm(extent)/2
        return np.linalg.norm(scales[:, :2], axis=1) / 2.0
    elif obstacle_type == "sphere":
        # Sphere scale is radius. Use max radius in XY plane.
        return np.max(scales[:, :2], axis=1)
    else:
        # Custom types
        # If base_radius is provided, use it. Otherwise assume 1.0 (unit size)
        # Use max scale in XY plane
        r = base_radius if base_radius is not None else 1.0
        return np.max(scales[:, :2], axis=1) * r

