import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf, UsdPhysics, UsdShade

from .obstacles_generator import append_trimesh_meshes, mesh_placer
from .mesh_loader import make_custom_meshes, make_cube_meshes, make_sphere_meshes
from .custom_terrain_config import CustomTerrainCfg

def save_height_map_to_usd(heightmap, config: CustomTerrainCfg, mesh_placement: dict | None, spawn_points: np.ndarray | None = None) -> None:
    """
    heightmap: (rows, cols) float32 numpy array (높이 데이터)
    usd_path: 저장할 .usd 파일 경로 (예: "terrain.usd")
    meter_per_grid: 격자 간격
    spawn_points: (Optional) (N, 3) numpy array of spawn points for debugging
    """
    rows, cols = heightmap.shape
    
    # 1. USD 스테이지 생성
    stage = Usd.Stage.CreateNew(str(config.SAVE_PATH))
    
    # 기본 UpAxis를 Z축으로 설정 (Isaac Sim 표준)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # 기본 단위를 미터로 설정
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # 1) 루트 Xform 정의
    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)   
    # World를 기본 prim으로
    # 2. Mesh Prim 정의 (/World/terrain 경로에 생성)
    mesh_path = "/World/terrain"
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
    
    
    # -------------------------------------------------------
    # 3. 데이터 생성 (Vectorization으로 고속 처리)
    # -------------------------------------------------------
    
    # (1) 정점(Points) 생성: (x, y, z)
    # x: 행 방향, y: 열 방향
    x_length = (cols-1) * config.meter_per_grid
    y_length = (rows-1) * config.meter_per_grid
    x = np.linspace(-x_length/2, x_length/2, cols)
    y = np.linspace(-y_length/2, y_length/2, rows)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    
    # 정점 배열 합치기 (N, 3)
    points = np.column_stack((xv.ravel(), yv.ravel(), heightmap.ravel()))
    
    # (2) 면(Topology) 생성: 격자를 삼각형으로 연결
    # Vertex 인덱스 계산을 위한 그리드 생성
    # 마지막 행과 열은 사각형의 시작점이 될 수 없으므로 제외
    row_idx = np.arange(rows - 1)
    col_idx = np.arange(cols - 1)
    cv, rv = np.meshgrid(col_idx, row_idx, indexing='xy')
    
    # 현재 점(top-left)의 1차원 인덱스들
    p_idxs = rv * cols + cv
    
    # 사각형 하나를 두 개의 삼각형으로 쪼개기 위한 인덱스 계산
    # Triangle 1: (tl, tr, bl) -> (idx, idx+1, idx+cols)
    # Triangle 2: (tr, br, bl) -> (idx+1, idx+cols+1, idx+cols)
    
    t1 = np.column_stack((p_idxs.ravel(), p_idxs.ravel() + 1, p_idxs.ravel() + cols))
    t2 = np.column_stack((p_idxs.ravel() + 1, p_idxs.ravel() + cols + 1, p_idxs.ravel() + cols))
    
    # 전체 인덱스 리스트 (Face Vertex Indices)
    face_vertex_indices = np.vstack((t1, t2)).reshape(-1)
    
    # 각 면은 삼각형이므로 점 3개씩 가짐 (Face Vertex Counts)
    face_vertex_counts = np.full(len(face_vertex_indices) // 3, 3)

    # -------------------------------------------------------
    # 4. 장해물 설치
    # -------------------------------------------------------
    if config.obstacles is not None:
        if mesh_placement is None:
            raise ValueError("mesh placement is not generated")
        meshes = []
        for obstacles_config in config.obstacles:
            mesh_type = obstacles_config.type
            placement = mesh_placement.get(mesh_type, None)
            if placement is None:
                continue
            match mesh_type:
                case "custom_mesh":
                    if obstacles_config.path is None:
                        raise ValueError("Custom mesh path is not specified in the configuration.")
                    custom_meshes = make_custom_meshes(
                        path=obstacles_config.path,
                        mesh_placement=placement,
                    )
                    meshes.extend(custom_meshes)
                case "cube":
                    cube_meshes = make_cube_meshes(
                        mesh_placement=placement,
                    )
                    meshes.extend(cube_meshes)
                case "sphere":
                    sphere_meshes = make_sphere_meshes(
                        mesh_placement=placement,
                    )
                    meshes.extend(sphere_meshes)
                case _:
                    raise ValueError(f"Unknown obstacle type: {mesh_type}")
            
            # 메쉬들을 지형 메쉬에 추가
            points, face_vertex_indices, face_vertex_counts = append_trimesh_meshes(
                points,
                face_vertex_indices,
                face_vertex_counts,
                meshes,
            )

    # -------------------------------------------------------
    # 5. USD 속성 설정 (pxr 타입으로 변환하여 입력)
    # -------------------------------------------------------
    
    # Points 설정
    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    
    # Topology 설정
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_vertex_counts.astype(np.int32)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_vertex_indices.astype(np.int32)))
    
    # (선택) Extent(경계박스) 계산 - 퍼포먼스를 위해 중요
    mesh_prim.GetExtentAttr().Set(mesh_prim.ComputeExtent(mesh_prim.GetPointsAttr().Get()))
    
    # (1) CollisionAPI 적용: "이것은 충돌 가능한 물체입니다"
    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    
    # (2) MeshCollisionAPI 적용: 충돌 모양 설정
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
    
    # 중요: 지형은 오목(Concave)하므로 Convex Hull로 감싸면 안 됩니다.
    # "none"으로 설정하면 원본 메쉬(Triangle Mesh) 그대로 충돌을 계산합니다.
    mesh_collision_api.CreateApproximationAttr().Set("none")
    
    # -------------------------------------------------------
    # [선택] 물리 재질(Material) 추가 (마찰력 등)
    # -------------------------------------------------------
    # 별도의 Material Prim 생성
    mat_path = "/World/material/terrain"
    material = UsdShade.Material.Define(stage, mat_path) # UsdShade import 필요
    
    # PhysicsMaterialAPI 적용
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(1.0)  # 정지 마찰력
    phys_mat.CreateDynamicFrictionAttr().Set(1.0) # 운동 마찰력
    phys_mat.CreateRestitutionAttr().Set(0.0)     # 반발력(통통 튀는 정도)

    # 메쉬에 재질 바인딩 (Binding)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(material)

    # -------------------------------------------------------
    # 6. (Optional) Debug Mesh Color 설정
    # -------------------------------------------------------
    mesh_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.06, 0.025)])  # 녹색 계열로 설정 (디버그용)


    # -------------------------------------------------------
    # 6. (Optional) Debug Spawn Points
    # -------------------------------------------------------
    if spawn_points is not None:
        spawn_root_path = "/World/debug/spawn_points"
        # Create a scope to organize debug points
        UsdGeom.Scope.Define(stage, spawn_root_path)
        
        for i, point in enumerate(spawn_points):
            # Create a sphere for each spawn point
            sphere_path = f"{spawn_root_path}/point_{i}"
            sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)
            
            # Set position (x, y, z)
            sphere_prim.AddTranslateOp().Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            
            # Set radius (small enough to be a marker)
            sphere_prim.GetRadiusAttr().Set(0.2)
            
            # Set color (Red for visibility)
            sphere_prim.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

    # 5. 저장
    stage.GetRootLayer().Save()
    print(f"Successfully saved USD directly to: {config.SAVE_PATH}")