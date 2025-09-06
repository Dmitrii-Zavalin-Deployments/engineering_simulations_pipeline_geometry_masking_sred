# src/geometry_topology_analyzer.py

"""
Geometry Topology Analyzer
--------------------------
Detects thin walls, cavities, and nested voids in voxelized geometry.
Enhances masking accuracy and edge-case resilience.
"""

def analyze_topology(voxel_grid):
    """
    Analyzes voxel grid topology for structural edge cases.

    Parameters:
        voxel_grid (dict): Voxel grid metadata from voxelizer_engine

    Returns:
        Dict[str, Any]: {
            "thin_wall_detected": bool,
            "nested_voids": int,
            "bounding_box_volume": float,
            "grid_resolution": float
        }
    """
    shape = voxel_grid["shape"]
    bbox = voxel_grid["bbox"]
    resolution = voxel_grid["resolution"]

    nx, ny, nz = shape
    min_x, min_y, min_z, max_x, max_y, max_z = bbox

    # Compute bounding box volume
    bbox_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    # Heuristic: thin wall if any dimension is < 3 voxels
    thin_wall_detected = any(dim < 3 for dim in shape)

    # Placeholder: nested voids count (to be replaced with real mesh sampling)
    # For now, assume 1 void if all dimensions > 2, else 0
    nested_voids = 1 if all(dim > 2 for dim in shape) else 0

    return {
        "thin_wall_detected": thin_wall_detected,
        "nested_voids": nested_voids,
        "bounding_box_volume": round(bbox_volume, 6),
        "grid_resolution": resolution
    }



