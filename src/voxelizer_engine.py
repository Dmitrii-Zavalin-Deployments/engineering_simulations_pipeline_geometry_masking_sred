# src/voxelizer_engine.py

"""
Voxelizer Engine
----------------
Generates a voxel grid placeholder and shape from STEP geometry using bounding box and resolution.
This module replaces stubbed shape logic in gmsh_runner.py and feeds downstream masking modules.
"""

def generate_voxel_grid(step_path, resolution):
    """
    Generates a voxel grid placeholder and shape from STEP geometry.

    Parameters:
        step_path (str): Path to STEP file
        resolution (float): Grid resolution in meters

    Returns:
        tuple: (voxel_grid, shape) where:
            - voxel_grid: dict with bounding box and metadata
            - shape: [nx, ny, nz] voxel dimensions
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.model.add("voxelizer_model")
        gmsh.open(step_path)

        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise ValueError("No volume entities found in STEP file.")

        entity_tag = volumes[0][1]
        min_x, min_y, min_z, max_x, max_y, max_z = gmsh.model.getBoundingBox(3, entity_tag)

        if (max_x - min_x) <= 0 or (max_y - min_y) <= 0 or (max_z - min_z) <= 0:
            raise ValueError("Invalid bounding box dimensions.")

        nx = max(1, int((max_x - min_x) / resolution))
        ny = max(1, int((max_y - min_y) / resolution))
        nz = max(1, int((max_z - min_z) / resolution))

        shape = [nx, ny, nz]

        # Placeholder voxel grid metadata
        voxel_grid = {
            "bbox": [min_x, min_y, min_z, max_x, max_y, max_z],
            "resolution": resolution,
            "shape": shape,
            "grid": None  # Future: 3D array or spatial index
        }

        return voxel_grid, shape
    finally:
        gmsh.finalize()



