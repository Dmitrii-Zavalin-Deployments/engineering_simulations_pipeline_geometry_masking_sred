# src/shape_builder.py

"""
Shape Builder
-------------
Computes voxel grid shape [nx, ny, nz] from STEP geometry using bounding box and resolution.
Lightweight utility for geometry-aware masking and stub generation.
"""

def build_shape_from_geometry(step_path, resolution):
    """
    Computes voxel grid shape from STEP geometry.

    Parameters:
        step_path (str): Path to STEP file
        resolution (float): Grid resolution in meters

    Returns:
        List[int]: [nx, ny, nz] voxel grid dimensions
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.model.add("shape_builder_model")
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

        return [nx, ny, nz]
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass



