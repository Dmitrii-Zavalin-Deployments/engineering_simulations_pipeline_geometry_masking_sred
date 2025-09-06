# src/flow_region_masker.py

"""
Flow Region Masker
------------------
Applies geometry-aware fluid/solid classification to a voxel grid based on flow context.
Supports 'internal', 'external', and 'mixed' modes for CFD preprocessing.
Uses Gmsh point containment to determine voxel classification.
"""

def apply_mask(voxel_grid, flow_region):
    """
    Classifies each voxel as fluid (1) or solid (0) based on flow_region.

    Parameters:
        voxel_grid (dict): Voxel grid metadata from shape_builder or voxelizer_engine.
                           Must include 'source_path' to reopen STEP geometry.
        flow_region (str): One of 'internal', 'external', 'mixed'

    Returns:
        List[int]: Flattened binary mask (fluid=1, solid=0)
    """
    import gmsh

    shape = voxel_grid["shape"]
    bbox = voxel_grid["bbox"]
    resolution = voxel_grid["resolution"]
    source_path = voxel_grid.get("source_path")
    if not source_path:
        raise ValueError("Missing 'source_path' in voxel_grid metadata.")

    nx, ny, nz = shape
    min_x, min_y, min_z = bbox[0], bbox[1], bbox[2]

    gmsh.initialize()
    try:
        gmsh.model.add("masking_model")
        gmsh.open(source_path)

        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise ValueError("No volume entities found in geometry.")

        entity_tag = volumes[0][1]
        mask = []

        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    i = flatten_index(x, y, z, nx, ny, nz)

                    # Compute voxel center coordinates
                    px = min_x + (x + 0.5) * resolution
                    py = min_y + (y + 0.5) * resolution
                    pz = min_z + (z + 0.5) * resolution

                    inside = gmsh.model.isInside(3, entity_tag, [px, py, pz])

                    if flow_region == "internal":
                        value = 1 if inside else 0
                    elif flow_region == "external":
                        value = 1 if not inside else 0
                    elif flow_region == "mixed":
                        value = 1  # All fluid — origin tracked separately
                    else:
                        raise ValueError(f"Unsupported flow_region: {flow_region}")

                    mask.append(value)

        return mask
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


def flatten_index(x, y, z, nx, ny, nz):
    """
    Converts 3D voxel coordinates to flattened index using x-major order.

    Parameters:
        x, y, z (int): Voxel coordinates
        nx, ny, nz (int): Grid dimensions

    Returns:
        int: Flattened index
    """
    return x + y * nx + z * nx * ny



