# src/gmsh_geometry.py

import gmsh
import os
from src.gmsh_core import (
    initialize_gmsh_model,
    compute_bounding_box,
    classify_voxel_by_corners
)

def filter_internal_solids(volumes, global_bbox):
    """
    Filters out external solids based on bounding box intersection.
    Returns a list of volume tags that are fully enclosed.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = global_bbox
    internal_tags = []

    for dim, tag in volumes:
        vmin_x, vmin_y, vmin_z, vmax_x, vmax_y, vmax_z = gmsh.model.getBoundingBox(dim, tag)
        touches_boundary = (
            vmin_x <= min_x or vmax_x >= max_x or
            vmin_y <= min_y or vmax_y >= max_y or
            vmin_z <= min_z or vmax_z >= max_z
        )
        if not touches_boundary:
            internal_tags.append(tag)

    return internal_tags

def extract_geometry_mask(step_path, resolution=None, flow_region="internal", padding_factor=5, no_slip=True):
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    if resolution is None:
        raise ValueError("Resolution must be explicitly defined. No default fallback is allowed.")

    gmsh.initialize()
    try:
        initialize_gmsh_model(step_path)
        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise ValueError("No volume entities found in STEP file.")

        min_x, min_y, min_z, max_x, max_y, max_z = compute_bounding_box(volumes)
        dim_x, dim_y, dim_z = max_x - min_x, max_y - min_y, max_z - min_z
        min_dim = min(dim_x, dim_y, dim_z)

        if resolution >= min_dim:
            raise ValueError(
                f"Resolution {resolution:.2f} mm is too large for the model.\n"
                f"The smallest model dimension is {min_dim:.2f} mm, so resolution must be smaller."
            )

        if flow_region == "external":
            pad = padding_factor * resolution
            min_x -= pad
            min_y -= pad
            min_z -= pad
            max_x += pad
            max_y += pad
            max_z += pad

        nx = max(1, int((max_x - min_x) / resolution))
        ny = max(1, int((max_y - min_y) / resolution))
        nz = max(1, int((max_z - min_z) / resolution))
        shape = [nx, ny, nz]

        if flow_region == "internal":
            solid_volume_tags = filter_internal_solids(volumes, (min_x, min_y, min_z, max_x, max_y, max_z))
        else:
            solid_volume_tags = [tag for dim, tag in volumes]

        mask = []
        for x_idx in range(nx):
            px = min_x + (x_idx + 0.5) * resolution
            for y_idx in range(ny):
                py = min_y + (y_idx + 0.5) * resolution
                for z_idx in range(nz):
                    pz = min_z + (z_idx + 0.5) * resolution
                    value = classify_voxel_by_corners(px, py, pz, resolution, solid_volume_tags)
                    mask.append(value)

        return {
            "geometry_mask_flat": mask,
            "geometry_mask_shape": shape,
            "mask_encoding": {
                "fluid": 1,
                "solid": 0,
                "boundary": -1
            },
            "flattening_order": "x-major"
        }

    finally:
        if gmsh.isInitialized():
            gmsh.finalize()



