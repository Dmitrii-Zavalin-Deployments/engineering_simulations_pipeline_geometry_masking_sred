# src/gmsh_geometry.py

import gmsh
import os
from datetime import datetime
from src.gmsh_core import (
    initialize_gmsh_model,
    compute_bounding_box,
    volume_bbox_volume,
    classify_voxel_by_corners
)

def validate_flow_region_and_update(model_data, volumes):
    """
    Validates whether the geometry is cube-bounded. If not, updates flow_region to 'external'
    and adds a timestamped comment to flow_region_comment.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = compute_bounding_box(volumes)
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    dim_z = max_z - min_z

    # Check if bounding box forms a rectangular prism
    if dim_x == 0 or dim_y == 0 or dim_z == 0:
        raise ValueError("Invalid geometry: one or more dimensions are zero.")

    # If any dimension differs significantly, treat as non-cube-bounded
    tolerance = 1e-6
    if abs(dim_x - dim_y) > tolerance or abs(dim_y - dim_z) > tolerance or abs(dim_x - dim_z) > tolerance:
        model_data["model_properties"]["flow_region"] = "external"
        timestamp = datetime.utcnow().isoformat() + "Z"
        model_data["model_properties"]["flow_region_comment"] = (
            f"Auto-switched to external due to non-cube geometry at {timestamp}"
        )

def extract_geometry_mask(step_path, resolution=None, flow_region="internal", padding_factor=5, no_slip=True, model_data=None):
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

        # Optional validation hook
        if model_data and flow_region == "internal":
            validate_flow_region_and_update(model_data, volumes)
            flow_region = model_data["model_properties"]["flow_region"]

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

        if len(volumes) > 1:
            sorted_volumes = sorted(volumes, key=lambda v: volume_bbox_volume(gmsh.model.getBoundingBox(*v)))
            fluid_volume_tags = [sorted_volumes[0][1]]
        else:
            fluid_volume_tags = [volumes[0][1]]

        mask = []
        for x_idx in range(nx):
            px = min_x + (x_idx + 0.5) * resolution
            for y_idx in range(ny):
                py = min_y + (y_idx + 0.5) * resolution
                for z_idx in range(nz):
                    pz = min_z + (z_idx + 0.5) * resolution
                    value = classify_voxel_by_corners(px, py, pz, resolution, fluid_volume_tags)
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



