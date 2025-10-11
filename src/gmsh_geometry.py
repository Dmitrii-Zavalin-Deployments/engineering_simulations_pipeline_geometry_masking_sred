# src/gmsh_geometry.py

import gmsh
import os
from src.utils.input_validation import load_resolution_profile
from src.gmsh_core import (
    initialize_gmsh_model,
    compute_bounding_box,
    classify_voxel_by_corners
)

def identify_fluid_volumes(step_path):
    """
    Loads a STEP file into Gmsh and identifies fluid volumes (holes) based on bounding box comparison.
    Returns a list of volume tags that are fully enclosed and likely fluid regions.

    Strategy:
    1. Load the STEP file into Gmsh
    2. Get all 3D volumes (outer shell, internal holes, disconnected parts)
    3. Compute bounding boxes for each volume
    4. Compare each volume’s bounding box to the global bounding box
       - If a volume touches the outer bounding box → likely external shell
       - If a volume is fully enclosed → likely internal hole
    5. Return only fluid volume tags
    """
    gmsh.initialize()
    gmsh.model.add("fluid_volume_filter")
    gmsh.open(step_path)

    volumes = gmsh.model.getEntities(3)
    if not volumes:
        gmsh.finalize()
        raise ValueError("No 3D volumes found in STEP file.")

    all_bboxes = [gmsh.model.getBoundingBox(dim, tag) for dim, tag in volumes]
    min_x = min(b[0] for b in all_bboxes)
    min_y = min(b[1] for b in all_bboxes)
    min_z = min(b[2] for b in all_bboxes)
    max_x = max(b[3] for b in all_bboxes)
    max_y = max(b[4] for b in all_bboxes)
    max_z = max(b[5] for b in all_bboxes)
    (min_x, min_y, min_z, max_x, max_y, max_z)

    fluid_tags = []
    for dim, tag in volumes:
        vmin_x, vmin_y, vmin_z, vmax_x, vmax_y, vmax_z = gmsh.model.getBoundingBox(dim, tag)
        touches_outer_bbox = (
            vmin_x <= min_x or vmax_x >= max_x or
            vmin_y <= min_y or vmax_y >= max_y or
            vmin_z <= min_z or vmax_z >= max_z
        )
        if not touches_outer_bbox:
            fluid_tags.append(tag)

    gmsh.finalize()
    return fluid_tags

def extract_geometry_mask(step_path, resolution=None, flow_region="internal", padding_factor=5, no_slip=True):
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    if resolution is None:
        try:
            profile = load_resolution_profile()
            resolution = profile.get("default_resolution", {}).get("dx", 2)
        except Exception:
            resolution = 2

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
            if len(volumes) > 1:
                fluid_volume_tags = identify_fluid_volumes(step_path)
            else:
                fluid_volume_tags = [volumes[0][1]]

        mask = []
        for x_idx in range(nx):
            px = min_x + (x_idx + 0.5) * resolution
            for y_idx in range(ny):
                py = min_y + (y_idx + 0.5) * resolution
                for z_idx in range(nz):
                    pz = min_z + (z_idx + 0.5) * resolution

                    if flow_region == "internal":
                        value = classify_voxel_by_corners(px, py, pz, resolution, fluid_volume_tags)
                    elif flow_region == "external":
                        point = [px, py, pz]
                        is_inside_any = any(gmsh.model.isInside(3, tag, point) for _, tag in volumes)
                        value = 1 if not is_inside_any else 0
                    else:
                        raise ValueError(f"Unsupported flow_region: {flow_region}")

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



