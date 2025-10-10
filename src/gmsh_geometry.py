# src/gmsh_geometry.py

import gmsh
import os
from src.utils.input_validation import load_resolution_profile
from src.gmsh_core import initialize_gmsh_model, compute_bounding_box, volume_bbox_volume


def extract_geometry_mask(step_path, resolution=None, flow_region="internal", padding_factor=5):
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

        fluid_volume_tags = []
        fluid_is_inside = False
        fluid_bbox = None

        if flow_region == "internal":
            if len(volumes) > 1:
                sorted_volumes = sorted(volumes, key=lambda v: volume_bbox_volume(gmsh.model.getBoundingBox(*v)))
                fluid_volume_tags = [sorted_volumes[0][1]]
                fluid_is_inside = True
            else:
                fluid_volume_tags = [volumes[0][1]]
                bbox_volume = volume_bbox_volume(gmsh.model.getBoundingBox(3, fluid_volume_tags[0]))
                total_bbox_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
                fluid_is_inside = bbox_volume < 0.5 * total_bbox_volume

            fluid_bbox = gmsh.model.getBoundingBox(3, fluid_volume_tags[0])
            margin = 0.5 * resolution
            fx_min, fy_min, fz_min, fx_max, fy_max, fz_max = fluid_bbox
            fx_min += margin
            fy_min += margin
            fz_min += margin
            fx_max -= margin
            fy_max -= margin
            fz_max -= margin

        mask = []
        for x_idx in range(nx):
            px = min_x + (x_idx + 0.5) * resolution
            for y_idx in range(ny):
                py = min_y + (y_idx + 0.5) * resolution
                for z_idx in range(nz):
                    pz = min_z + (z_idx + 0.5) * resolution
                    point = [px, py, pz]

                    if flow_region == "internal":
                        is_inside = any(gmsh.model.isInside(3, tag, point) for tag in fluid_volume_tags)
                        if is_inside and fluid_is_inside:
                            in_fluid_bbox = (
                                fx_min <= px <= fx_max and
                                fy_min <= py <= fy_max and
                                fz_min <= pz <= fz_max
                            )
                            value = 1 if in_fluid_bbox else 0
                        elif is_inside:
                            value = 0
                        else:
                            value = 1
                    elif flow_region == "external":
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
                "solid": 0
            },
            "flattening_order": "x-major"
        }

    finally:
        if gmsh.isInitialized():
            gmsh.finalize()



