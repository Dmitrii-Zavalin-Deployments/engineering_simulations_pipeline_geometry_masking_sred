# src/gmsh_geometry.py

import gmsh
import os
import json
from datetime import datetime
from src.gmsh_core import (
    initialize_gmsh_model,
    compute_bounding_box,
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

    if dim_x == 0 or dim_y == 0 or dim_z == 0:
        raise ValueError("Invalid geometry: one or more dimensions are zero.")

    tolerance = 1e-6
    if abs(dim_x - dim_y) > tolerance or abs(dim_y - dim_z) > tolerance or abs(dim_x - dim_z) > tolerance:
        model_data["model_properties"]["flow_region"] = "external"
        timestamp = datetime.utcnow().isoformat() + "Z"
        model_data["model_properties"]["flow_region_comment"] = (
            f"Auto-switched to external due to non-cube geometry at {timestamp}"
        )

def extract_geometry_mask(step_path, resolution=None, flow_region="internal", padding_factor=5, no_slip=True, model_data=None, debug=False):
    if debug:
        print(f"[DEBUG] STEP path received: {step_path}")
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    if debug:
        print(f"[DEBUG] Resolution received: {resolution}")
    if resolution is None:
        raise ValueError("Resolution must be explicitly defined. No default fallback is allowed.")

    if debug:
        print("[DEBUG] Initializing Gmsh...")
    gmsh.initialize()
    try:
        if debug:
            print("[DEBUG] Loading STEP model...")
        initialize_gmsh_model(step_path)

        if debug:
            print("[DEBUG] Fetching volume entities...")
        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise ValueError("No volume entities found in STEP file.")
        if debug:
            print(f"[DEBUG] Volume count: {len(volumes)}")

        if model_data and flow_region == "internal":
            if debug:
                print("[DEBUG] Validating flow region based on geometry...")
            validate_flow_region_and_update(model_data, volumes)
            flow_region = model_data["model_properties"]["flow_region"]
            if debug:
                print(f"[DEBUG] Flow region after validation: {flow_region}")

        if debug:
            print("[DEBUG] Computing bounding box...")
        min_x, min_y, min_z, max_x, max_y, max_z = compute_bounding_box(volumes)
        if debug:
            print(f"[DEBUG] Bounding box: min=({min_x:.3f}, {min_y:.3f}, {min_z:.3f}), max=({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")

        dim_x, dim_y, dim_z = max_x - min_x, max_y - min_y, max_z - min_z
        min_dim = min(dim_x, dim_y, dim_z)
        if debug:
            print(f"[DEBUG] Dimensions: dx={dim_x:.3f}, dy={dim_y:.3f}, dz={dim_z:.3f}, min_dim={min_dim:.3f}")

        if resolution >= min_dim:
            raise ValueError(
                f"Resolution {resolution:.2f} mm is too large for the model.\n"
                f"The smallest model dimension is {min_dim:.2f} mm, so resolution must be smaller."
            )

        if flow_region == "external":
            pad = padding_factor * resolution
            if debug:
                print(f"[DEBUG] Applying external padding: {pad:.3f}")
            min_x -= pad
            min_y -= pad
            min_z -= pad
            max_x += pad
            max_y += pad
            max_z += pad
            if debug:
                print(f"[DEBUG] Padded bounding box: min=({min_x:.3f}, {min_y:.3f}, {min_z:.3f}), max=({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")

        nx = max(1, int((max_x - min_x) / resolution))
        ny = max(1, int((max_y - min_y) / resolution))
        nz = max(1, int((max_z - min_z) / resolution))
        shape = [nx, ny, nz]
        if debug:
            print(f"[DEBUG] Grid shape: nx={nx}, ny={ny}, nz={nz}")

        mask = []
        volume_tags = [v[1] for v in volumes]
        if debug:
            print(f"[DEBUG] Volume tags: {volume_tags}")

        for x_idx in range(nx):
            px = min_x + (x_idx + 0.5) * resolution
            for y_idx in range(ny):
                py = min_y + (y_idx + 0.5) * resolution
                for z_idx in range(nz):
                    pz = min_z + (z_idx + 0.5) * resolution
                    if debug:
                        print(f"\n[DEBUG] Voxel index: ({x_idx}, {y_idx}, {z_idx}) → center=({px:.3f}, {py:.3f}, {pz:.3f})")
                    value = classify_voxel_by_corners(px, py, pz, resolution, volume_tags)
                    mask.append(value)

        result = {
            "geometry_mask_flat": mask,
            "geometry_mask_shape": shape,
            "mask_encoding": {
                "fluid": 1,
                "solid": 0,
                "boundary": -1
            },
            "flattening_order": "x-major"
        }

        if debug:
            print("\n--- DEBUG: Geometry Mask Output ---")
            print(json.dumps(result, indent=2))

        return result

    finally:
        if gmsh.isInitialized():
            if debug:
                print("[DEBUG] Finalizing Gmsh...")
            gmsh.finalize()



