# src/gmsh_runner.py

# -------------------------------------------------------------------
# Gmsh-based geometry processor for STEP domain extraction pipeline
# -------------------------------------------------------------------

try:
    import gmsh
except ImportError:
    raise RuntimeError("Gmsh module not found. Run: pip install gmsh==4.11.1")

import json
import os

# ✅ Import volume integrity checker
from src.utils.gmsh_input_check import validate_step_has_volumes
# ✅ Import fallback resolution profile loader
from src.utils.input_validation import load_resolution_profile


def extract_bounding_box_with_gmsh(step_path, resolution=None, flow_region="internal", padding_factor=5):
    """
    Parses STEP geometry with Gmsh and returns geometry_definition block
    including voxel grid shape and binary mask.

    Parameters:
        step_path (str or Path): Path to STEP file
        resolution (float or None): Grid resolution in millimeters (model units).
        flow_region (str): Flow context ("internal" or "external")
        padding_factor (int): Number of voxel layers to pad around bounding box for external

    Returns:
        dict: geometry_definition dictionary matching schema
    """
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    if resolution is None:
        try:
            profile = load_resolution_profile()
            resolution = profile.get("default_resolution", {}).get("dx", 2)  # default in mm
        except Exception:
            resolution = 2  # 🔧 Final default fallback in mm

    gmsh.initialize()
    try:
        gmsh.model.add("domain_model")
        gmsh.logger.start()

        validate_step_has_volumes(step_path)
        gmsh.open(str(step_path))

        volumes = gmsh.model.getEntities(3)
        if not volumes:
            raise ValueError("No volume entities found in STEP file.")
        entity_tag = volumes[0][1]

        # Get bounding box of the geometry (in mm)
        min_x, min_y, min_z, max_x, max_y, max_z = gmsh.model.getBoundingBox(3, entity_tag)

        # Calculate model dimensions
        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        min_dim = min(dim_x, dim_y, dim_z)

        # ✅ New check: resolution must be smaller than smallest model dimension
        if resolution >= min_dim:
            raise ValueError(
                f"Resolution {resolution:.2f} mm is too large for the model.\n"
                f"The smallest model dimension is {min_dim:.2f} mm, so resolution must be smaller.\n"
                f"Please update 'default_resolution' in 'flow_data.json' "
                f"(in the engineering_simulation_pipeline folder in Dropbox) "
                f"to be less than {min_dim:.2f} mm."
            )

        # Expand bounding box for external
        if flow_region == "external":
            pad = padding_factor * resolution
            min_x -= pad
            min_y -= pad
            min_z -= pad
            max_x += pad
            max_y += pad
            max_z += pad

        # Compute voxel grid shape
        nx = max(1, int((max_x - min_x) / resolution))
        ny = max(1, int((max_y - min_y) / resolution))
        nz = max(1, int((max_z - min_z) / resolution))
        shape = [nx, ny, nz]

        total_voxels = nx * ny * nz
        print(f"Voxel grid shape: {shape} → total voxels: {total_voxels}")

        max_voxels = 10_000_000
        safe_resolution_mm = (max_x - min_x) / (max_voxels ** (1/3))
        if total_voxels > max_voxels:
            raise MemoryError(
                f"Voxel grid too large: {total_voxels} exceeds safe limit of {max_voxels}.\n"
                f"Model units are in millimeters. The current resolution is likely too fine.\n"
                f"To avoid exceeding memory limits in GitHub Actions, update the "
                f"'default_resolution' value in 'flow_data.json' "
                f"(located in the engineering_simulation_pipeline folder in Dropbox) "
                f"to at least {safe_resolution_mm:.2f} mm "
                f"(which is {safe_resolution_mm:.5f} mm) or larger.\n"
                f"This will reduce the voxel count and keep the job within CI memory limits."
            )

        # ✅ Write resolution advice to JSON
        # advice_path = os.path.join(os.path.dirname(step_path), "geometry_resolution_advice.json")
        # advice_payload = {
        #    "minimum_model_dimension_mm": round(min_dim, 5),
        #    "recommended_minimum_resolution_mm": round(safe_resolution_mm, 5)
        #}
        # with open(advice_path, "w") as f:
        #    json.dump(advice_payload, f, indent=2)
        # print(f"📄 Resolution advice written to: {advice_path}")

        # Build mask using geometry-aware classification
        mask = []
        for z in range(nz):
            pz = min_z + (z + 0.5) * resolution
            for y in range(ny):
                py = min_y + (y + 0.5) * resolution
                for x in range(nx):
                    px = min_x + (x + 0.5) * resolution
                    inside = gmsh.model.isInside(3, entity_tag, [px, py, pz])

                    if flow_region == "internal":
                        value = 1 if not inside else 0
                    elif flow_region == "external":
                        value = 1 if not inside else 0
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
        try:
            gmsh.finalize()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gmsh STEP parser for geometry mask metadata")
    parser.add_argument("--step", type=str, required=True, help="Path to STEP file")
    parser.add_argument("--resolution", type=float, help="Grid resolution in millimeters (model units)")
    parser.add_argument("--flow_region", type=str, choices=["internal", "external"], default="internal", help="Flow context for masking")
    parser.add_argument("--padding_factor", type=int, default=5, help="Number of voxel layers to pad for external")
    parser.add_argument("--output", type=str, help="Path to write geometry mask JSON")

    args = parser.parse_args()

    result = extract_bounding_box_with_gmsh(
        step_path=args.step,
        resolution=args.resolution,
        flow_region=args.flow_region,
        padding_factor=args.padding_factor
    )

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)



