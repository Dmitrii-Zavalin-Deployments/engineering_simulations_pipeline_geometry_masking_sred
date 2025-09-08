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

# 🧱 Import modular masking components
from shape_builder import build_shape_from_geometry
from semantic_mask_generator import generate_semantic_stub


def extract_bounding_box_with_gmsh(step_path, resolution=None, flow_region="internal"):
    """
    Parses STEP geometry with Gmsh and returns geometry_definition block
    including voxel grid shape and binary mask.

    Parameters:
        step_path (str or Path): Path to STEP file
        resolution (float or None): Grid resolution in meters. If None, fallback profile will be used.
        flow_region (str): Flow context ("internal", "external", "mixed")

    Returns:
        dict: geometry_definition dictionary matching schema
    """
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    if resolution is None:
        # 🧩 Load from fallback profile
        try:
            profile = load_resolution_profile()
            resolution = profile.get("default_resolution", {}).get("dx", 0.01)
        except Exception:
            resolution = 0.01  # 🔧 Final default fallback

    gmsh.initialize()  # ✅ Defensive session entry
    try:
        gmsh.model.add("domain_model")
        gmsh.logger.start()

        validate_step_has_volumes(step_path)
        gmsh.open(str(step_path))  # ✅ Ensure fileName is str

        # ✅ Compute voxel grid shape from geometry
        shape = build_shape_from_geometry(step_path, resolution)

        # ✅ Log voxel grid shape and total size
        total_voxels = shape[0] * shape[1] * shape[2]
        print(f"Voxel grid shape: {shape} → total voxels: {total_voxels}")

        # ✅ Safety check to prevent memory exhaustion
        max_voxels = 10_000_000
        if total_voxels > max_voxels:
            raise MemoryError(f"Voxel grid too large: {total_voxels} voxels exceeds safe limit of {max_voxels}")

        # ✅ Generate semantic mask using geometry-aware stub logic
        geometry_definition = generate_semantic_stub(step_path, resolution, flow_region)

        return geometry_definition
    finally:
        try:
            gmsh.finalize()  # ✅ Safe shutdown
        except Exception:
            pass  # Prevent crash if already finalized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gmsh STEP parser for geometry mask metadata")
    parser.add_argument("--step", type=str, required=True, help="Path to STEP file")
    parser.add_argument("--resolution", type=float, help="Grid resolution in meters")
    parser.add_argument("--flow_region", type=str, choices=["internal", "external", "mixed"], default="internal", help="Flow context for masking")
    parser.add_argument("--output", type=str, help="Path to write geometry mask JSON")

    args = parser.parse_args()

    result = extract_bounding_box_with_gmsh(
        step_path=args.step,
        resolution=args.resolution,
        flow_region=args.flow_region
    )

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)



