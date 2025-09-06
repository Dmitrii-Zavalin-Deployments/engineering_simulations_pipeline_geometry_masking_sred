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
from voxelizer_engine import generate_voxel_grid
from flow_region_masker import apply_mask
from fluid_origin_mapper import map_fluid_origins  # ✅ Mixed flow origin tracking
from geometry_topology_analyzer import analyze_topology  # ✅ Structural edge-case detection


def extract_bounding_box_with_gmsh(step_path, resolution=None, flow_region="internal"):
    """
    Parses STEP geometry with Gmsh and returns geometry_definition block
    including voxel grid, binary mask, and optional metadata.

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

        # 🧱 Modular voxelization and masking pipeline
        voxel_grid, shape = generate_voxel_grid(step_path, resolution)
        mask = apply_mask(voxel_grid, flow_region)

        # ✅ Track fluid origin metadata only for 'mixed' flow regions
        origin_map = map_fluid_origins(mask, voxel_grid, flow_region) if flow_region == "mixed" else {}

        # ✅ Analyze geometry topology for structural anomalies
        topology_flags = analyze_topology(voxel_grid)

        return {
            "geometry_mask_flat": mask,
            "geometry_mask_shape": shape,
            "mask_encoding": {
                "fluid": 1,
                "solid": 0
            },
            "flattening_order": "x-major",
            "fluid_origin": origin_map,
            "topology_flags": topology_flags
        }
    finally:
        gmsh.finalize()  # ✅ Guaranteed shutdown


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



