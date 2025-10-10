# src/gmsh_runner.py

import argparse
import json
from src.gmsh_geometry import extract_geometry_mask

def main():
    parser = argparse.ArgumentParser(description="Gmsh STEP parser for geometry mask metadata")
    parser.add_argument("--step", type=str, required=True, help="Path to STEP file")
    parser.add_argument("--resolution", type=float, help="Grid resolution in millimeters (model units)")
    parser.add_argument("--flow_region", type=str, choices=["internal", "external"], default="internal", help="Flow context for masking")
    parser.add_argument("--padding_factor", type=int, default=5, help="Number of voxel layers to pad for external")
    parser.add_argument("--no_slip", type=lambda x: x.lower() == "true", default=True, help="Boundary condition: no-slip (True) or slip (False)")
    parser.add_argument("--output", type=str, help="Path to write geometry mask JSON")

    args = parser.parse_args()

    print(f"[INFO] Running Gmsh mask extraction with:")
    print(f"       STEP file       : {args.step}")
    print(f"       Resolution      : {args.resolution}")
    print(f"       Flow region     : {args.flow_region}")
    print(f"       Padding factor  : {args.padding_factor}")
    print(f"       No-slip         : {args.no_slip}")
    print(f"       Output path     : {args.output}")

    result = extract_geometry_mask(
        step_path=args.step,
        resolution=args.resolution,
        flow_region=args.flow_region,
        padding_factor=args.padding_factor,
        no_slip=args.no_slip
    )

    # Post-process boundary voxels based on no_slip flag
    boundary_count = result["geometry_mask_flat"].count(-1)
    print(f"[DEBUG] Found {boundary_count} boundary voxels (value = -1) before applying no_slip policy.")

    if boundary_count > 0:
        if args.no_slip:
            result["geometry_mask_flat"] = [0 if v == -1 else v for v in result["geometry_mask_flat"]]
            print("[INFO] Boundary voxels reclassified as solid (0) due to no_slip = True.")
        else:
            result["geometry_mask_flat"] = [1 if v == -1 else v for v in result["geometry_mask_flat"]]
            print("[INFO] Boundary voxels reclassified as fluid (1) due to no_slip = False.")

        # Remove boundary from mask_encoding
        if "boundary" in result["mask_encoding"]:
            del result["mask_encoding"]["boundary"]

    print("[INFO] Final geometry mask:")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Geometry mask written to: {args.output}")

if __name__ == "__main__":
    main()



