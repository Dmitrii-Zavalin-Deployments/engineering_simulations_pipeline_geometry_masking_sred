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

from src.utils.gmsh_input_check import validate_step_has_volumes
from src.utils.input_validation import load_resolution_profile


def extract_bounding_box_with_gmsh(step_path, resolution=None, flow_region="internal", padding_factor=5):
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
        gmsh.model.add("domain_model")
        gmsh.logger.start()

        validate_step_has_volumes(step_path)
        gmsh.open(str(step_path))

        volumes = gmsh.model.getEntities(3)
        print(f"Found {len(volumes)} volume entities.")
        if not volumes:
            raise ValueError("No volume entities found in STEP file.")

        # Compute global bounding box
        all_bboxes = [gmsh.model.getBoundingBox(dim, tag) for dim, tag in volumes]
        min_x = min(b[0] for b in all_bboxes)
        min_y = min(b[1] for b in all_bboxes)
        min_z = min(b[2] for b in all_bboxes)
        max_x = max(b[3] for b in all_bboxes)
        max_y = max(b[4] for b in all_bboxes)
        max_z = max(b[5] for b in all_bboxes)

        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        min_dim = min(dim_x, dim_y, dim_z)

        print(f"Bounding box: x=({min_x:.2f}, {max_x:.2f}), y=({min_y:.2f}, {max_y:.2f}), z=({min_z:.2f}, {max_z:.2f})")
        print(f"Model dimensions: dx={dim_x:.2f}, dy={dim_y:.2f}, dz={dim_z:.2f}, min_dim={min_dim:.2f}")
        print(f"Using resolution: {resolution:.2f} mm")

        if resolution >= min_dim:
            raise ValueError(
                f"Resolution {resolution:.2f} mm is too large for the model.\n"
                f"The smallest model dimension is {min_dim:.2f} mm, so resolution must be smaller.\n"
                f"Please update 'default_resolution' in 'flow_data.json' to be less than {min_dim:.2f} mm."
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
        total_voxels = nx * ny * nz
        print(f"Voxel grid shape: {shape} → total voxels: {total_voxels}")

        max_voxels = 10_000_000
        safe_resolution_mm = (max_x - min_x) / (max_voxels ** (1/3))
        if total_voxels > max_voxels:
            raise MemoryError(
                f"Voxel grid too large: {total_voxels} exceeds safe limit of {max_voxels}.\n"
                f"Update 'default_resolution' to at least {safe_resolution_mm:.2f} mm."
            )

        # Determine fluid volumes (for internal flow)
        fluid_volume_tags = []
        fluid_is_inside = False

        if flow_region == "internal":
            if len(volumes) > 1:
                sorted_volumes = sorted(volumes, key=lambda v: volume_bbox_volume(gmsh.model.getBoundingBox(*v)))
                fluid_volume_tags = [sorted_volumes[0][1]]
                fluid_is_inside = True
                print(f"Assuming volume tag(s) {fluid_volume_tags} as fluid region(s).")
            else:
                fluid_volume_tags = [volumes[0][1]]
                center_point = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
                center_inside = gmsh.model.isInside(3, fluid_volume_tags[0], center_point)
                fluid_is_inside = not center_inside

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
                        if fluid_is_inside:
                            value = 1 if is_inside else 0
                        else:
                            value = 0 if is_inside else 1
                    elif flow_region == "external":
                        is_inside_any = any(gmsh.model.isInside(3, tag, point) for _, tag in volumes)
                        value = 1 if not is_inside_any else 0
                    else:
                        raise ValueError(f"Unsupported flow_region: {flow_region}")

                    mask.append(value)

        fluid_count = sum(mask)
        solid_count = len(mask) - fluid_count
        print(f"Mask summary → Fluid voxels: {fluid_count}, Solid voxels: {solid_count}")

        print("Sample voxel classifications (X-Major Indexing):")
        sample_indices = [0, total_voxels // 4, total_voxels // 2, 3 * total_voxels // 4, total_voxels - 1]
        for i in [idx for idx in sample_indices if idx < len(mask)]:
            x_idx = i // (ny * nz)
            remainder = i % (ny * nz)
            y_idx = remainder // nz
            z_idx = remainder % nz
            px = min_x + (x_idx + 0.5) * resolution
            py = min_y + (y_idx + 0.5) * resolution
            pz = min_z + (z_idx + 0.5) * resolution
            label = "fluid" if mask[i] == 1 else "solid"
            print(f"Voxel ({x_idx},{y_idx},{z_idx}) at ({px:.2f},{py:.2f},{pz:.2f}) → {label}")

        return {
            "geometry_mask_flat": mask,
            "geometry_mask_shape": shape,
            "mask_encoding": {
                "fluid": 1,
                "solid": 0
            },
            "flattening_order": "x-major"
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def volume_bbox_volume(bbox):
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)


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



