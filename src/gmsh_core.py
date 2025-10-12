import gmsh

def initialize_gmsh_model(step_path):
    """
    Initializes the Gmsh model and loads the STEP file.
    """
    gmsh.model.add("domain_model")
    gmsh.logger.start()
    gmsh.open(str(step_path))
    return gmsh.model

def compute_bounding_box(volumes):
    """
    Computes the global bounding box for a list of volume entities.
    Returns (min_x, min_y, min_z, max_x, max_y, max_z).
    """
    all_bboxes = [gmsh.model.getBoundingBox(dim, tag) for dim, tag in volumes]
    min_x = min(b[0] for b in all_bboxes)
    min_y = min(b[1] for b in all_bboxes)
    min_z = min(b[2] for b in all_bboxes)
    max_x = max(b[3] for b in all_bboxes)
    max_y = max(b[4] for b in all_bboxes)
    max_z = max(b[5] for b in all_bboxes)
    return min_x, min_y, min_z, max_x, max_y, max_z

def get_decimal_precision(resolution):
    """
    Returns the number of decimal places in the resolution.
    For example: 0.5 → 1, 0.125 → 3
    """
    return max(0, len(str(resolution).split('.')[-1].rstrip('0')))

def is_inside_model_geometry(corner, volume_tags, precision):
    """
    Returns True if the corner is inside any of the model's volumes.
    Applies resolution-based rounding to neutralize floating-point drift.
    """
    rounded_corner = [round(c, precision) for c in corner]
    print(f"[DEBUG] Testing corner (rounded to {precision}): {rounded_corner}")
    for tag in volume_tags:
        inside = gmsh.model.isInside(3, tag, rounded_corner)
        print(f"[DEBUG]   Volume tag {tag}: isInside = {inside}")
        if inside:
            return True
    return False

def classify_voxel_by_corners(px, py, pz, resolution, volume_tags):
    """
    Classifies a voxel based on its 8 corners:
    - Returns 0 if all corners are inside geometry (solid)
    - Returns 1 if all corners are outside geometry (fluid)
    - Returns -1 if mixed (boundary)
    """
    precision = get_decimal_precision(resolution)
    print(f"\n[DEBUG] Classifying voxel at center: ({px:.3f}, {py:.3f}, {pz:.3f})")
    half = 0.5 * resolution
    corners = [
        [px - half, py - half, pz - half],  # corner 0
        [px - half, py - half, pz + half],  # corner 1
        [px - half, py + half, pz - half],  # corner 2
        [px - half, py + half, pz + half],  # corner 3
        [px + half, py - half, pz - half],  # corner 4
        [px + half, py - half, pz + half],  # corner 5
        [px + half, py + half, pz - half],  # corner 6
        [px + half, py + half, pz + half],  # corner 7
    ]

    statuses = []
    for i, corner in enumerate(corners):
        result = is_inside_model_geometry(corner, volume_tags, precision)
        statuses.append(result)
        print(f"[DEBUG]   Corner {i}: {corner} → inside = {result}")

    if all(statuses):
        print("[DEBUG] → Classification: SOLID (0)")
        return 0
    elif not any(statuses):
        print("[DEBUG] → Classification: FLUID (1)")
        return 1
    else:
        print("[DEBUG] → Classification: BOUNDARY (-1)")
        return -1

# Future helpers can be added here:
# def sort_volumes_by_size(volumes): ...
# def probe_center_point(bbox): ...



