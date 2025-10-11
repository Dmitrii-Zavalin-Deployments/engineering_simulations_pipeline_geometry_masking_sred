# src/gmsh_core.py

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

def volume_bbox_volume(bbox):
    """
    Computes the volume of a bounding box.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

def is_point_inside_bbox(px, py, pz, bbox):
    """
    Checks whether a point (px, py, pz) is inside the given bounding box.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    return (min_x <= px <= max_x and
            min_y <= py <= max_y and
            min_z <= pz <= max_z)

def shrink_bbox(bbox, margin):
    """
    Shrinks a bounding box inward by a given margin.
    Returns a new bounding box with reduced dimensions.
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    return (
        min_x + margin, min_y + margin, min_z + margin,
        max_x - margin, max_y - margin, max_z - margin
    )

def bbox_center(bbox):
    """
    Computes the center point of a bounding box.
    Returns a list [cx, cy, cz].
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    return [(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2]

def is_inside_any_solid(corner, solid_volume_tags):
    """
    Returns True if the corner is inside any of the solid volumes.
    """
    for tag in solid_volume_tags:
        if gmsh.model.isInside(3, tag, corner):
            return True
    return False

def classify_voxel_by_corners(px, py, pz, resolution, solid_volume_tags):
    """
    Classifies a voxel based on its 8 corners:
    - Returns 0 if all corners are inside solid
    - Returns 1 if all corners are outside solid (fluid)
    - Returns -1 if mixed (boundary)
    """
    half = 0.5 * resolution
    corners = [
        [px - half, py - half, pz - half],
        [px - half, py -half, pz + half],
        [px - half, py + half, pz - half],
        [px - half, py + half, pz + half],
        [px + half, py - half, pz - half],
        [px + half, py - half, pz + half],
        [px + half, py + half, pz - half],
        [px + half, py + half, pz + half],
    ]

    statuses = [is_inside_any_solid(corner, solid_volume_tags) for corner in corners]
    if all(statuses):
        return 0  # solid
    elif not any(statuses):
        return 1  # fluid
    else:
        return -1  # boundary


# Future helpers can be added here:
# def sort_volumes_by_size(volumes): ...
# def probe_center_point(bbox): ...



