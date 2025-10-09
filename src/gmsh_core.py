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

# Future helpers can be added here:
# def sort_volumes_by_size(volumes): ...
# def probe_center_point(bbox): ...



