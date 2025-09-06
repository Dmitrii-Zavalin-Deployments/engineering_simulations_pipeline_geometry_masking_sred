# src/fluid_origin_mapper.py

"""
Fluid Origin Mapper
-------------------
Generates metadata map for 'mixed' flow origin tracking.
Distinguishes fluid voxels by internal vs external classification.
"""

def map_fluid_origins(mask, voxel_grid, flow_region):
    """
    Maps fluid voxel indices to their origin type for 'mixed' flow regions.

    Parameters:
        mask (List[int]): Flattened binary mask (fluid=1, solid=0)
        voxel_grid (dict): Voxel grid metadata from voxelizer_engine
        flow_region (str): One of 'internal', 'external', 'mixed'

    Returns:
        Dict[str, List[int]]: {
            "internal": [...],
            "external": [...]
        } or empty dict if not mixed
    """
    if flow_region != "mixed":
        return {}

    shape = voxel_grid["shape"]
    nx, ny, nz = shape

    internal = []
    external = []

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                i = flatten_index(x, y, z, nx, ny, nz)
                if mask[i] != 1:
                    continue

                # Heuristic: center voxels → internal, edge voxels → external
                if 0 < x < nx - 1 and 0 < y < ny - 1 and 0 < z < nz - 1:
                    internal.append(i)
                else:
                    external.append(i)

    return {
        "internal": internal,
        "external": external
    }


def flatten_index(x, y, z, nx, ny, nz):
    """
    Converts 3D voxel coordinates to flattened index using x-major order.

    Parameters:
        x, y, z (int): Voxel coordinates
        nx, ny, nz (int): Grid dimensions

    Returns:
        int: Flattened index
    """
    return x + y * nx + z * nx * ny



