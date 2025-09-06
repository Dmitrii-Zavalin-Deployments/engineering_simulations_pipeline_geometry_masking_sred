# src/flow_region_masker.py

"""
Flow Region Masker
------------------
Applies binary fluid/solid classification to a voxel grid based on flow context.
Supports 'internal', 'external', and 'mixed' modes for CFD preprocessing.
"""

def apply_mask(voxel_grid, flow_region):
    """
    Classifies each voxel as fluid (1) or solid (0) based on flow_region.

    Parameters:
        voxel_grid (dict): Voxel grid metadata from voxelizer_engine
        flow_region (str): One of 'internal', 'external', 'mixed'

    Returns:
        List[int]: Flattened binary mask (fluid=1, solid=0)
    """
    shape = voxel_grid["shape"]
    nx, ny, nz = shape

    total_voxels = nx * ny * nz

    if flow_region == "internal":
        # Fluid inside the geometry — simulate cavity
        # For simplicity: center region fluid, rest solid
        mask = [0] * total_voxels
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    i = flatten_index(x, y, z, nx, ny, nz)
                    if 0 < x < nx - 1 and 0 < y < ny - 1 and 0 < z < nz - 1:
                        mask[i] = 1
        return mask

    elif flow_region == "external":
        # Fluid surrounds the geometry — simulate shell
        # For simplicity: outer layer fluid, center solid
        mask = [1] * total_voxels
        for z in range(1, nz - 1):
            for y in range(1, ny - 1):
                for x in range(1, nx - 1):
                    i = flatten_index(x, y, z, nx, ny, nz)
                    mask[i] = 0
        return mask

    elif flow_region == "mixed":
        # Fluid both inside and outside — simulate porous shell
        # For simplicity: all fluid
        return [1] * total_voxels

    else:
        raise ValueError(f"Unsupported flow_region: {flow_region}")


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



