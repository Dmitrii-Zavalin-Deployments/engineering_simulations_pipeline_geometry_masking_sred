import pytest
import numpy as np

# Adjust this import to point to your actual module
from src.generate_blender_mesh_format import get_1d_index

@pytest.mark.parametrize("grid_shape,expected_boundary_count", [
    ([2, 2, 2], 8),
    ([3, 3, 3], 26),
    ([4, 4, 4], 56),
    ([3, 3, 1], 9),
    ([3, 1, 3], 9),
    ([1, 3, 3], 9),
])
def test_boundary_index_detection(grid_shape, expected_boundary_count):
    nz, ny, nx = grid_shape
    num_nodes = nz * ny * nx
    all_indices = set()
    boundary_indices = set()

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                idx = get_1d_index(ix, iy, iz, nx, ny)
                all_indices.add(idx)
                if ix in [0, nx - 1] or iy in [0, ny - 1] or iz in [0, nz - 1]:
                    boundary_indices.add(idx)

    assert boundary_indices.issubset(all_indices)
    assert len(boundary_indices) == expected_boundary_count



