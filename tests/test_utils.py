import pytest

# Import your utility function
from src.generate_blender_mesh_format import get_1d_index

def test_get_1d_index_consistency():
    # Grid shape: z = 2, y = 3, x = 4
    nx, ny, nz = 4, 3, 2

    # Flattened index should range from 0 to 23
    total_nodes = nx * ny * nz
    visited = set()

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                idx = get_1d_index(ix, iy, iz, nx, ny)
                # Check index is within expected bounds
                assert 0 <= idx < total_nodes
                visited.add(idx)

    # Every index from 0 to N-1 should be present
    assert len(visited) == total_nodes
    assert visited == set(range(total_nodes))

def test_get_1d_index_specific_case():
    # Simple 2×2×2 grid (nx=2, ny=2, nz=2)
    nx, ny = 2, 2

    # Manually validated indices
    assert get_1d_index(0, 0, 0, nx, ny) == 0
    assert get_1d_index(1, 0, 0, nx, ny) == 1
    assert get_1d_index(0, 1, 0, nx, ny) == 2
    assert get_1d_index(1, 1, 0, nx, ny) == 3
    assert get_1d_index(0, 0, 1, nx, ny) == 4
    assert get_1d_index(1, 0, 1, nx, ny) == 5
    assert get_1d_index(0, 1, 1, nx, ny) == 6
    assert get_1d_index(1, 1, 1, nx, ny) == 7



