# tests/conftest.py

import pytest
import numpy as np

@pytest.fixture
def sample_grid_shape():
    """Provides a default 3x3x3 grid shape for test cases."""
    return [3, 3, 3]

@pytest.fixture
def zero_velocity_field(sample_grid_shape):
    """Generates a velocity field with all zeros."""
    nz, ny, nx = sample_grid_shape
    total_nodes = nz * ny * nx
    return [[0.0, 0.0, 0.0] for _ in range(total_nodes)]

@pytest.fixture
def constant_velocity_field(sample_grid_shape):
    """Generates a velocity field with constant [1.0, 0.0, 0.0] motion."""
    nz, ny, nx = sample_grid_shape
    total_nodes = nz * ny * nx
    return [[1.0, 0.0, 0.0] for _ in range(total_nodes)]

@pytest.fixture
def random_velocity_field(sample_grid_shape):
    """Generates a randomized velocity field for stress testing."""
    nz, ny, nx = sample_grid_shape
    total_nodes = nz * ny * nx
    rng = np.random.default_rng(seed=42)
    return rng.uniform(-1.0, 1.0, size=(total_nodes, 3)).tolist()

@pytest.fixture
def flat_nodes_coords(sample_grid_shape):
    """Generates a linearly distributed coordinate grid (flattened)."""
    nz, ny, nx = sample_grid_shape
    coords = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                coords.append([ix * 1.0, iy * 1.0, iz * 1.0])
    return coords



