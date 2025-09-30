import pytest
from src.gmsh_runner import extract_bounding_box_with_gmsh
from tests.unit.conftest import dummy_gmsh_instance
from tests.helpers.assertions import assert_error_contains
from tests.helpers.constants import EXPECTED_ERROR_PREFIX

# Paths to mock files
CUBE_STEP_PATH = "test_models/test_cube.step"

def test_gmsh_error_on_too_many_voxels(dummy_gmsh_instance):
    """Tests that the function raises an error when voxel count exceeds a limit."""
    # This test relies on an internal check in gmsh_runner.py that raises a ValueError.
    # The actual voxel count is hard to simulate, so we'll test for the expected error type and message.
    # Assumes the function raises a ValueError with a specific message.
    
    with pytest.raises(ValueError) as e:
        extract_bounding_box_with_gmsh(
            CUBE_STEP_PATH,
            resolution=0.001, # A very fine resolution to trigger a high voxel count
            flow_region="internal"
        )
    
    assert_error_contains(e, EXPECTED_ERROR_PREFIX)
    assert "too many voxels" in str(e.value)
