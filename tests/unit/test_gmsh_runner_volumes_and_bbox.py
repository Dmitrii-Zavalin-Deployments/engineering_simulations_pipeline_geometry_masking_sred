import pytest
from src.gmsh_runner import extract_bounding_box_with_gmsh
from tests.unit.conftest import dummy_gmsh_instance

# Path to a mock file
CUBE_STEP_PATH = "test_models/test_cube.step"

def test_volumes_are_found(dummy_gmsh_instance):
    """Tests that the script correctly identifies volumes in the STEP file."""
    result = extract_bounding_box_with_gmsh(
        CUBE_STEP_PATH,
        resolution=1.0,
        flow_region="internal"
    )
    assert "volumes" in result
    assert len(result["volumes"]) > 0

def test_bounding_box_is_generated(dummy_gmsh_instance):
    """Tests that a bounding box is generated with correct dimensions."""
    result = extract_bounding_box_with_gmsh(
        CUBE_STEP_PATH,
        resolution=1.0,
        flow_region="internal"
    )
    assert "bounding_box" in result
    bbox = result["bounding_box"]
    assert len(bbox) == 6 # min_x, min_y, min_z, max_x, max_y, max_z
    assert bbox[0] < bbox[3] # min_x < max_x
    assert bbox[1] < bbox[4] # min_y < max_y
    assert bbox[2] < bbox[5] # min_z < max_z
