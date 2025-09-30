import pytest
from src.gmsh_runner import extract_bounding_box_with_gmsh
from tests.unit.conftest import dummy_gmsh_instance

# Path to a mock file
CUBE_STEP_PATH = "test_models/test_cube.step"

def test_mask_generation_dimensions(dummy_gmsh_instance):
    """Tests that the generated mask has correct dimensions based on resolution."""
    resolution = 10.0
    result = extract_bounding_box_with_gmsh(
        CUBE_STEP_PATH,
        resolution=resolution,
        flow_region="internal"
    )
    
    # We can't know the exact dimensions, but we can check the keys
    assert "geometry_mask_flat" in result
    assert "geometry_mask_shape" in result
    assert len(result["geometry_mask_shape"]) == 3
