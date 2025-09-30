from src.gmsh_runner import extract_bounding_box_with_gmsh

# Paths to mock files
CUBE_STEP_PATH = "test_models/test_cube.step"
HOLLOW_CYLINDER_STEP_PATH = "test_models/hollow_cylinder.step"

def test_internal_flow_on_solid_model(dummy_gmsh_instance):
    """Tests that internal flow correctly identifies the interior of a solid."""
    result = extract_bounding_box_with_gmsh(
        CUBE_STEP_PATH,
        resolution=1.0,
        flow_region="internal"
    )
    assert result["geometry_mask_flat"].count(1) > 0
    assert result["geometry_mask_flat"].count(0) > 0

def test_external_flow_on_solid_model(dummy_gmsh_instance):
    """Tests that external flow correctly identifies the exterior of a solid."""
    result = extract_bounding_box_with_gmsh(
        CUBE_STEP_PATH,
        resolution=1.0,
        flow_region="external"
    )
    assert result["geometry_mask_flat"].count(1) > 0
    assert result["geometry_mask_flat"].count(0) > 0

def test_internal_flow_on_hollow_model(dummy_gmsh_instance):
    """Tests that internal flow correctly identifies the void space of a hollow object."""
    result = extract_bounding_box_with_gmsh(
        HOLLOW_CYLINDER_STEP_PATH,
        resolution=1.0,
        flow_region="internal"
    )
    assert result["geometry_mask_flat"].count(1) > 0
    assert result["geometry_mask_flat"].count(0) > 0
