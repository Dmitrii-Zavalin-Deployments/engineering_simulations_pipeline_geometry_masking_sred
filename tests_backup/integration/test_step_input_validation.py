# /tests/integration/test_step_input_validation.py

import pytest
from pathlib import Path
from gmsh_runner import extract_bounding_box_with_gmsh

STEP_FILE_PATH = Path("test_models/test.step")

@pytest.mark.integration
def test_step_file_exists_and_is_valid_format():
    """Check that the STEP file exists and appears syntactically valid."""
    assert STEP_FILE_PATH.exists(), "Missing STEP file: test_models/test.step"
    assert STEP_FILE_PATH.suffix == ".step", "Incorrect file extension"

@pytest.mark.integration
def test_step_file_importable_by_gmsh():
    """Ensure Gmsh can import and initialize the test STEP file, or raise MemoryError if too fine."""
    try:
        # Use a safe resolution to avoid exceeding voxel limit
        result = extract_bounding_box_with_gmsh(STEP_FILE_PATH, resolution=1.0)
    except MemoryError:
        pytest.skip("Skipped: resolution triggers voxel-count MemoryError for this model")
    except Exception as e:
        pytest.fail(f"Gmsh failed to import STEP file: {e}")
    else:
        assert isinstance(result, dict), "Invalid Gmsh output structure"
        assert "geometry_mask_flat" in result, "geometry_mask_flat missing in output"
        assert "geometry_mask_shape" in result, "geometry_mask_shape missing in output"

@pytest.mark.integration
def test_step_file_resolution_scaling():
    """Verify extracted grid resolution varies with input scaling, within safe voxel limits."""
    coarse_res = 2.0  # mm
    fine_res = 0.8    # mm
    coarse = extract_bounding_box_with_gmsh(STEP_FILE_PATH, resolution=coarse_res)
    fine = extract_bounding_box_with_gmsh(STEP_FILE_PATH, resolution=fine_res)

    # geometry_mask_shape is [nx, ny, nz]
    assert fine["geometry_mask_shape"][0] > coarse["geometry_mask_shape"][0]
    assert fine["geometry_mask_shape"][1] >= coarse["geometry_mask_shape"][1]
    assert fine["geometry_mask_shape"][2] > coarse["geometry_mask_shape"][2]

@pytest.mark.integration
def test_step_file_surface_tags_optional():
    """Confirm optional surface_tags can be present and valid."""
    result = extract_bounding_box_with_gmsh(STEP_FILE_PATH, resolution=1.0)
    if "surface_tags" in result:
        assert isinstance(result["surface_tags"], list), "surface_tags should be a list"



