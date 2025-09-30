# tests/unit/test_gmsh_runner.py

import pytest
from pathlib import Path

# Import the function to be tested
from src.gmsh_runner import extract_bounding_box_with_gmsh

# Define the correct path to the mock models
# The path is now relative to the repository root
TEST_MODELS_DIR = Path(__file__).parent.parent.parent / "test_models"
PERFECT_CUBE_STEP = TEST_MODELS_DIR / "test_cube.step"
HOLLOW_CYLINDER_STEP = TEST_MODELS_DIR / "hollow_cylinder.step"
CUBE_WITH_HOLE_STEP = TEST_MODELS_DIR / "cube_with_hole.step"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown_gmsh():
    """Ensures Gmsh is initialized and finalized for the test module."""
    yield
    # No teardown needed as the function under test calls gmsh.finalize()


def test_perfect_cube_internal_flow():
    """Tests that an internal flow on a perfect cube correctly identifies the interior as fluid."""
    if not PERFECT_CUBE_STEP.is_file():
        pytest.skip(f"Mock file not found: {PERFECT_CUBE_STEP}")

    result = extract_bounding_box_with_gmsh(
        step_path=PERFECT_CUBE_STEP,
        resolution=1.0,
        flow_region="internal"
    )

    mask = result["geometry_mask_flat"]
    fluid_count = mask.count(1)
    solid_count = mask.count(0)
    total_voxels = len(mask)

    assert fluid_count > 0.9 * total_voxels
    assert solid_count < 0.1 * total_voxels


def test_perfect_cube_external_flow():
    """Tests that an external flow on a perfect cube correctly identifies the exterior as fluid."""
    if not PERFECT_CUBE_STEP.is_file():
        pytest.skip(f"Mock file not found: {PERFECT_CUBE_STEP}")

    result = extract_bounding_box_with_gmsh(
        step_path=PERFECT_CUBE_STEP,
        resolution=1.0,
        flow_region="external",
        padding_factor=2
    )

    mask = result["geometry_mask_flat"]
    fluid_count = mask.count(1)
    solid_count = mask.count(0)

    assert fluid_count > solid_count * 2
    assert solid_count > 0


def test_hollow_cylinder_internal_flow():
    """Tests that internal flow on a hollow object correctly identifies the void as fluid."""
    if not HOLLOW_CYLINDER_STEP.is_file():
        pytest.skip(f"Mock file not found: {HOLLOW_CYLINDER_STEP}")

    result = extract_bounding_box_with_gmsh(
        step_path=HOLLOW_CYLINDER_STEP,
        resolution=1.0,
        flow_region="internal"
    )

    mask = result["geometry_mask_flat"]
    fluid_count = mask.count(1)
    solid_count = mask.count(0)

    assert fluid_count > 0
    assert solid_count > 0
    assert fluid_count > solid_count


def test_resolution_too_coarse_for_hole():
    """Tests that the script raises an error when resolution is too coarse for internal features."""
    if not CUBE_WITH_HOLE_STEP.is_file():
        pytest.skip(f"Mock file not found: {CUBE_WITH_HOLE_STEP}")
    
    # Use a resolution value larger than the hole's diameter
    too_coarse_resolution = 100.0

    with pytest.raises(ValueError, match="Resolution .* is too large for the model."):
        extract_bounding_box_with_gmsh(
            step_path=CUBE_WITH_HOLE_STEP,
            resolution=too_coarse_resolution,
            flow_region="internal"
        )


def test_resolution_correctly_fine_for_hole():
    """Tests that a fine resolution correctly captures an internal hole."""
    if not CUBE_WITH_HOLE_STEP.is_file():
        pytest.skip(f"Mock file not found: {CUBE_WITH_HOLE_STEP}")

    # Use a resolution value smaller than the hole's diameter
    fine_resolution = 1.0

    result = extract_bounding_box_with_gmsh(
        step_path=CUBE_WITH_HOLE_STEP,
        resolution=fine_resolution,
        flow_region="internal"
    )

    mask = result["geometry_mask_flat"]
    fluid_count = mask.count(1)
    solid_count = mask.count(0)

    assert fluid_count > 0
    assert solid_count > 0
    assert fluid_count < solid_count



