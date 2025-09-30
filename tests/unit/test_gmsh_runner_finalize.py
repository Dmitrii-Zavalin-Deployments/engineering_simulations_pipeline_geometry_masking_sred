import pytest
import sys
from src.gmsh_runner import extract_bounding_box_with_gmsh
from tests.unit.conftest import dummy_gmsh_instance, gmsh_finalized_checker

# Path to a mock file
MOCK_STEP_PATH = "test_models/test_cube.step"

def test_gmsh_runner_finalizes_correctly(gmsh_finalized_checker):
    """
    Tests that the gmsh.finalize() method is called after the function finishes,
    even if an error occurs.
    """
    try:
        extract_bounding_box_with_gmsh(MOCK_STEP_PATH, resolution=1.0, flow_region="internal")
    except Exception as e:
        # We expect the test to pass the checker regardless of the outcome
        pass
    
    # The gmsh_finalized_checker fixture will handle the assertion
    # as it runs after the test function finishes.
