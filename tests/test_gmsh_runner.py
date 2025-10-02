# tests/test_gmsh_runner.py

import json
import unittest
from pathlib import Path
from src.gmsh_runner import extract_bounding_box_with_gmsh
from src.utils.gmsh_input_check import ValidationError

class GmshRunnerTests(unittest.TestCase):
    """
    Unit tests for the Gmsh runner's geometry mask extraction.
    """
    
    # Define the base path for test models
    # Note: Path is adjusted to navigate from 'tests/' to 'tests/test_models/'
    TEST_MODELS_DIR = Path(__file__).parent / "test_models"
    
    # Define the name of the file that is expected to fail
    INVALID_GEOMETRY_FILE = "mock_invalid_geometry.step"

    def test_all_models(self):
        """
        Iterates through all valid .step files in the test_models directory, 
        runs the gmsh runner, and asserts the output against the corresponding JSON file.
        The invalid geometry test case is handled separately.
        """
        test_dir = self.TEST_MODELS_DIR
        print(f"Searching for test models in: {test_dir}")

        # Loop through all files in the test directory
        for step_file in test_dir.glob("*.step"):
            # Skip the explicitly invalid geometry file
            if step_file.name == self.INVALID_GEOMETRY_FILE:
                continue

            with self.subTest(step_file=step_file.name):
                print(f"\n--- Testing model: {step_file.name} ---")

                # Define the corresponding output JSON file path
                json_file_name = step_file.stem + "_internal_output.json"
                json_path = test_dir / json_file_name

                # Skip if the corresponding JSON file doesn't exist
                if not json_path.exists():
                    print(f"Skipping: {json_path} not found.")
                    continue

                # Load the expected output from the JSON file
                with open(json_path, "r") as f:
                    expected_output = json.load(f)

                # Determine flow region from file name if applicable
                flow_region = "external" if "external" in step_file.stem else "internal"
                print(f"Detected flow region: {flow_region}")

                # Run the gmsh runner function
                try:
                    actual_output = extract_bounding_box_with_gmsh(
                        step_path=str(step_file),
                        resolution=None, # Use default resolution
                        flow_region=flow_region
                    )

                    # Assert that the actual output matches the expected output
                    self.assertEqual(actual_output, expected_output, 
                                     f"Output mismatch for model: {step_file.name}")
                    print(f"✅ Test passed for {step_file.name}")

                except Exception as e:
                    self.fail(f"Test failed for {step_file.name} unexpectedly with error: {e}")

    def test_invalid_geometry_raises_error(self):
        """
        Tests that extract_bounding_box_with_gmsh raises a ValueError 
        when provided with a STEP file that contains no 3D volumes.
        """
        invalid_step_path = str(self.TEST_MODELS_DIR / self.INVALID_GEOMETRY_FILE)
        
        # Assert that a ValueError is raised for the invalid file
        with self.assertRaises(ValidationError) as context:
            extract_bounding_box_with_gmsh(step_path=invalid_step_path)
            
        expected_error_message = f"STEP file contains no 3D volumes: {invalid_step_path}"
        self.assertIn("STEP file contains no 3D volumes", str(context.exception))
        
        print(f"✅ Test passed for {self.INVALID_GEOMETRY_FILE} (correctly raised ValueError)")


if __name__ == "__main__":
    unittest.main()



