# tests/test_gmsh_runner.py

import json
import os
import unittest
from pathlib import Path
from src.gmsh_runner import extract_bounding_box_with_gmsh

class GmshRunnerTests(unittest.TestCase):
    """
    Unit tests for the Gmsh runner's geometry mask extraction.
    """

    def test_all_models(self):
        """
        Iterates through all .step files in the test_models directory, runs the gmsh runner,
        and asserts the output against the corresponding JSON file.
        """
        # Get the path to the 'test_models' subdirectory
        test_dir = Path(__file__).parent / "test_models"
        print(f"Searching for test models in: {test_dir}")

        # Loop through all files in the test directory
        for step_file in test_dir.glob("*.step"):
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
                        flow_region=flow_region
                    )

                    # Assert that the actual output matches the expected output
                    self.assertEqual(actual_output, expected_output, 
                                     f"Output mismatch for model: {step_file.name}")
                    print(f"✅ Test passed for {step_file.name}")

                except Exception as e:
                    self.fail(f"Test failed for {step_file.name} with error: {e}")

if __name__ == "__main__":
    unittest.main()
