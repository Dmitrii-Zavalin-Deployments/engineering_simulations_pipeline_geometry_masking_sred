# tests/helpers/compare_json.py

# -----------------------------------------------------------------------------
# JSON Output Comparator
# Usage: python3 compare_json.py <expected_path> <output_path>
# Compares two JSON files (expected vs generated) and exits with code 1 on mismatch.
# -----------------------------------------------------------------------------

import json
import sys
import argparse
import difflib
from pathlib import Path

# Define the directory where detailed error reports should be saved
ERROR_OUTPUT_DIR = Path("tests/integration_tests_errors")


def compare_json_outputs(expected_path: str, output_path: str):
    """
    Loads, compares, and prints a unified diff for two JSON files.
    Exits with status 1 on failure, 0 on success.
    
    If the test fails, a detailed error report is saved to the ERROR_OUTPUT_DIR.
    """
    expected_path = Path(expected_path)
    output_path = Path(output_path)
    
    # NOTE: The logic to remove the old STALE_MARKER_FILE is removed 
    # since that feature is no longer supported.

    # 1. Load files
    try:
        with open(expected_path) as f:
            expected = json.load(f)
        with open(output_path) as f:
            output = json.load(f)
    except FileNotFoundError:
        print(f'❌ INTEGRATION TEST FAILED: Missing generated output file {output_path.name}.')
        sys.exit(1)
    except json.JSONDecodeError:
        print(f'❌ INTEGRATION TEST FAILED: Generated output file {output_path.name} is invalid JSON.')
        sys.exit(1)

    # 2. Compare content
    if expected != output:
        
        # Initialize output string for console and file
        error_output_content = ""
        error_output_content += f'❌ INTEGRATION TEST FAILED: Output mismatch for {expected_path.name}\n'

        show_full_diff = True
        
        # --- Enhanced Check for Array Length Mismatch ---
        # This check remains useful for debugging, as length mismatches in
        # large arrays like geometry_mask_flat indicate a serious issue.
        if 'geometry_mask_flat' in expected and 'geometry_mask_flat' in output:
            expected_mask = expected['geometry_mask_flat']
            output_mask = output['geometry_mask_flat']
            
            if isinstance(expected_mask, list) and isinstance(output_mask, list) and len(expected_mask) != len(output_mask):
                show_full_diff = False # Suppress the giant diff
                
                error_output_content += "=========================================================================\n"
                error_output_content += f"⚠️ CRITICAL MISMATCH DETECTED: 'geometry_mask_flat' array length differs.\n"
                error_output_content += f"   Expected length: {len(expected_mask)}, Generated length: {len(output_mask)}\n"
                error_output_content += "   Full JSON diff suppressed due to array size difference.\n"
                error_output_content += "=========================================================================\n"
                
        # Print initial error content
        print(error_output_content, end='')

        if show_full_diff:
            # Convert dicts to sorted, indented JSON strings for stable line-by-line diff
            expected_str = json.dumps(expected, indent=2, sort_keys=True)
            output_str = json.dumps(output, indent=2, sort_keys=True)

            diff_lines = list(difflib.unified_diff(
                expected_str.splitlines(keepends=True),
                output_str.splitlines(keepends=True),
                fromfile=expected_path.name,
                tofile=output_path.name,
                lineterm=''
            ))

            diff_header = '\n--- JSON DIFF (Expected vs Generated) ---\n'
            diff_footer = '--------------------------------------\n'
            
            # Add diff to both file content and console output
            error_output_content += diff_header
            error_output_content += "".join(diff_lines)
            error_output_content += diff_footer
            
            # Print the diff to console
            print(diff_header, end='')
            sys.stdout.writelines(diff_lines)
            print(diff_footer, end='')


        # --- Write Detailed Error File ---
        try:
            ERROR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            error_filename = f"diff_{expected_path.stem}_{output_path.stem}.txt"
            error_filepath = ERROR_OUTPUT_DIR / error_filename
            with open(error_filepath, 'w') as f:
                f.write(error_output_content)
            print(f"📄 Detailed error file saved to: {error_filepath}")
        except Exception as e:
            print(f"Error saving detailed error file: {e}")


        # Clean up temporary output file before exiting on failure
        output_path.unlink(missing_ok=True)
        sys.exit(1) # Fail the CI job
    else:
        print(f'✅ INTEGRATION TEST PASSED: {expected_path.name} matches expected output.')

    # Clean up temporary output file on success
    output_path.unlink(missing_ok=True)
    sys.exit(0) # Pass the CI job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two JSON files and generate a diff.")
    parser.add_argument("expected_path", type=str, help="Path to the expected JSON file.")
    parser.add_argument("output_path", type=str, help="Path to the generated output JSON file.")
    args = parser.parse_args()

    compare_json_outputs(args.expected_path, args.output_path)



