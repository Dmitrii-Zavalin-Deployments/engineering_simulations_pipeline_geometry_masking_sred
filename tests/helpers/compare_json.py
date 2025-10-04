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

def compare_json_outputs(expected_path: str, output_path: str):
    """
    Loads, compares, and prints a unified diff for two JSON files.
    Exits with status 1 on failure, 0 on success.
    """
    expected_path = Path(expected_path)
    output_path = Path(output_path)

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
        # Convert dicts to sorted, indented JSON strings for stable line-by-line diff
        # We use sort_keys=True to ensure key order doesn't cause spurious diffs.
        expected_str = json.dumps(expected, indent=2, sort_keys=True)
        output_str = json.dumps(output, indent=2, sort_keys=True)

        diff = difflib.unified_diff(
            expected_str.splitlines(keepends=True),
            output_str.splitlines(keepends=True),
            fromfile=expected_path.name,
            tofile=output_path.name,
            lineterm=''
        )

        print(f'❌ INTEGRATION TEST FAILED: Output mismatch for {expected_path.name}')
        print('\n--- JSON DIFF (Expected vs Generated) ---')
        sys.stdout.writelines(diff)
        print('--------------------------------------\n')

        # Clean up temporary output file before exiting on failure
        output_path.unlink(missing_ok=True)
        sys.exit(1)
    else:
        print(f'✅ INTEGRATION TEST PASSED: {expected_path.name} matches expected output.')

    # Clean up temporary output file on success
    output_path.unlink(missing_ok=True)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two JSON files and generate a diff.")
    parser.add_argument("expected_path", type=str, help="Path to the expected JSON file.")
    parser.add_argument("output_path", type=str, help="Path to the generated output JSON file.")
    args = parser.parse_args()

    compare_json_outputs(args.expected_path, args.output_path)




