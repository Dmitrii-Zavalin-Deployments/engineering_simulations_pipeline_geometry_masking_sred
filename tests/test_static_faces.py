import pytest
import json
from pathlib import Path

from src.generate_blender_mesh_format import generate_fluid_mesh_data_json

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_faces_in_valid_2x2x2_fixture(tmp_path):
    # Input fixture with perfectly cubical grid and static velocity
    input_path = tmp_path / "valid_input_2x2x2.json"
    output_path = tmp_path / "out.json"

    # Copy a valid fixture (you’ll need to create this separately)
    input_path.write_text((FIXTURE_DIR / "valid_input_2x2x2.json").read_text())

    # Run the mesh generator
    generate_fluid_mesh_data_json(str(input_path), str(output_path))

    # Load the output JSON to validate static face structure
    with open(output_path, "r") as f:
        output = json.load(f)

    static_faces = output.get("static_faces")
    boundary_vertices = output["time_steps"][0]["vertices"]

    assert isinstance(static_faces, list)
    assert len(static_faces) > 0

    for face in static_faces:
        assert len(face) == 4  # All faces must be quads
        for idx in face:
            assert isinstance(idx, int)
            assert 0 <= idx < len(boundary_vertices)



