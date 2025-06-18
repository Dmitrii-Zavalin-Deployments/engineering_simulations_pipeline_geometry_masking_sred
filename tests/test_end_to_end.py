import pytest
import json
from pathlib import Path

from src.generate_blender_mesh_format import generate_fluid_mesh_data_json

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def test_end_to_end_valid_input(tmp_path):
    input_path = tmp_path / "valid_input_2x2x2.json"
    output_path = tmp_path / "out.json"

    # Copy valid input
    input_path.write_text((FIXTURE_DIR / "valid_input_2x2x2.json").read_text())

    # Run pipeline
    generate_fluid_mesh_data_json(str(input_path), str(output_path))

    # Load result
    with open(output_path) as f:
        data = json.load(f)

    # Top-level checks
    assert isinstance(data, dict)
    assert "mesh_name" in data
    assert "static_faces" in data
    assert "time_steps" in data

    # Static faces
    static_faces = data["static_faces"]
    assert isinstance(static_faces, list)
    assert all(isinstance(face, list) and len(face) == 4 for face in static_faces)

    # Time steps
    steps = data["time_steps"]
    assert isinstance(steps, list)
    assert len(steps) == 2  # From fixture

    vtx0 = steps[0]["vertices"]
    vtx1 = steps[1]["vertices"]
    assert len(vtx0) == len(vtx1)

    # Check vertex motion matches velocity [0.1, 0, 0] over dt = 1.0
    for a, b in zip(vtx0, vtx1):
        dx = round(b[0] - a[0], 6)
        dy = round(b[1] - a[1], 6)
        dz = round(b[2] - a[2], 6)

        assert dx == pytest.approx(0.1)
        assert dy == pytest.approx(0.0)
        assert dz == pytest.approx(0.0)



