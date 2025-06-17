import pytest
import json
from pathlib import Path

from src.generate_blender_mesh_format import generate_fluid_mesh_data_json

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def test_euler_displacement_matches_velocity(tmp_path):
    # Setup input and output paths
    input_path = tmp_path / "valid_input_2x2x2.json"
    output_path = tmp_path / "out.json"

    # Copy fixture into temp path
    input_path.write_text((FIXTURE_DIR / "valid_input_2x2x2.json").read_text())

    # Run the generator
    generate_fluid_mesh_data_json(str(input_path), str(output_path))

    # Load the result
    with open(output_path) as f:
        data = json.load(f)

    time_steps = data["time_steps"]
    assert len(time_steps) == 2  # Should have 2 frames: t=0, t=1

    vertices_t0 = time_steps[0]["vertices"]
    vertices_t1 = time_steps[1]["vertices"]

    assert len(vertices_t0) == len(vertices_t1)

    # Velocity in fixture is [0.1, 0.0, 0.0], dt = 1.0
    expected_displacement = 0.1

    # Check that each vertex in t=1 is offset by +0.1 in X compared to t=0
    for v0, v1 in zip(vertices_t0, vertices_t1):
        dx = round(v1[0] - v0[0], 6)
        dy = round(v1[1] - v0[1], 6)
        dz = round(v1[2] - v0[2], 6)
        assert dx == pytest.approx(expected_displacement)
        assert dy == 0.0
        assert dz == 0.0



