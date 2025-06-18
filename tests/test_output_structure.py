import pytest
import json
from pathlib import Path

from src.generate_blender_mesh_format import generate_fluid_mesh_data_json

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def test_output_json_structure_is_valid(tmp_path):
    # Setup file paths
    input_path = tmp_path / "valid_input_2x2x2.json"
    output_path = tmp_path / "out.json"

    # Copy valid test input
    input_path.write_text((FIXTURE_DIR / "valid_input_2x2x2.json").read_text())

    # Run the mesh generator
    generate_fluid_mesh_data_json(str(input_path), str(output_path))

    # Load the generated output
    with open(output_path) as f:
        output = json.load(f)

    # Top-level structure
    assert isinstance(output, dict)
    assert "mesh_name" in output
    assert "static_faces" in output
    assert "time_steps" in output

    # Mesh name
    assert isinstance(output["mesh_name"], str)

    # Static faces
    static_faces = output["static_faces"]
    assert isinstance(static_faces, list)
    assert len(static_faces) > 0
    for face in static_faces:
        assert isinstance(face, list)
        assert len(face) == 4
        assert all(isinstance(i, int) and i >= 0 for i in face)

    # Time steps
    time_steps = output["time_steps"]
    assert isinstance(time_steps, list)
    assert len(time_steps) >= 2

    num_vertices = None
    for step_index, step in enumerate(time_steps):
        assert isinstance(step, dict)
        assert "time" in step
        assert "frame" in step
        assert "vertices" in step

        assert isinstance(step["time"], float)
        assert step["frame"] == step_index
        assert isinstance(step["vertices"], list)
        for vtx in step["vertices"]:
            assert isinstance(vtx, list)
            assert len(vtx) == 3
            assert all(isinstance(coord, (int, float)) for coord in vtx)

        if num_vertices is None:
            num_vertices = len(step["vertices"])
        else:
            assert len(step["vertices"]) == num_vertices



