import pytest
import json
from pathlib import Path

# Adjust the import below if you rename or relocate your source file
from src.generate_blender_mesh_format import generate_fluid_mesh_data_json

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def test_missing_velocity_history(tmp_path):
    input_path = tmp_path / "invalid_missing_keys.json"
    input_path.write_text((FIXTURE_DIR / "invalid_missing_keys.json").read_text())

    with pytest.raises(KeyError):
        generate_fluid_mesh_data_json(str(input_path), tmp_path / "out.json")

def test_malformed_grid_shape(tmp_path):
    input_path = tmp_path / "malformed_grid_shape.json"
    input_path.write_text((FIXTURE_DIR / "malformed_grid_shape.json").read_text())

    with pytest.raises(ValueError):
        generate_fluid_mesh_data_json(str(input_path), tmp_path / "out.json")

def test_nonmonotonic_time_points(tmp_path):
    input_path = tmp_path / "nonmonotonic_time_points.json"
    input_path.write_text((FIXTURE_DIR / "nonmonotonic_time_points.json").read_text())

    with pytest.raises(ValueError):
        generate_fluid_mesh_data_json(str(input_path), tmp_path / "out.json")

def test_nodes_coords_size_mismatch(tmp_path):
    input_path = tmp_path / "invalid_nodes_coords_size.json"
    input_path.write_text((FIXTURE_DIR / "invalid_nodes_coords_size.json").read_text())

    with pytest.raises(ValueError):
        generate_fluid_mesh_data_json(str(input_path), tmp_path / "out.json")

def test_velocity_length_mismatch(tmp_path):
    input_path = tmp_path / "velocity_length_mismatch.json"
    input_path.write_text((FIXTURE_DIR / "velocity_length_mismatch.json").read_text())

    with pytest.raises(ValueError):
        generate_fluid_mesh_data_json(str(input_path), tmp_path / "out.json")

def test_invalid_json_file(tmp_path):
    broken_path = tmp_path / "corrupt.json"
    broken_path.write_text("{ not: valid JSON }")

    with pytest.raises(json.JSONDecodeError):
        generate_fluid_mesh_data_json(str(broken_path), tmp_path / "out.json")



