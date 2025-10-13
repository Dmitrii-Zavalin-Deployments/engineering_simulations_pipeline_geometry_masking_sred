# tests/unit/test_gmsh_input_check.py

import pytest
import os
from src.utils.gmsh_input_check import validate_step_has_volumes, ValidationError

# --- Fixtures and Mocks ---

@pytest.fixture(autouse=True)
def mock_gmsh(monkeypatch):
    monkeypatch.setattr("gmsh.model.add", lambda name: None)
    monkeypatch.setattr("gmsh.open", lambda path: None)
    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [(1, 1)] if dim == 3 else [])

def test_valid_step_file_with_volumes(tmp_path, monkeypatch):
    # Create a dummy STEP file
    step_file = tmp_path / "valid.step"
    step_file.write_text("dummy content")

    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [(3, 1)])  # Simulate volume presence
    validate_step_has_volumes(str(step_file))  # Should not raise

def test_missing_step_file(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    with pytest.raises(FileNotFoundError, match="STEP file not found"):
        validate_step_has_volumes("nonexistent.step")

def test_step_file_with_no_volumes(tmp_path, monkeypatch):
    step_file = tmp_path / "empty.step"
    step_file.write_text("dummy content")

    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [])  # Simulate no volumes
    with pytest.raises(ValidationError, match="STEP file contains no 3D volumes"):
        validate_step_has_volumes(str(step_file))

def test_valid_dict_payload_with_solids(monkeypatch):
    payload = {"solids": ["mock_solid_1", "mock_solid_2"]}
    monkeypatch.setattr("os.path.isfile", lambda path: True)
    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [(3, 1)])
    validate_step_has_volumes(payload)  # Should not raise

def test_invalid_dict_payload_missing_solids():
    payload = {"not_solids": []}
    with pytest.raises(KeyError, match="Missing or invalid 'solids' list"):
        validate_step_has_volumes(payload)

def test_invalid_dict_payload_solids_not_list():
    payload = {"solids": "not_a_list"}
    with pytest.raises(KeyError, match="Missing or invalid 'solids' list"):
        validate_step_has_volumes(payload)



