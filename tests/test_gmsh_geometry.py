# tests/test_gmsh_geometry.py

import pytest
import json
from unittest import mock
from src.gmsh_geometry import validate_flow_region_and_update, extract_geometry_mask

# --- validate_flow_region_and_update ---

def test_cube_bounded_geometry_preserves_internal(monkeypatch):
    volumes = [(3, 1)]
    monkeypatch.setattr("src.gmsh_core.compute_bounding_box", lambda vols: (0, 0, 0, 1, 1, 1))
    model_data = {"model_properties": {"flow_region": "internal"}}
    validate_flow_region_and_update(model_data, volumes)
    assert model_data["model_properties"]["flow_region"] == "internal"
    assert "flow_region_comment" not in model_data["model_properties"]

def test_non_cube_geometry_switches_to_external(monkeypatch):
    volumes = [(3, 1)]
    monkeypatch.setattr("src.gmsh_core.compute_bounding_box", lambda vols: (0, 0, 0, 1, 2, 1))
    model_data = {"model_properties": {"flow_region": "internal"}}
    validate_flow_region_and_update(model_data, volumes)
    assert model_data["model_properties"]["flow_region"] == "external"
    assert "Auto-switched to external" in model_data["model_properties"]["flow_region_comment"]

def test_zero_dimension_raises(monkeypatch):
    volumes = [(3, 1)]
    monkeypatch.setattr("src.gmsh_core.compute_bounding_box", lambda vols: (0, 0, 0, 0, 1, 1))
    model_data = {"model_properties": {"flow_region": "internal"}}
    with pytest.raises(ValueError, match="one or more dimensions are zero"):
        validate_flow_region_and_update(model_data, volumes)

# --- extract_geometry_mask ---

@pytest.fixture
def mock_gmsh(monkeypatch):
    monkeypatch.setattr("gmsh.initialize", lambda: None)
    monkeypatch.setattr("gmsh.finalize", lambda: None)
    monkeypatch.setattr("gmsh.isInitialized", lambda: True)
    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [(3, 1)])
    monkeypatch.setattr("src.gmsh_core.initialize_gmsh_model", lambda path: None)
    # ✅ Ensure non-zero dimensions for both internal and external flow
    monkeypatch.setattr("src.gmsh_core.compute_bounding_box", lambda vols: (0, 0, 0, 1, 1, 1))
    monkeypatch.setattr("src.gmsh_core.classify_voxel_by_corners", lambda px, py, pz, res, tags: 1)

def test_extract_geometry_mask_internal(monkeypatch, mock_gmsh, tmp_path):
    step_file = tmp_path / "model.step"
    step_file.write_text("dummy")

    model_data = {
        "model_properties": {
            "flow_region": "internal",
            "no_slip": True
        }
    }

    result = extract_geometry_mask(
        step_path=str(step_file),
        resolution=0.5,
        flow_region="internal",
        padding_factor=1,
        no_slip=True,
        model_data=model_data,
        debug=False
    )

    assert result["geometry_mask_shape"] == [2, 2, 2]
    assert set(result["geometry_mask_flat"]) == {1}
    assert result["mask_encoding"]["fluid"] == 1

def test_extract_geometry_mask_external(monkeypatch, mock_gmsh, tmp_path):
    step_file = tmp_path / "model.step"
    step_file.write_text("dummy")

    model_data = {
        "model_properties": {
            "flow_region": "external",
            "no_slip": False
        }
    }

    result = extract_geometry_mask(
        step_path=str(step_file),
        resolution=0.5,
        flow_region="external",
        padding_factor=1,
        no_slip=False,
        model_data=model_data,
        debug=False
    )

    assert result["geometry_mask_shape"] == [4, 4, 4]
    assert set(result["geometry_mask_flat"]) == {1}

def test_missing_step_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_geometry_mask(
            step_path="missing.step",
            resolution=0.5,
            flow_region="internal",
            padding_factor=1,
            no_slip=True,
            model_data={},
            debug=False
        )

def test_missing_resolution_raises(tmp_path):
    step_file = tmp_path / "model.step"
    step_file.write_text("dummy")

    with pytest.raises(ValueError, match="Resolution must be explicitly defined"):
        extract_geometry_mask(
            step_path=str(step_file),
            resolution=None,
            flow_region="internal",
            padding_factor=1,
            no_slip=True,
            model_data={},
            debug=False
        )

def test_resolution_too_large(monkeypatch, tmp_path):
    step_file = tmp_path / "model.step"
    step_file.write_text("dummy")

    monkeypatch.setattr("gmsh.initialize", lambda: None)
    monkeypatch.setattr("gmsh.finalize", lambda: None)
    monkeypatch.setattr("gmsh.isInitialized", lambda: True)
    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [(3, 1)])
    monkeypatch.setattr("src.gmsh_core.initialize_gmsh_model", lambda path: None)
    # ✅ Ensure bounding box is small enough to trigger resolution error
    monkeypatch.setattr("src.gmsh_core.compute_bounding_box", lambda vols: (0, 0, 0, 1, 1, 1))

    with pytest.raises(ValueError, match="Resolution 2.00 mm is too large"):
        extract_geometry_mask(
            step_path=str(step_file),
            resolution=2.0,
            flow_region="internal",
            padding_factor=1,
            no_slip=True,
            model_data={},
            debug=False
        )



