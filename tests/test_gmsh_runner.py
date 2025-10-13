# tests/test_gmsh_runner.py

import os
import json
import pytest
import tempfile
from unittest import mock
from src import gmsh_runner
from src.utils import gmsh_input_check  # ✅ Corrected import path
from src.gmsh_geometry import extract_geometry_mask

# Mock Gmsh lifecycle
@pytest.fixture(autouse=True)
def mock_gmsh(monkeypatch):
    monkeypatch.setattr("gmsh.initialize", lambda: None)
    monkeypatch.setattr("gmsh.finalize", lambda: None)
    monkeypatch.setattr("gmsh.isInitialized", lambda: True)

# Mock volume validation
@pytest.fixture(autouse=True)
def mock_file_check(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda path: True)

@pytest.fixture(autouse=True)
def mock_step_validation(monkeypatch):
    monkeypatch.setattr("src.utils.gmsh_input_check.validate_step_has_volumes", lambda path: True)

@pytest.fixture
def mock_volume_check(monkeypatch):
    monkeypatch.setattr("src.utils.gmsh_input_check.validate_step_has_volumes", lambda path: True)

# Sample flow_data.json content
@pytest.fixture
def sample_flow_data(tmp_path):
    flow_data = {
        "model_properties": {
            "default_resolution": 0.5,
            "flow_region": "internal",
            "flow_region_comment": "",
            "no_slip": True
        }
    }
    flow_path = tmp_path / "flow_data.json"
    with open(flow_path, "w") as f:
        json.dump(flow_data, f)
    monkeypatch = mock.patch("src.gmsh_runner.FLOW_DATA_PATH", str(flow_path))  # ✅ Updated to patch constant
    monkeypatch.start()
    yield flow_path
    monkeypatch.stop()

# Mock extract_geometry_mask
@pytest.fixture
def mock_geometry_mask(monkeypatch):
    def fake_mask(**kwargs):
        return {
            "geometry_mask_flat": [-1, 1, 0],
            "geometry_mask_shape": [1, 1, 3],
            "mask_encoding": {"fluid": 1, "solid": 0, "boundary": -1},
            "flattening_order": "x-major"
        }
    monkeypatch.setattr("src.gmsh_geometry.extract_geometry_mask", fake_mask)

def test_valid_internal_flow(sample_flow_data, mock_volume_check, mock_geometry_mask):
    args = mock.Mock()
    args.step = "tests/test_models/test_cube.step"
    args.resolution = 0.5
    args.flow_region = "internal"
    args.padding_factor = 5
    args.no_slip = True
    args.output = None
    args.debug = False

    with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
        gmsh_runner.main()

def test_valid_external_flow(sample_flow_data, mock_volume_check, mock_geometry_mask):
    args = mock.Mock()
    args.step = "tests/test_models/hollow_cylinder.step"
    args.resolution = 0.5
    args.flow_region = "external"
    args.padding_factor = 2
    args.no_slip = False
    args.output = None
    args.debug = False

    with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
        gmsh_runner.main()

def test_missing_flow_data(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    args = mock.Mock()
    args.step = "tests/test_models/test_cube.step"
    args.resolution = 0.5
    args.flow_region = "internal"
    args.padding_factor = 5
    args.no_slip = True
    args.output = None
    args.debug = False

    with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
        with pytest.raises(FileNotFoundError):
            gmsh_runner.main()

def test_invalid_step_file(monkeypatch):
    monkeypatch.setattr("src.utils.gmsh_input_check.validate_step_has_volumes", lambda path: (_ for _ in ()).throw(FileNotFoundError("STEP file not found")))
    args = mock.Mock()
    args.step = "invalid_path.step"
    args.resolution = 0.5
    args.flow_region = "internal"
    args.padding_factor = 5
    args.no_slip = True
    args.output = None
    args.debug = False

    with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
        with pytest.raises(RuntimeError, match="STEP file validation failed"):
            gmsh_runner.main()

def test_boundary_reclassification_no_slip(sample_flow_data, mock_volume_check):
    def mock_mask(**kwargs):
        return {
            "geometry_mask_flat": [-1, -1, 1],
            "geometry_mask_shape": [1, 1, 3],
            "mask_encoding": {"fluid": 1, "solid": 0, "boundary": -1},
            "flattening_order": "x-major"
        }
    with mock.patch("src.gmsh_geometry.extract_geometry_mask", mock_mask):
        args = mock.Mock()
        args.step = "tests/test_models/test_cube.step"
        args.resolution = 0.5
        args.flow_region = "internal"
        args.padding_factor = 5
        args.no_slip = True
        args.output = None
        args.debug = False

        with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
            gmsh_runner.main()

def test_boundary_reclassification_slip(sample_flow_data, mock_volume_check):
    def mock_mask(**kwargs):
        return {
            "geometry_mask_flat": [-1, -1, 0],
            "geometry_mask_shape": [1, 1, 3],
            "mask_encoding": {"fluid": 1, "solid": 0, "boundary": -1},
            "flattening_order": "x-major"
        }
    with mock.patch("src.gmsh_geometry.extract_geometry_mask", mock_mask):
        args = mock.Mock()
        args.step = "tests/test_models/test_cube.step"
        args.resolution = 0.5
        args.flow_region = "internal"
        args.padding_factor = 5
        args.no_slip = False
        args.output = None
        args.debug = False

        with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
            gmsh_runner.main()

def test_output_written(sample_flow_data, mock_volume_check):
    def mock_mask(**kwargs):
        return {
            "geometry_mask_flat": [0, 1, 1],
            "geometry_mask_shape": [1, 1, 3],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    with mock.patch("src.gmsh_geometry.extract_geometry_mask", mock_mask):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            args = mock.Mock()
            args.step = "tests/test_models/test_cube.step"
            args.resolution = 0.5
            args.flow_region = "internal"
            args.padding_factor = 5
            args.no_slip = True
            args.output = tmp.name
            args.debug = False

            with mock.patch("argparse.ArgumentParser.parse_args", return_value=args):
                gmsh_runner.main()

            with open(tmp.name, "r") as f:
                data = json.load(f)
                assert "geometry_mask_flat" in data



