# tests/test_gmsh_core.py

import pytest
from unittest import mock
from src.gmsh_core import (
    initialize_gmsh_model,
    compute_bounding_box,
    get_decimal_precision,
    is_inside_model_geometry,
    classify_voxel_by_corners
)

# --- get_decimal_precision ---

@pytest.mark.parametrize("resolution,expected", [
    (0.5, 1),
    (0.125, 3),
    (1.0, 0),
    (0.0001, 4),
    (0.123000, 3),
])
def test_get_decimal_precision(resolution, expected):
    assert get_decimal_precision(resolution) == expected

# --- compute_bounding_box ---

def test_compute_bounding_box():
    mock_bboxes = {
        (3, 1): [0, 0, 0, 1, 1, 1],
        (3, 2): [1, 1, 1, 2, 2, 2]
    }

    with mock.patch("gmsh.model.getBoundingBox", side_effect=lambda dim, tag: mock_bboxes[(dim, tag)]):
        bbox = compute_bounding_box([(3, 1), (3, 2)])
        assert bbox == (0, 0, 0, 2, 2, 2)

# --- initialize_gmsh_model ---

def test_initialize_gmsh_model(monkeypatch):
    monkeypatch.setattr("gmsh.model.add", lambda name: None)
    monkeypatch.setattr("gmsh.logger.start", lambda: None)
    monkeypatch.setattr("gmsh.open", lambda path: None)
    monkeypatch.setattr("gmsh.model", mock.Mock())
    result = initialize_gmsh_model("mock_path.step")
    assert result == gmsh.model

# --- is_inside_model_geometry ---

def test_is_inside_model_geometry_all_inside(monkeypatch):
    monkeypatch.setattr("gmsh.model.isInside", lambda dim, tag, pt: True)
    corner = [1.23456, 2.34567, 3.45678]
    volume_tags = [101, 102]
    assert is_inside_model_geometry(corner, volume_tags, precision=3) is True

def test_is_inside_model_geometry_none_inside(monkeypatch):
    monkeypatch.setattr("gmsh.model.isInside", lambda dim, tag, pt: False)
    corner = [1.23456, 2.34567, 3.45678]
    volume_tags = [101, 102]
    assert is_inside_model_geometry(corner, volume_tags, precision=2) is False

def test_is_inside_model_geometry_mixed(monkeypatch):
    def mock_is_inside(dim, tag, pt):
        return tag == 102
    monkeypatch.setattr("gmsh.model.isInside", mock_is_inside)
    corner = [1.23456, 2.34567, 3.45678]
    volume_tags = [101, 102]
    assert is_inside_model_geometry(corner, volume_tags, precision=2) is True

# --- classify_voxel_by_corners ---

@pytest.mark.parametrize("inside_pattern,expected", [
    ([True] * 8, 0),     # All corners inside → solid
    ([False] * 8, 1),    # All corners outside → fluid
    ([True, False] * 4, -1),  # Mixed → boundary
])
def test_classify_voxel_by_corners(monkeypatch, inside_pattern, expected):
    monkeypatch.setattr("gmsh.model.isInside", lambda dim, tag, pt: inside_pattern.pop(0))
    result = classify_voxel_by_corners(
        px=1.0, py=1.0, pz=1.0,
        resolution=0.5,
        volume_tags=[101]
    )
    assert result == expected



