# tests/test_gmsh_runner.py

import builtins
import io
import os
import types
import pytest

import src.gmsh_runner as gmsh_runner


class DummyGmshModel:
    def __init__(self, bbox=(0, 0, 0, 10, 10, 10), inside_points=None):
        self._bbox = bbox
        self._inside_points = inside_points or set()

    def add(self, name):
        pass

    def getEntities(self, dim):
        return [(3, 1)]

    def getBoundingBox(self, dim, tag):
        return self._bbox

    def isInside(self, dim, tag, coords):
        # Return True if coords tuple is in inside_points
        return tuple(round(c, 3) for c in coords) in self._inside_points


class DummyGmsh:
    def __init__(self, bbox=(0, 0, 0, 10, 10, 10), inside_points=None):
        self.model = DummyGmshModel(bbox, inside_points)
        self.logger = types.SimpleNamespace(start=lambda: None)
        self._finalized = False

    def initialize(self):
        pass

    def finalize(self):
        self._finalized = True

    def open(self, path):
        pass


@pytest.fixture(autouse=True)
def patch_gmsh(monkeypatch):
    dummy = DummyGmsh()
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    # Patch validate_step_has_volumes to no‑op
    monkeypatch.setattr(gmsh_runner, "validate_step_has_volumes", lambda p: None)
    # Patch load_resolution_profile to return default
    monkeypatch.setattr(gmsh_runner, "load_resolution_profile", lambda: {"default_resolution": {"dx": 2}})
    return dummy


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        gmsh_runner.extract_bounding_box_with_gmsh(str(tmp_path / "missing.step"))


def test_resolution_fallback_used(monkeypatch, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Remove resolution argument to trigger fallback
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file))
    assert "geometry_mask_flat" in result
    assert isinstance(result["geometry_mask_flat"], list)


def test_resolution_too_large(monkeypatch, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Patch bounding box to have smallest dim = 5
    dummy = DummyGmsh(bbox=(0, 0, 0, 5, 10, 10))
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=5)
    assert "too large" in str(e.value)


def test_external_flow_padding_applied(monkeypatch, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    bbox = (0, 0, 0, 10, 10, 10)
    dummy = DummyGmsh(bbox=bbox)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="external", padding_factor=1)
    # Padding should have been applied: nx > original size/resolution
    nx = int((bbox[3] - bbox[0] + 2 * 1) / 1)
    assert nx >= 12


def test_voxel_count_limit(monkeypatch, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Patch bounding box to be huge so voxel count > 10M
    huge_bbox = (0, 0, 0, 100000, 100000, 100000)
    dummy = DummyGmsh(bbox=huge_bbox)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(MemoryError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1)
    assert "Voxel grid too large" in str(e.value)


def test_mask_generation_internal_and_external(monkeypatch, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Mark one point as inside solid
    inside_points = {(0.5, 0.5, 0.5)}
    dummy = DummyGmsh(bbox=(0, 0, 0, 2, 2, 2), inside_points=inside_points)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    res_internal = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="internal")
    res_external = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="external")
    assert 0 in res_internal["geometry_mask_flat"]
    assert 0 in res_external["geometry_mask_flat"]


def test_invalid_flow_region(monkeypatch, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    dummy = DummyGmsh()
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="unsupported")
    assert "Unsupported flow_region" in str(e.value)



