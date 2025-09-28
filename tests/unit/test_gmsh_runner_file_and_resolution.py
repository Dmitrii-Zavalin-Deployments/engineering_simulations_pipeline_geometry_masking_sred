# tests/unit/test_gmsh_runner_file_and_resolution.py

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


@pytest.fixture
def gmsh_mock_factory(monkeypatch):
    """
    Factory fixture to create and patch a DummyGmsh instance for tests.
    """
    def _create_mock(**kwargs):
        mock = DummyGmsh(**kwargs)
        monkeypatch.setattr(gmsh_runner, "gmsh", mock)
        return mock
    return _create_mock


def test_file_not_found(tmp_path):
    """Non-existent STEP file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        gmsh_runner.extract_bounding_box_with_gmsh(str(tmp_path / "missing.step"))

def test_resolution_explicit_used(gmsh_mock_factory, tmp_path):
    """Explicit resolution should be used to compute shape."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    gmsh_mock_factory()
    # BBox is 10 in each axis; resolution 5 → nx=ny=nz=2
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=5, flow_region="internal")
    assert result["geometry_mask_shape"] == [2, 2, 2]
    assert len(result["geometry_mask_flat"]) == 8

def test_resolution_from_profile(gmsh_mock_factory, monkeypatch, tmp_path):
    """Resolution should be loaded from profile when not provided."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Profile provides dx=3 → shape floor(10/3)=3
    monkeypatch.setattr(gmsh_runner, "load_resolution_profile", lambda: {"default_resolution": {"dx": 3}})
    gmsh_mock_factory()
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=None, flow_region="internal")
    assert result["geometry_mask_shape"] == [3, 3, 3]



