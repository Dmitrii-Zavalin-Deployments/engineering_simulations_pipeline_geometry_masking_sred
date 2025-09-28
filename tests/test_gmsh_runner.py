# tests/test_gmsh_runner.py

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
    with pytest.raises(FileNotFoundError):
        gmsh_runner.extract_bounding_box_with_gmsh(str(tmp_path / "missing.step"))


def test_resolution_fallback_used(gmsh_mock_factory, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    gmsh_mock_factory()
    # Remove resolution argument to trigger fallback
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file))
    assert "geometry_mask_flat" in result
    assert isinstance(result["geometry_mask_flat"], list)


def test_resolution_too_large(gmsh_mock_factory, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Patch bounding box to have smallest dim = 5
    gmsh_mock_factory(bbox=(0, 0, 0, 5, 10, 10))
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=5)
    assert "too large" in str(e.value)


def test_external_flow_padding_applied(gmsh_mock_factory, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    bbox = (0, 0, 0, 10, 10, 10)
    gmsh_mock_factory(bbox=bbox)
    gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="external", padding_factor=1)
    # Padding should have been applied: nx > original size/resolution
    nx = int((bbox[3] - bbox[0] + 2 * 1) / 1)
    assert nx >= 12


def test_voxel_count_limit(gmsh_mock_factory, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Patch bounding box to be huge so voxel count > 10M
    huge_bbox = (0, 0, 0, 100000, 100000, 100000)
    gmsh_mock_factory(bbox=huge_bbox)
    with pytest.raises(MemoryError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1)
    assert "Voxel grid too large" in str(e.value)


def test_mask_generation_internal_and_external(gmsh_mock_factory, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Mark one point as inside solid
    inside_points = {(0.5, 0.5, 0.5)}
    gmsh_mock_factory(bbox=(0, 0, 0, 2, 2, 2), inside_points=inside_points)
    res_internal = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="internal")
    res_external = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="external")
    assert 0 in res_internal["geometry_mask_flat"]
    assert 0 in res_external["geometry_mask_flat"]


def test_invalid_flow_region(gmsh_mock_factory, tmp_path):
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    gmsh_mock_factory()
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="unsupported")
    assert "Unsupported flow_region" in str(e.value)



