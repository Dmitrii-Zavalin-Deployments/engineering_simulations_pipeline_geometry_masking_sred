# tests/unit/test_gmsh_runner_volumes_and_bbox.py

import pytest
import src.gmsh_runner as gmsh_runner
from tests.unit.conftest import DummyGmsh  # reuse dummy classes/fixture from conftest

def test_no_volumes_raises_value_error(_patch_gmsh, monkeypatch, tmp_path):
    """If gmsh.model.getEntities(3) returns empty, should raise ValueError."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Dummy gmsh with no volumes
    dummy = DummyGmsh(entities_ret=[])
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1)
    assert "No volume entities" in str(e.value)

def test_resolution_too_large_against_min_dim(_patch_gmsh, monkeypatch, tmp_path):
    """If resolution >= smallest model dimension, should raise ValueError and finalize."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Smallest dimension is 5 (x); resolution=5 should trigger error
    dummy = DummyGmsh(bbox=(0, 0, 0, 5, 10, 10))
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=5)
    assert "too large" in str(e.value)
    # Ensure finalize was attempted even on exception
    assert dummy._finalized is True



