# tests/unit/test_gmsh_runner_finalize.py

import pytest
import src.gmsh_runner as gmsh_runner
from tests.unit.conftest import DummyGmsh  # reuse dummy classes/fixture from conftest

def test_finalize_attempted_even_on_error(gmsh_session, _patch_gmsh, monkeypatch, tmp_path):
    """
    gmsh.finalize() should be attempted even if an exception is raised mid-function.
    Here we trigger a ValueError by setting resolution equal to the smallest model dimension.
    """
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Smallest dimension is 5 (x); resolution=5 should trigger ValueError
    dummy = DummyGmsh(bbox=(0, 0, 0, 5, 10, 10))
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(ValueError):
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=5)
    # Ensure finalize was called
    assert dummy._finalized is True

def test_finalize_errors_are_swallowed(gmsh_session, _patch_gmsh, monkeypatch, tmp_path):
    """
    If gmsh.finalize() itself raises an error, it should be swallowed and not propagate.
    """
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    dummy = DummyGmsh(bbox=(0, 0, 0, 2, 2, 2))
    # Replace finalize with one that raises
    def raising_finalize():
        raise RuntimeError("finalize failed")
    dummy.finalize = raising_finalize
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    # Should still return normally despite finalize raising
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1)
    assert "geometry_mask_shape" in res
    assert "geometry_mask_flat" in res



