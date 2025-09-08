# tests/unit/test_gmsh_runner_file_and_resolution.py

import pytest
import src.gmsh_runner as gmsh_runner
from .conftest import DummyGmsh  # reuse dummy classes/fixture if placed in conftest

def test_file_not_found(tmp_path):
    """Non-existent STEP file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        gmsh_runner.extract_bounding_box_with_gmsh(str(tmp_path / "missing.step"))

def test_resolution_explicit_used(tmp_path):
    """Explicit resolution should be used to compute shape."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # BBox is 10 in each axis; resolution 5 → nx=ny=nz=2
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=5, flow_region="internal")
    assert result["geometry_mask_shape"] == [2, 2, 2]
    assert len(result["geometry_mask_flat"]) == 8

def test_resolution_from_profile(monkeypatch, tmp_path):
    """Resolution should be loaded from profile when not provided."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Profile provides dx=3 → shape floor(10/3)=3
    monkeypatch.setattr(gmsh_runner, "load_resolution_profile", lambda: {"default_resolution": {"dx": 3}})
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=None, flow_region="internal")
    assert result["geometry_mask_shape"] == [3, 3, 3]

def test_resolution_profile_missing_or_invalid_falls_back_to_2mm(monkeypatch, tmp_path):
    """If profile loader fails or is missing keys, fallback to 2 mm."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Simulate loader raising → fallback to 2 mm (shape=5,5,5)
    monkeypatch.setattr(
        gmsh_runner, "load_resolution_profile",
        lambda: (_ for _ in ()).throw(RuntimeError("bad profile"))
    )
    result = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=None, flow_region="internal")
    assert result["geometry_mask_shape"] == [5, 5, 5]



