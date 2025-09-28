# tests/unit/test_gmsh_runner_voxel_limits.py

import pytest
import src.gmsh_runner as gmsh_runner
from tests.unit.conftest import DummyGmsh  # reuse dummy classes/fixture from conftest

def test_voxel_count_within_limit_returns_mask_and_shape(patch_gmsh, tmp_path):
    """When voxel count <= 10M, should return valid mask and shape."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=2)
    shape = res["geometry_mask_shape"]
    # Mask length matches product of shape
    assert len(res["geometry_mask_flat"]) == shape[0] * shape[1] * shape[2]
    # Encoding and flattening order match spec
    assert res["mask_encoding"] == {"fluid": 1, "solid": 0}
    assert res["flattening_order"] == "x-major"

def test_voxel_count_exceeds_limit_raises_memory_error(patch_gmsh, monkeypatch, tmp_path):
    """When voxel count > 10M, should raise MemoryError with safe-resolution suggestion."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # Huge bbox to force voxel count over limit
    huge_bbox = (0, 0, 0, 100000, 100000, 100000)
    dummy = DummyGmsh(bbox=huge_bbox)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    with pytest.raises(MemoryError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1)
    msg = str(e.value)
    assert "Voxel grid too large" in msg
    assert "mm" in msg  # safe-resolution suggestion present



