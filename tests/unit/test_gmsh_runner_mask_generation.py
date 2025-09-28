# tests/unit/test_gmsh_runner_mask_generation.py

import pytest
import src.gmsh_runner as gmsh_runner
from tests.unit.conftest import DummyGmsh  # reuse dummy classes/fixture from conftest

def test_single_voxel_grid_when_bbox_smaller_than_resolution(monkeypatch, tmp_path):
    """BBox smaller than resolution should clamp nx, ny, nz to 1."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    small_bbox = (0, 0, 0, 0.5, 0.5, 0.5)
    dummy = DummyGmsh(bbox=small_bbox)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=0.4)
    assert res["geometry_mask_shape"] == [1, 1, 1]
    assert len(res["geometry_mask_flat"]) == 1

def test_multi_voxel_grid_shape_and_mask_length(monkeypatch, tmp_path):
    """BBox larger than resolution should produce correct shape and mask length."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    bbox = (0, 0, 0, 9, 6, 3)
    dummy = DummyGmsh(bbox=bbox)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=2.9)
    assert res["geometry_mask_shape"] == [3, 2, 1]
    assert len(res["geometry_mask_flat"]) == 6

def test_all_inside_results_in_all_fluid(monkeypatch, tmp_path):
    """If isInside() returns True for all voxels, mask should be all fluid (1)."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    dummy = DummyGmsh(bbox=(0, 0, 0, 2, 2, 2))
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    monkeypatch.setattr(gmsh_runner.gmsh.model, "isInside", lambda *args: True)
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="internal")
    assert set(res["geometry_mask_flat"]) == {1}

def test_all_outside_results_in_all_solid(monkeypatch, tmp_path):
    """If isInside() returns False for all voxels, mask should be all solid (0)."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    dummy = DummyGmsh(bbox=(0, 0, 0, 2, 2, 2))
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    monkeypatch.setattr(gmsh_runner.gmsh.model, "isInside", lambda *args: False)
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=1, flow_region="internal")
    assert set(res["geometry_mask_flat"]) == {0}

def test_mixed_inside_outside_produces_mixed_mask(monkeypatch, tmp_path):
    """If some voxels are inside and some outside, mask should contain both 0 and 1."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    res_val = 0.9
    min_x = min_y = min_z = 0
    centre = (min_x + 0.5 * res_val,
              min_y + 0.5 * res_val,
              min_z + 0.5 * res_val)
    inside_points = {centre}
    dummy = DummyGmsh(bbox=(0, 0, 0, 2, 2, 1), inside_points=inside_points)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    res = gmsh_runner.extract_bounding_box_with_gmsh(str(step_file), resolution=res_val)
    assert 0 in res["geometry_mask_flat"] and 1 in res["geometry_mask_flat"]



