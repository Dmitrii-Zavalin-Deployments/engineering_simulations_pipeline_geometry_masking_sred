# tests/unit/test_gmsh_runner_flow_regions.py

import pytest
import src.gmsh_runner as gmsh_runner
from tests.unit.conftest import DummyGmsh  # reuse dummy classes/fixture from conftest

def test_internal_flow_classification_mixed_points(patch_gmsh, monkeypatch, tmp_path):
    """Internal flow: some voxels inside (solid=0), others outside (fluid=1)."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    # 2x2x2 grid; mark (0.5,0.5,0.5) as inside
    inside_points = {(0.5, 0.5, 0.5)}
    dummy = DummyGmsh(bbox=(0, 0, 0, 2, 2, 2), inside_points=inside_points)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    res = gmsh_runner.extract_bounding_box_with_gmsh(
        str(step_file), resolution=1, flow_region="internal"
    )
    assert 0 in res["geometry_mask_flat"] and 1 in res["geometry_mask_flat"]

def test_external_flow_padding_and_classification(patch_gmsh, monkeypatch, tmp_path):
    """External flow: padding applied to bbox and classification still 'fluid if not inside'."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    bbox = (0, 0, 0, 10, 10, 10)
    dummy = DummyGmsh(bbox=bbox)
    monkeypatch.setattr(gmsh_runner, "gmsh", dummy)
    res = gmsh_runner.extract_bounding_box_with_gmsh(
        str(step_file), resolution=1, flow_region="external", padding_factor=1
    )
    # Expect padding by 1 resolution on each side: effective length = 10 + 2*1 = 12 → nx floor(12/1)=12
    assert res["geometry_mask_shape"][0] >= 12
    # No inside points → all voxels fluid (1)
    assert set(res["geometry_mask_flat"]) == {1}

def test_unsupported_flow_region(patch_gmsh, tmp_path):
    """Unsupported flow_region should raise ValueError."""
    step_file = tmp_path / "file.step"
    step_file.write_text("dummy")
    with pytest.raises(ValueError) as e:
        gmsh_runner.extract_bounding_box_with_gmsh(
            str(step_file), resolution=1, flow_region="unsupported"
        )
    assert "Unsupported flow_region" in str(e.value)



