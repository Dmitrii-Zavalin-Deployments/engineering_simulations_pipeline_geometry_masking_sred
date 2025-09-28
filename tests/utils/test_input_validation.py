# 📄 tests/utils/test_input_validation.py

import pytest
import tempfile
import pathlib
from unittest.mock import patch
import src.utils.input_validation as iv
from src.utils.gmsh_input_check import validate_step_has_volumes
from tests.unit.conftest import patch_gmsh

# ------------------------------------------------------------------------------------
# 🧪 Volume Validation Tests — validate_step_has_volumes
# ------------------------------------------------------------------------------------

def step_with_volume():
    return { "solids": [ {"id": 101}, {"id": 202} ] }

def step_empty():
    return { "solids": [] }

@patch("os.path.isfile", return_value=True)
@patch("src.utils.input_validation.validate_step_file", return_value=True)
def test_step_with_volume_passes(patch_gmsh, _, __):
    try:
        validate_step_has_volumes(step_with_volume())
    except Exception as e:
        pytest.fail(f"Unexpected error in step_with_volume: {e}")

@patch("os.path.isfile", return_value=True)
@patch("src.utils.input_validation.validate_step_file", return_value=True)
def test_step_missing_solids_key_raises(_, __):
    invalid = {}
    with pytest.raises(KeyError):
        validate_step_has_volumes(invalid)

@patch("os.path.isfile", return_value=True)
@patch("src.utils.input_validation.validate_step_file", return_value=True)
@patch("gmsh.model.getEntities", side_effect=Exception("Simulated Gmsh crash"))
def test_step_with_no_volumes_raises(_, __, ___):
    with pytest.raises(Exception):
        validate_step_has_volumes(step_empty())

@patch("os.path.isfile", return_value=True)
@patch("src.utils.input_validation.validate_step_file", return_value=True)
@pytest.mark.parametrize("bad_input", [None, "step", 42, ["solids"], {"solids": None}])
def test_invalid_step_types_raise_typeerror_or_file_not_found(_, __, bad_input):
    with pytest.raises((TypeError, FileNotFoundError)):
        validate_step_has_volumes(bad_input)

@patch("os.path.isfile", return_value=True)
@patch("src.utils.input_validation.validate_step_file", return_value=True)
def test_volume_validator_runtime_safe(patch_gmsh, _, __):
    import time
    try:
        start = time.time()
        validate_step_has_volumes(step_with_volume())
        assert time.time() - start < 0.2
    except Exception as e:
        pytest.fail(f"Volume validator runtime test failed: {e}")

# ------------------------------------------------------------------------------------
# 🧪 STEP File Path Validation Tests — validate_step_file
# ------------------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_input", [42, {}, None, [], set(), object()])
def test_invalid_step_types_raise_typeerror_or_file_not_found(invalid_input):
    with pytest.raises((TypeError, FileNotFoundError)):
        iv.validate_step_file(invalid_input)

@pytest.mark.parametrize("bad_path", ["nonexistent.step", "/fake/dir/file.step"])
def test_nonexistent_file_raises_file_not_found(bad_path):
    with pytest.raises(FileNotFoundError):
        iv.validate_step_file(bad_path)

def test_valid_step_file_passes():
    with tempfile.NamedTemporaryFile(suffix=".step") as temp_file:
        assert iv.validate_step_file(temp_file.name)

def test_valid_pathlike_step_file_passes():
    with tempfile.NamedTemporaryFile(suffix=".step") as temp_file:
        path_obj = pathlib.Path(temp_file.name)
        assert iv.validate_step_file(path_obj)

def test_non_step_extension_still_passes_if_file_exists():
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        assert iv.validate_step_file(temp_file.name)

# ------------------------------------------------------------------------------------
# 🧪 Mock Fixture Verification — validate_step_file
# ------------------------------------------------------------------------------------

def test_step_file_validation(mock_validate_step_file):
    result = iv.validate_step_file("fake/path/to/model.step")
    assert result is True
    mock_validate_step_file.assert_called_once()

# ------------------------------------------------------------------------------------
# 🧪 Resolution Profile Loader — load_resolution_profile
# ------------------------------------------------------------------------------------

def test_load_resolution_profile(tmp_path):
    """
    Validates that resolution YAML is parsed correctly and keys are accessible.
    """
    mock_profile = tmp_path / "resolution_profile.yaml"
    mock_profile.write_text(
        "default_resolution:\n  dx: 0.1\n  dy: 0.1\n  dz: 0.1\n"
    )

    from src.utils.input_validation import load_resolution_profile
    loaded = load_resolution_profile(path=mock_profile)

    assert "default_resolution" in loaded
    assert loaded["default_resolution"]["dx"] == 0.1
    assert loaded["default_resolution"]["dy"] == 0.1
    assert loaded["default_resolution"]["dz"] == 0.1



