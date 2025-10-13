# tests/test_domain_definition_writer.py

import pytest
from src.domain_definition_writer import validate_domain_bounds, DomainValidationError

def test_valid_domain_bounds():
    domain = {
        "min_x": 0.0, "max_x": 10.0,
        "min_y": 1.0, "max_y": 5.0,
        "min_z": "2.0", "max_z": "6.0"
    }
    # Should not raise
    validate_domain_bounds(domain)

@pytest.mark.parametrize("missing_key", [
    "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"
])
def test_missing_keys(missing_key):
    domain = {
        "min_x": 0.0, "max_x": 10.0,
        "min_y": 1.0, "max_y": 5.0,
        "min_z": 2.0, "max_z": 6.0
    }
    domain.pop(missing_key)
    with pytest.raises(DomainValidationError, match="Missing domain bounds for axis"):
        validate_domain_bounds(domain)

@pytest.mark.parametrize("bad_value", [
    {"min_x": "abc", "max_x": 10.0},
    {"min_y": 1.0, "max_y": "xyz"},
    {"min_z": "abc", "max_z": 6.0}  # ✅ Updated from None to "abc"
])
def test_non_numeric_values(bad_value):
    domain = {
        "min_x": 0.0, "max_x": 10.0,
        "min_y": 1.0, "max_y": 5.0,
        "min_z": 2.0, "max_z": 6.0
    }
    domain.update(bad_value)
    with pytest.raises(DomainValidationError, match="Non-numeric bounds for axis"):
        validate_domain_bounds(domain)

@pytest.mark.parametrize("axis,min_val,max_val", [
    ("x", 10.0, 5.0),
    ("y", 3.0, 2.0),
    ("z", 7.0, 6.0)
])
def test_invalid_logical_bounds(axis, min_val, max_val):
    domain = {
        "min_x": 0.0, "max_x": 10.0,
        "min_y": 1.0, "max_y": 5.0,
        "min_z": 2.0, "max_z": 6.0
    }
    domain[f"min_{axis}"] = min_val
    domain[f"max_{axis}"] = max_val
    with pytest.raises(DomainValidationError, match=f"Invalid domain: max_{axis}"):
        validate_domain_bounds(domain)



