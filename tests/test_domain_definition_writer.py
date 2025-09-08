# tests/test_domain_definition_writer.py

import pytest
import json
from jsonschema import validate, ValidationError
from pathlib import Path
from src.domain_definition_writer import validate_domain_bounds, DomainValidationError

# 📍 Normalize schema path
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "domain_schema.json"

def load_schema():
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def domain_schema():
    return load_schema()

# ────────────────────────────────
# 🔬 Domain Bounds Validation Unit Tests
# ────────────────────────────────

def test_valid_domain_bounds():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    validate_domain_bounds(domain)

@pytest.mark.parametrize("axis, min_val, max_val", [
    ("x", 5.0, 1.0),
    ("y", 2.0, 1.5),
    ("z", 10.0, 0.0),
])
def test_invalid_bounds_trigger_exception(axis, min_val, max_val):
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    domain[f"min_{axis}"] = min_val
    domain[f"max_{axis}"] = max_val

    with pytest.raises(DomainValidationError) as exc:
        validate_domain_bounds(domain)
    assert axis in str(exc.value)

@pytest.mark.parametrize("missing_key", [
    "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"
])
def test_missing_keys_trigger_exception(missing_key):
    """
    Tests whether validate_domain_bounds raises DomainValidationError
    when domain keys are missing. Exception messages are validated
    by expected axis ('x', 'y', 'z') identification.
    """
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    domain.pop(missing_key)

    with pytest.raises(DomainValidationError) as exc:
        validate_domain_bounds(domain)

    axis = missing_key[-1]
    assert f"axis '{axis}'" in str(exc.value)
    assert isinstance(exc.value, DomainValidationError)

def test_non_numeric_values_are_invalid():
    domain = {
        "min_x": "zero", "max_x": 5.0,
        "min_y": 0.0, "max_y": "five",
        "min_z": None, "max_z": 3.0
    }
    with pytest.raises(DomainValidationError) as exc:
        validate_domain_bounds(domain)
    assert "Non-numeric bounds" in str(exc.value)

def test_extremely_large_float_bounds():
    domain = {
        "min_x": -1e300, "max_x": 1e300,
        "min_y": -1e-12, "max_y": 1e-12,
        "min_z": 0.0, "max_z": 1.0
    }
    validate_domain_bounds(domain)

# ────────────────────────────────
# 📐 JSON Schema Validation Tests
# ────────────────────────────────

def test_payload_matches_schema(domain_schema):
    payload = {
        "domain_definition": {
            "geometry_mask_flat": [0, 1, 0],
            "geometry_mask_shape": [3, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }
    validate(instance=payload, schema=domain_schema)
    assert True

@pytest.mark.parametrize("key", [
    "geometry_mask_flat", "geometry_mask_shape", "mask_encoding", "flattening_order"
])
def test_missing_keys_trigger_validation_error(domain_schema, key):
    """
    Tests that removing any required key from the domain_definition
    causes a schema validation error, and that the error message
    mentions the missing key.
    """
    domain = {
        "geometry_mask_flat": [0, 1, 0],
        "geometry_mask_shape": [3, 1, 1],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }
    domain.pop(key)
    payload = {"domain_definition": domain}

    with pytest.raises(ValidationError) as exc:
        validate(instance=payload, schema=domain_schema)
    assert key in str(exc.value)

def test_flat_payload_structure_rejected(domain_schema):
    flat_payload = {
        "geometry_mask_flat": [0, 1, 0],
        "geometry_mask_shape": [3, 1, 1],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }
    with pytest.raises(ValidationError):
        validate(instance=flat_payload, schema=domain_schema)

def test_extra_properties_rejected(domain_schema):
    payload = {
        "domain_definition": {
            "geometry_mask_flat": [0, 1, 0],
            "geometry_mask_shape": [3, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major",
            "extra": 99
        }
    }
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=domain_schema)



