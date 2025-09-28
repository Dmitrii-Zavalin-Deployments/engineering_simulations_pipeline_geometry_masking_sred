# 📄 tests/conftest.py

import sys
import pathlib
import pytest
from unittest.mock import patch, mock_open

# Adds src/ directory to sys.path for all tests
SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="function")
def mock_validate_step_file():
    with patch("os.path.isfile", return_value=True):
        with patch("src.utils.input_validation.validate_step_file", return_value=True) as mock_func:
            yield mock_func


def get_payload_with_defaults(overrides=None):
    base = {
        "hello": "world",
        "flag": True,
        "thresholds": {"warn_val": 150, "max_val": 150},
        "limits": {"upper": 10.0, "lower": 5.0},
        "metrics": {"score": 0.3},
        "values": {"x": 5},
        "system": {"subsystem": {"value": 42}},
        "expected": {"value": 42},
        "config": {"enabled": "true"},
        "domain_definition": {"max_z": 100.0, "min_z": 90.5},
        "a": {"b": 10},
        "x": {"y": 10},
        "rules": {"status_code": "not_a_number", "expected_code": 200},
    }

    if overrides:
        for key, value in overrides.items():
            if isinstance(base.get(key), dict) and not isinstance(value, dict):
                raise TypeError(f"Cannot override structured key '{key}' with scalar value: {value}")
            elif isinstance(base.get(key), dict) and isinstance(value, dict):
                base[key].update(value)
            else:
                base[key] = value
    return base



