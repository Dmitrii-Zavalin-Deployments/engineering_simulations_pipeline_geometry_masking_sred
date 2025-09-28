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



