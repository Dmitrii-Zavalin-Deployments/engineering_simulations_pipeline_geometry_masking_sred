import pytest
from src.validation.expression_utils import evaluate_expression

def test_evaluate_expression_basic_arithmetic():
    """Test basic arithmetic expressions."""
    assert evaluate_expression("1 + 2") == 3
    assert evaluate_expression("10 - 5") == 5
    assert evaluate_expression("3 * 4") == 12
    assert evaluate_expression("10 / 2") == 5.0
    assert evaluate_expression("(2 + 3) * 4") == 20

def test_evaluate_expression_with_variables():
    """Test expressions with context variables."""
    context = {"x": 5, "y": 10}
    assert evaluate_expression("x + y", context) == 15
    assert evaluate_expression("x * (y - 2)", context) == 40
    assert evaluate_expression("x / 2", context) == 2.5

def test_evaluate_expression_with_complex_variables():
    """Test expressions with complex variable names and nesting."""
    context = {"voxel_size": 2, "limit": 100}
    assert evaluate_expression("voxel_size * 50", context) == 100
    assert evaluate_expression("voxel_size < limit", context) is True

def test_evaluate_expression_with_unsupported_ops():
    """Test that unsupported operations raise a ValueError."""
    with pytest.raises(ValueError):
        evaluate_expression("1 ** 2")

def test_evaluate_expression_with_invalid_expression():
    """Test that invalid expressions raise an exception."""
    with pytest.raises(Exception):
        evaluate_expression("1 + ")

def test_evaluate_expression_with_missing_variable():
    """Test that a missing variable in context raises an exception."""
    context = {"x": 5}
    with pytest.raises(NameError):
        evaluate_expression("x + y", context)
