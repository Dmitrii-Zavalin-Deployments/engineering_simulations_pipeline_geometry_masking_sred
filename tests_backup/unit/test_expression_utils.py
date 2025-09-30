import pytest
from src.validation.expression_utils import (
    normalize_quotes,
    parse_literal,
    is_literal,
    is_symbolic_reference,
    is_valid_numeric_literal
)

# --- Test normalize_quotes ---
@pytest.mark.parametrize("input_expr, expected_output", [
    # Redundant quotes
    ("'''hello'''", "'hello'"),
    ('"""world"""', '"world"'),
    ("''string''", "'string'"),
    ('""data""', '"data"'),
    
    # Mixed quotes (should handle inner-most)
    ("''' 'inner' '''", "' 'inner' '"), # Original logic is simple replacement, not recursive parsing
    
    # Already clean
    ("'clean'", "'clean'"),
    ('"clean"', '"clean"'),
    ("numeric", "numeric"),
    
    # Edge cases
    ("", ""),
    (" ", ""),
    (" ' ' ", "' '"),
    ("'''", "'"), # Three quotes becomes one
    ('"""', '"'), # Three quotes becomes one
])
def test_normalize_quotes(input_expr, expected_output):
    """Test that redundant or deeply nested quotes are correctly normalized."""
    assert normalize_quotes(input_expr) == expected_output


# --- Test parse_literal ---
@pytest.mark.parametrize("input_value, expected_type, expected_output", [
    # Numerics
    ("42", int, 42),
    ("3.1415", float, 3.1415),
    ("-100", int, -100),
    ("0.0", float, 0.0),
    ("00123", int, 123), # Handles zero padding via isdigit/int conversion

    # Booleans/Nulls (case-insensitive)
    ("true", bool, True),
    ("TRUE", bool, True),
    ("True", bool, True),
    ("false", bool, False),
    ("null", type(None), None),
    ("NULL", type(None), None),
    ("none", type(None), None),

    # Strings (with and without quotes)
    ("'hello'", str, "hello"),
    ('"world"', str, "world"),
    ("unquoted string", str, "unquoted string"),
    ("'42'", str, "42"), # Should be treated as a string, not int
    ('"3.14"', str, "3.14"), # Should be treated as a string, not float
    
    # Nested strings (after quote normalization)
    ("'''nested'''", str, "nested"),
    
    # Non-string input (should be returned as is)
    (123, int, 123),
    (True, bool, True),
    (None, type(None), None),
    
    # Expressions that should fail ast.literal_eval and return as string
    ("x.y", str, "x.y"),
    ("[1, 2]", list, [1, 2]), # ast.literal_eval should handle lists
])
def test_parse_literal_conversions(input_value, expected_type, expected_output):
    """Test safe conversion of string representations to Python literals."""
    result = parse_literal(input_value)
    assert result == expected_output
    assert isinstance(result, expected_type)

def test_parse_literal_complex_types():
    """Test parse_literal handling of complex types that ast.literal_eval can handle."""
    # List
    assert parse_literal("[1, 2, 'three']") == [1, 2, 'three']
    assert isinstance(parse_literal("[1, 2, 'three']"), list)
    
    # Dictionary
    assert parse_literal("{'a': 1, 'b': 'two'}") == {'a': 1, 'b': 'two'}
    assert isinstance(parse_literal("{'a': 1, 'b': 'two'}"), dict)
    
    # Tuple
    assert parse_literal("(1, 2)") == (1, 2)
    assert isinstance(parse_literal("(1, 2)"), tuple)

def test_parse_literal_non_literal_fallback():
    """Test that invalid expressions are returned as strings (fallback)."""
    assert parse_literal("x + 1") == "x + 1"
    assert parse_literal("system.variable") == "system.variable"
    assert parse_literal("$variable") == "$variable"

# --- Test is_literal ---
@pytest.mark.parametrize("token, expected", [
    # Primitive types
    ("true", True),
    ("FALSE", True),
    ("null", True),
    ("NONE", True),
    
    # Numeric
    ("42", True),
    ("-10", False), # is_numeric() check is simple and does not support '-'
    ("3.14", False), # is_numeric() check is simple and does not support '.'
    
    # String literals
    ("'hello'", True),
    ('"world"', True),
    (" ' 42 ' ", True),
    
    # Non-literals (should return False)
    ("variable.name", False),
    ("10.5", False),
    ("-5", False),
    ("not literal", False),
])
def test_is_literal(token, expected):
    """Test detection of primitive literal tokens."""
    assert is_literal(token) == expected

# --- Test is_symbolic_reference ---
@pytest.mark.parametrize("input_val, expected", [
    # True symbolic references
    ("x.y", True),
    ("system.status.code", True),
    ("user[id]", True),
    ("data[1].value", True),
    
    # False (Literals/Numerics)
    ("42", False),
    ("150.0", False),
    ("-3.14", False),
    ("true", False),
    ("Null", False),
    ("'string'", False),
    ('"string"', False),
    
    # Edge Cases
    ("", False),
    (" ", False),
    ("4.2.1", True), # Contains a '.', so it is considered symbolic (e.g. version string)
    ("42[id]", True), # Contains brackets
    (123, False), # Non-string input
])
def test_is_symbolic_reference(input_val, expected):
    """Test detection of symbolic references (variables, paths) excluding literals."""
    assert is_symbolic_reference(input_val) == expected

# --- Test is_valid_numeric_literal ---
@pytest.mark.parametrize("input_val, expected", [
    # Valid Numerics
    ("42", True),
    ("-100", True),
    ("3.1415", True),
    ("1e5", True),
    (".5", True),
    
    # Invalid Numerics
    ("hello", False),
    ("42a", False),
    ("10.5.2", False),
    ("", False),
    (" ", False),
    ("true", False),
    ("null", False),
])
def test_is_valid_numeric_literal(input_val, expected):
    """Test safe determination of whether a string can be coerced to float."""
    assert is_valid_numeric_literal(input_val) == expected



