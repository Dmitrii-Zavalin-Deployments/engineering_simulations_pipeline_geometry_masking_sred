# 📄 src/rules/operators.py

import re

# 🔗 Centralized operator function definitions
def op_eq(a, b): return a == b
def op_ne(a, b): return a != b
def op_lt(a, b): return a < b
def op_le(a, b): return a <= b
def op_gt(a, b): return a > b
def op_ge(a, b): return a >= b
def op_in(a, b): return a in b
def op_not_in(a, b): return a not in b

def op_matches(a, b):
    """Regex match — assumes 'b' is a regex pattern"""
    if not isinstance(b, str):
        raise TypeError("Regex pattern must be a string")
    return re.fullmatch(b, str(a)) is not None



