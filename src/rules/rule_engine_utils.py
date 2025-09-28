# 📄 src/rules/rule_engine_utils.py

import logging
from src.rules.config import debug_log

logger = logging.getLogger(__name__)

class RuleEvaluationError(Exception):
    """Raised when a rule evaluation fails due to missing keys or invalid logic."""

def is_symbolic_reference(val: str) -> bool:
    return isinstance(val, str) and '.' in val and not val.replace('.', '', 1).isdigit()

def get_nested_value(payload: dict, path: str):
    keys = path.split(".")
    value = payload
    for k in keys:
        if not isinstance(value, dict):
            raise RuleEvaluationError(
                f"Expected dict at '{k}' in path '{path}', but got {type(value)}"
            )
        if k not in value:
            raise RuleEvaluationError(f"Missing key in expression: {path}")
        value = value[k]
        if value is None and k != keys[-1]:
            raise RuleEvaluationError(
                f"Null value encountered at '{k}' in path '{path}'"
            )
        debug_log(f"Resolved key '{k}' → {value}")
    return value



