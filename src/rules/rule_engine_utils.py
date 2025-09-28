# 📄 src/rules/rule_engine_utils.py

import logging
from src.rules.config import debug_log

logger = logging.getLogger(__name__)

class RuleEvaluationError(Exception):
    """Raised when a rule evaluation fails due to missing keys or invalid logic."""

def is_symbolic_reference(val: str) -> bool:
    return isinstance(val, str) and '.' in val and not val.replace('.', '', 1).isdigit()



