# 📄 src/rules/rule_engine_utils.py

import logging
from src.rules.config import debug_log

logger = logging.getLogger(__name__)

def is_symbolic_reference(val: str) -> bool:
    return isinstance(val, str) and '.' in val and not val.replace('.', '', 1).isdigit()



