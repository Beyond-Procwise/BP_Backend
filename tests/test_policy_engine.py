import os
import sys

import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.policy_engine import PolicyEngine


def _policy_rows():
    details = {
        "rules": {
            "default_weights": {"price": 0.6, "delivery": 0.4},
        }
    }
    return [
        {
            "policy_id": 1,
            "policy_name": "WeightAllocationPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Default supplier ranking weights",
            "policy_details": json.dumps(details),
        }
    ]


def test_valid_criteria_allowed():
    engine = PolicyEngine(policy_rows=_policy_rows())
    result = engine.validate_workflow('supplier_ranking', 'user1', {'criteria': ['price', 'delivery']})
    assert result['allowed'] is True


def test_invalid_criteria_blocked():
    engine = PolicyEngine(policy_rows=_policy_rows())
    result = engine.validate_workflow('supplier_ranking', 'user1', {'criteria': ['unknown']})
    assert result['allowed'] is False


def test_missing_criteria_defaults_to_policy_weights():
    engine = PolicyEngine(policy_rows=_policy_rows())
    result = engine.validate_workflow('supplier_ranking', 'user1', {})
    assert result['allowed'] is True


def test_policy_engine_queries_bp_policy_table():
    captured = {}

    class CapturingCursor:
        def __init__(self):
            self.description = None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def execute(self, query, params=None):
            captured["query"] = query
        def fetchall(self):
            return []

    class CapturingConn:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def cursor(self):
            return CapturingCursor()

    PolicyEngine(connection_factory=lambda: CapturingConn())
    assert "proc.bp_policy" in captured["query"]
    assert "proc.policy " not in captured["query"]
