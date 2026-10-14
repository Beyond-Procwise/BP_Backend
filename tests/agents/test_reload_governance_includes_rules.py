"""Hot-reload covers rules as well as prompts and policies.

Editing a threshold in proc.bp_rule should take effect the same way editing a
prompt does. Without this the rule book is the one governance table that needs
a service restart, which is how tables quietly stop being edited.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.routers.agents import _do_reload_governance
from engines.rule_book import RuleBook


def _rows(window):
    return [
        {
            "rule_id": 1,
            "rule_name": "Contract Expiry Opportunity",
            "detector_slug": "contract_expiry_check",
            "finding_type": "opportunity",
            "scope": "contracts",
            "required_fields": json.dumps(["negotiation_window_days"]),
            "conditions": json.dumps({"negotiation_window_days": window}),
            "severity": "medium",
            "rule_status": 1,
            "version": 1,
        }
    ]


def test_reload_governance_reports_and_refreshes_the_rule_book(monkeypatch):
    book = RuleBook(rule_rows=_rows(90))
    nick = SimpleNamespace(
        policy_engine=SimpleNamespace(
            reload_policies=lambda: None, list_policies=lambda: []
        ),
        prompt_engine=SimpleNamespace(refresh=lambda: None, all_prompts=lambda: []),
        rule_book=book,
    )

    from src.services.extraction_feedback.hint_store import HINT_STORE

    monkeypatch.setattr(HINT_STORE, "refresh", lambda: 0)
    # The operator has just edited the threshold in the database.
    monkeypatch.setattr(book, "_fetch_rows", lambda: _rows(30))

    report = _do_reload_governance(nick)

    assert report["rules"] == 1
    assert book.rule_for("contract_expiry_check").conditions == {
        "negotiation_window_days": 30
    }
