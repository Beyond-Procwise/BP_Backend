"""Send-time: a draft whose facts moved after it was written.

Shadow reports it and sends; enforce refuses. Each branch is paired with its
opposite so the check is seen both firing and staying quiet.
"""

import copy
from decimal import Decimal

import pytest

from src.services import draft_assurance as da
from src.services import email_dispatch_guard as guard
from tests.guardrails.test_send_path_gate import base_kwargs
from tests.services.test_draft_assurance import (GOOD, KEYS, TABLES, FakeConn as TableConn,
                                                 _family_rules, _inputs, family)  # noqa: F401


@pytest.fixture(autouse=True)
def _ready_by_default(monkeypatch):
    """These tests are about moved facts; readiness has its own tests below."""
    from src.services.draft_assurance import capture
    monkeypatch.setattr(capture, "readiness", lambda conn, uid: {"ready": True, "needs_redraft": False})


class GateConn:
    """The gate suite's fake (lookup seams) plus a cursor over the fact tables."""

    def __init__(self, gate_conn, tables):
        self._gate, self._tables = gate_conn, TableConn(tables)

    def cursor(self):
        return self._tables.cursor()

    def __getattr__(self, name):
        return getattr(self._gate, name)


class Engine:
    """The gate suite's policy engine, plus the family row."""

    def __init__(self, inner, mode):
        self._inner, self._mode = inner, mode

    def get_policy(self, slug):
        if slug == "email_family_negotiation_counter":
            rules = _family_rules()
            rules["mode"] = self._mode
            return {"details": {"rules": rules}, "version": 1}
        return self._inner.get_policy(slug)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _assured_draft(family):
    record = _inputs(family).finalize(GOOD, [], [])
    kwargs = base_kwargs()
    draft = dict(kwargs["draft"])
    draft["assurance"] = record
    return kwargs, draft


def _run(family, mode, tables):
    kwargs, draft = _assured_draft(family)
    kwargs["draft"] = draft
    kwargs["conn"] = GateConn(kwargs["conn"], tables)
    kwargs["policy_engine"] = Engine(kwargs["policy_engine"], mode)
    return guard.check_dispatch(**kwargs)


def _moved():
    t = copy.deepcopy(TABLES)
    t["supplier_response"][1]["price"] = Decimal("45.00")
    return t


def test_unchanged_facts_pass_in_both_modes(family):
    for mode in ("shadow", "enforce"):
        d = _run(family, mode, TABLES)
        assert d.allowed is True, (mode, d.reason)
        assert d.evidence["facts_recheck"] == {"checked": True, "mode": mode, "changed": []}


def test_shadow_reports_a_moved_fact_but_still_sends(family):
    d = _run(family, "shadow", _moved())
    assert d.allowed is True
    changed = d.evidence["facts_recheck"]["changed"]
    assert [c["fact"] for c in changed] == ["supplier_current_offer"]
    assert (changed[0]["was"], changed[0]["now"]) == ("47.50", "45.00")


def test_enforce_refuses_a_moved_fact_and_names_it(family):
    d = _run(family, "enforce", _moved())
    assert d.allowed is False
    assert "supplier_current_offer" in d.reason
    assert d.evidence["facts_recheck"]["changed"][0]["now"] == "45.00"


def test_enforce_refuses_when_the_row_has_gone(family):
    d = _run(family, "enforce", {})
    assert d.allowed is False


def test_an_unreadable_family_uses_the_mode_stored_on_the_draft(family):
    kwargs, draft = _assured_draft(family)
    draft["assurance"]["mode"] = "enforce"
    kwargs["draft"] = draft
    kwargs["conn"] = GateConn(kwargs["conn"], TABLES)
    # base engine knows no family row -> load fails
    d = guard.check_dispatch(**kwargs)
    assert d.allowed is False and "re-checked" in d.reason


def test_an_unreadable_family_on_a_shadow_draft_still_sends_and_says_so(family):
    kwargs, draft = _assured_draft(family)
    kwargs["draft"] = draft
    kwargs["conn"] = GateConn(kwargs["conn"], TABLES)
    d = guard.check_dispatch(**kwargs)
    assert d.allowed is True
    assert d.evidence["facts_recheck"]["checked"] is False


def test_drafts_without_assurance_are_untouched():
    d = guard.check_dispatch(**base_kwargs())
    assert d.allowed is True
    assert d.evidence["facts_recheck"]["checked"] is False


# --- who reviewed it, and whether it is ready --------------------------------------------------

def _gate(family, *, approval=None, accountability=None, mode="shadow", readiness=None, monkeypatch=None):
    from src.services.draft_assurance import capture
    if readiness is not None:
        monkeypatch.setattr(capture, "readiness", lambda conn, uid: readiness)
    kwargs, draft = _assured_draft(family)
    draft["assurance"]["mode"] = mode
    if accountability is not None:
        draft["assurance"]["accountability"] = accountability
    kwargs["draft"] = draft
    kwargs["conn"] = GateConn(kwargs["conn"], TABLES)
    kwargs["policy_engine"] = Engine(kwargs["policy_engine"], mode)
    if approval is not None:
        base = kwargs["approval_lookup"]()
        kwargs["approval_lookup"] = lambda **_: {**base, **approval}
    return guard.check_dispatch(**kwargs)


def test_a_draft_started_by_an_agent_and_reviewed_by_a_person_may_go(family):
    d = _gate(family, accountability={"initiated_by": "NegotiationAgent", "kind": "agent"})
    assert d.allowed is True and d.evidence["reviewed_by"] == "buyer@ourcompany.com"


def test_an_assured_draft_whose_approval_names_no_person_cannot_be_sent(family):
    d = _gate(family, approval={"actioned_by": ""}, accountability={"initiated_by": "NegotiationAgent", "kind": "agent"})
    assert d.allowed is False and "no person" in d.reason


def test_an_agent_cannot_be_its_own_reviewer(family):
    d = _gate(family, approval={"actioned_by": "NegotiationAgent"},
              accountability={"initiated_by": "NegotiationAgent", "kind": "agent"})
    assert d.allowed is False and "agent that wrote" in d.reason


def test_an_agent_initiated_draft_with_no_human_review_cannot_go_autonomously(family):
    d = _gate(family, approval={"autonomous": True, "agent": "email_dispatch_agent", "actioned_by": None},
              accountability={"initiated_by": "NegotiationAgent", "kind": "agent"})
    assert d.allowed is False and "no human" in d.reason


def test_the_reviewer_rule_does_not_apply_to_drafts_with_no_assurance_record():
    d = guard.check_dispatch(**base_kwargs())
    assert d.allowed is True and d.evidence["readiness"]["ready"] is None


def test_an_unready_draft_is_reported_in_shadow_and_refused_in_enforce(family, monkeypatch):
    unready = {"ready": False, "needs_redraft": False}
    shadow = _gate(family, mode="shadow", readiness=unready, monkeypatch=monkeypatch)
    assert shadow.allowed is True and shadow.evidence["readiness"]["ready"] is False
    enforce = _gate(family, mode="enforce", readiness=unready, monkeypatch=monkeypatch)
    assert enforce.allowed is False and "not ready" in enforce.reason


def test_unknown_readiness_is_refused_in_enforce_and_reported_in_shadow(family, monkeypatch):
    unknown = {"ready": None, "reason": "draft was never captured"}
    assert _gate(family, mode="shadow", readiness=unknown, monkeypatch=monkeypatch).allowed is True
    d = _gate(family, mode="enforce", readiness=unknown, monkeypatch=monkeypatch)
    assert d.allowed is False and "never captured" in d.reason
