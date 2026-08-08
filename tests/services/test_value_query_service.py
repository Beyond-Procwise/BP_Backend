"""Query-it: the one-click supplier query on a value finding.

This email accuses a supplier of over-billing or double-billing, so every figure in it has
to be the figure we actually stored. There is no model in this path — the template is
interpolated in Python from the discrepancy row, which makes grounding a property of the
construction rather than something to check afterwards.

The DB and SES are faked at the module boundary; everything else is the real code.
"""
from datetime import datetime, timezone

import pytest

from src.services import value_query_service as vq


# --- fakes ---------------------------------------------------------------

FINDING = {
    "discrepancy_id": 80,
    "issue_type": "amount_over_po",
    "status": "open",
    "doc_type": "invoice",
    "doc_pk_candidate": "INV-1042",
    "raw_value": "10950.00",
    "expected_value": "10000.00",
    "computed_value": "+950.00",
    "notes": "",
    "query_sent_at": None,
    "po_id": "PO-2210",
    "currency": "GBP",
    "deal_id": "D-1",
    "supplier_id": "SUP-Techworld",
    "supplier_name": "Techworld",
    "supplier_email": "ap@techworld.example",
}


class _FakeCursor:
    def __init__(self, row, log):
        self._row = row
        self._log = log
        self.description = None
        self._result = None

    def execute(self, sql, params=()):
        self._log.append((" ".join(sql.split()), params))
        if "FROM proc.bp_extraction_discrepancy" in sql:
            if self._row is None:
                self.description, self._result = None, []
                return
            cols = list(self._row.keys())
            self.description = [(c,) for c in cols]
            self._result = [tuple(self._row[c] for c in cols)]
        else:
            self.description, self._result = None, []

    def fetchall(self):
        return list(self._result or [])

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None


class _FakeConn:
    def __init__(self, row):
        self.row = row
        self.log = []
        self.committed = 0
        self.autocommit = False

    def cursor(self):
        return _FakeCursor(self.row, self.log)

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


@pytest.fixture
def conn():
    return _FakeConn(dict(FINDING))


@pytest.fixture(autouse=True)
def _no_governance(monkeypatch):
    """The governed override is read from bp_prompt; default to absent so the built-in
    template is what these tests exercise."""
    monkeypatch.setattr(vq, "_governed_template", lambda: None)


@pytest.fixture(autouse=True)
def _no_audit(monkeypatch):
    calls = []
    monkeypatch.setattr(vq.agent_actions, "record_action",
                        lambda **kw: calls.append(kw))
    return calls


class _FakePrincipal:
    """A minimal stand-in for api.auth.Principal (subject + claims)."""

    def __init__(self, subject="sub-approver", claims=None):
        self.subject = subject
        self.claims = claims or {"cognito:groups": ["bp-approvers"]}


@pytest.fixture
def _allow_guard(monkeypatch):
    """Stand the C1 guard aside for tests about the SEND MECHANICS --
    stamping, audit, error propagation -- rather than the guard itself,
    which has its own dedicated RED/GREEN proof below (and reuses the same
    checks proven in tests/guardrails/test_send_path_gate.py). This is the
    same kind of test double tests/test_email_dispatch_service.py already
    uses for its own non-guard tests.
    """
    monkeypatch.setattr(
        vq.guardrail, "authorize",
        lambda *a, **k: vq.guardrail.Decision(allowed=True, reason="test stub"),
    )
    monkeypatch.setattr(
        vq.email_dispatch_guard, "check_recipient_and_sensitivity",
        lambda **k: vq.guardrail.Decision(
            allowed=True, reason="test stub", evidence={"content_class": "internal"}
        ),
    )


# --- draft ---------------------------------------------------------------

def test_draft_figures_are_byte_equal_to_stored_values(conn):
    draft = vq.build_draft("disc:80", conn=conn)
    assert draft["to"] == "ap@techworld.example"
    assert draft["figures"] == {"delta": "950.00 GBP", "amount": "950.00", "currency": "GBP",
                                "doc_ref": "INV-1042", "po_ref": "PO-2210"}
    # The exact stored strings appear in the body — not a re-derived or re-rounded number.
    assert "950.00 GBP" in draft["body"]
    assert "INV-1042" in draft["body"] and "PO-2210" in draft["body"]
    assert "INV-1042" in draft["subject"] and "PO-2210" in draft["subject"]
    assert "Techworld" in draft["body"]


class _FakeAgent:
    """An agent that answers with whatever it was handed."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def call_ollama(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.reply}}


def test_the_default_tone_is_the_template_and_asks_no_model(conn):
    agent = _FakeAgent("something else")
    draft = vq.build_draft("disc:80", conn=conn, tone="formal", agent_nick=agent)
    assert draft["tone"] == "formal"
    assert draft["tone_applied"] is False
    assert agent.calls == [], "formal IS the stored template's voice"
    assert "950.00 GBP" in draft["body"]


def test_a_tone_rewrites_the_wording_and_keeps_every_figure(conn):
    rewritten = ("Hi Techworld Ltd,\n\nQuick one — INV-1042 looks 950.00 GBP over PO-2210. "
                 "Could you take a look?\n\nThanks")
    draft = vq.build_draft("disc:80", conn=conn, tone="direct",
                           agent_nick=_FakeAgent(rewritten))
    assert draft["tone_applied"] is True
    assert draft["body"] == rewritten
    # The subject and the figures are NOT the model's to change.
    assert draft["subject"] == vq.build_draft("disc:80", conn=conn)["subject"]
    assert draft["figures"]["delta"] == "950.00 GBP"


def test_a_tone_that_moves_the_money_is_thrown_away_and_admitted(conn):
    poisoned = "Hi Techworld Ltd, INV-1042 is 950.00 GBP over PO-2210, plus 190.00 GBP VAT."
    draft = vq.build_draft("disc:80", conn=conn, tone="warm",
                           agent_nick=_FakeAgent(poisoned))
    assert draft["tone_applied"] is False
    # The buyer gets the grounded draft, and is TOLD the tone did not take -- otherwise
    # they send a formal email believing it was warmed up.
    assert "950.00 GBP" in draft["body"] and "190.00" not in draft["body"]
    assert draft["tone_note"] and "190.00" in draft["tone_note"]


def test_the_template_is_interpolated_not_written_by_a_model():
    # Grounding by construction: every figure slot is a placeholder the service fills from
    # the row. If this ever became a generated body, these placeholders would disappear.
    for slot in ("{delta}", "{doc_ref}", "{po_ref}", "{supplier_name}"):
        assert slot in vq.DEFAULT_TEMPLATE["body"], slot
    assert "{doc_ref}" in vq.DEFAULT_TEMPLATE["subject"]


def test_a_duplicate_invoice_query_cites_both_invoices(conn):
    conn.row.update({
        "issue_type": "duplicate_invoice",
        "doc_pk_candidate": "INV-1042A",
        "computed_value": "+1321.06",
        "notes": "possible duplicate of INV-1042 (2024-05-20): relationship score 99.5/100",
    })
    draft = vq.build_draft("disc:80", conn=conn)
    # Both references, because "you billed us twice" is meaningless without naming the pair.
    assert "INV-1042A" in draft["body"] and "INV-1042" in draft["body"]
    # Thousands-separated for a human reader, but the same number to the penny: the only
    # transformation allowed between the stored value and the email is grouping.
    assert draft["figures"]["delta"] == "1,321.06 GBP"
    assert draft["figures"]["amount"].replace(",", "") == "1321.06"
    assert draft["figures"]["delta"] in draft["body"]
    assert draft["figures"]["duplicate_of"] == "INV-1042"


def test_draft_without_a_supplier_email_still_drafts_but_cannot_be_addressed(conn):
    conn.row["supplier_email"] = None
    draft = vq.build_draft("disc:80", conn=conn)
    assert draft["to"] is None            # the router turns this into a 409 on SEND, not here
    assert "950.00 GBP" in draft["body"]  # the draft is still useful to a human


def test_draft_refuses_a_finding_that_is_not_open(conn):
    conn.row["status"] = "resolved"
    with pytest.raises(ValueError, match="not open"):
        vq.build_draft("disc:80", conn=conn)


def test_draft_refuses_a_non_value_issue_type(conn):
    conn.row["issue_type"] = "po_not_found"
    with pytest.raises(ValueError, match="not a value finding"):
        vq.build_draft("disc:80", conn=conn)


def test_draft_refuses_an_opportunity_id(conn):
    # Only discrepancies are queryable — an opportunity has no supplier to challenge.
    with pytest.raises(ValueError, match="only discrepancy findings"):
        vq.build_draft("opp:14648", conn=conn)


def test_draft_refuses_an_unknown_finding():
    with pytest.raises(ValueError, match="not found"):
        vq.build_draft("disc:9999", conn=_FakeConn(None))


def test_an_amount_with_no_currency_goes_out_bare_never_with_a_guessed_symbol(conn):
    conn.row["currency"] = None
    draft = vq.build_draft("disc:80", conn=conn)
    assert draft["figures"]["delta"] == "950.00"
    assert draft["figures"]["currency"] == ""
    assert "£" not in draft["body"] and "$" not in draft["body"]


def test_a_governed_template_overrides_the_default(conn, monkeypatch):
    monkeypatch.setattr(vq, "_governed_template", lambda: {
        "subject": "Re {doc_ref}", "body": "We were billed {delta} extra on {doc_ref}."})
    draft = vq.build_draft("disc:80", conn=conn)
    assert draft["subject"] == "Re INV-1042"
    assert draft["body"] == "We were billed 950.00 GBP extra on INV-1042."


# --- send ----------------------------------------------------------------

def test_send_stamps_query_sent_and_audits(conn, monkeypatch, _no_audit, _allow_guard):
    sent = {}
    monkeypatch.setattr(vq, "_send_email",
                        lambda **kw: sent.update(kw) or True)
    out = vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                        agent_nick=object(), principal=_FakePrincipal(), conn=conn)
    assert out["status"] == "sent" and out["query_sent_at"]
    assert sent["to"] == "ap@techworld.example" and sent["subject"] == "S"

    updates = [s for s, _ in conn.log if "UPDATE proc.bp_extraction_discrepancy" in s]
    assert len(updates) == 1 and "query_sent_at" in updates[0]
    assert conn.committed == 1
    assert _no_audit and _no_audit[0]["agent"] == "value_found_query"
    assert _no_audit[0]["doc_pk"] == "INV-1042"


def test_send_failure_leaves_the_row_untouched(conn, monkeypatch, _allow_guard):
    def _boom(**kw):
        raise RuntimeError("SES refused")
    monkeypatch.setattr(vq, "_send_email", _boom)

    with pytest.raises(RuntimeError, match="SES refused"):
        vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                      agent_nick=object(), principal=_FakePrincipal(), conn=conn)
    # Nothing stamped, nothing committed: an unsent query must never look sent.
    assert not [s for s, _ in conn.log if "UPDATE" in s]
    assert conn.committed == 0


def test_send_reports_a_refusal_without_raising_as_a_failure(conn, monkeypatch, _allow_guard):
    # send_email returns success=False rather than raising — still a failure, still no stamp.
    monkeypatch.setattr(vq, "_send_email", lambda **kw: False)
    with pytest.raises(RuntimeError, match="not accepted"):
        vq.send_query("disc:80", to="a@b.example", subject="S", body="B",
                      agent_nick=object(), principal=_FakePrincipal(), conn=conn)
    assert not [s for s, _ in conn.log if "UPDATE" in s]


def test_send_rechecks_the_finding_is_still_open(conn, monkeypatch):
    conn.row["status"] = "resolved"
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)
    with pytest.raises(ValueError, match="not open"):
        vq.send_query("disc:80", to="a@b.example", subject="S", body="B",
                      agent_nick=object(), conn=conn)


def test_send_refuses_a_query_already_sent(conn, monkeypatch):
    conn.row["query_sent_at"] = datetime.now(timezone.utc)
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)
    with pytest.raises(ValueError, match="already"):
        vq.send_query("disc:80", to="a@b.example", subject="S", body="B",
                      agent_nick=object(), conn=conn)


def test_send_refuses_an_empty_recipient(conn, monkeypatch):
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="recipient"):
            vq.send_query("disc:80", to=bad, subject="S", body="B",
                          agent_nick=object(), conn=conn)


# --- C1: the send route must run the same gate the RFQ dispatch path does ---
#
# This route used to carry a caller-supplied `to`/`subject`/`body` straight to
# SES for any finding id -- no approval, no allow-list, no sensitivity check,
# no policy call. These tests exercise the REAL guardrail.authorize and
# email_dispatch_guard.check_recipient_and_sensitivity (no `_allow_guard`
# stub), so they prove the actual wiring, not a double standing in for it.

class _AllowlistConn:
    """Wraps a `_FakeConn` and answers the guard's own lookups, so checks 2
    (allow-list) and 3 (sensitivity) pass on real data rather than a stub."""

    def __init__(self, inner, *, emails=("ap@techworld.example",), clearance="internal"):
        self._inner = inner
        self._emails = set(emails)
        self._clearance = clearance

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def lookup_supplier_emails(self, supplier_id):
        return set(self._emails)

    def lookup_supplier_clearance(self, supplier_id):
        return self._clearance


def test_send_denies_a_recipient_not_on_the_supplier_allowlist(conn, monkeypatch):
    """The recipient still has to be a real address on the supplier master.
    The fake connection here answers no supplier-email rows at all, so any
    address is "unknown" -- the same failure mode check_dispatch's own
    allow-list check produces for the RFQ path."""
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)
    with pytest.raises(vq.DispatchDenied) as excinfo:
        vq.send_query("disc:80", to="someone-else@random.example", subject="S", body="B",
                      agent_nick=object(), principal=_FakePrincipal(), conn=conn)
    assert "allow-list" in excinfo.value.decision.reason.lower()
    assert not [s for s, _ in conn.log if "UPDATE" in s]


def test_send_denies_when_the_finding_has_no_supplier_on_file(conn, monkeypatch):
    """No supplier_id means no contact_email_1/2 to check the recipient
    against. Skipping the check because the join was empty would be exactly
    the silent bypass this layer exists to close -- it must deny instead."""
    conn.row["supplier_id"] = None
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)
    with pytest.raises(vq.DispatchDenied) as excinfo:
        vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                      agent_nick=object(), principal=_FakePrincipal(), conn=conn)
    assert "no supplier on file" in excinfo.value.decision.reason.lower()
    assert not [s for s, _ in conn.log if "UPDATE" in s]


def test_send_denies_without_an_authenticated_principal_even_when_recipient_and_content_are_fine(
    conn, monkeypatch
):
    """Recipient and content are both fine (the wrapped conn answers the
    allow-list and clearance lookups, and an explicit policy_engine gives
    email_sensitivity real rules to classify against); the only thing
    missing is a principal, and guardrail.authorize's own fail-closed rule
    for irreversible actions refuses it -- exactly as it does for the RFQ
    dispatch path, and regardless of what any policy row says.

    A near-duplicate of this test, sharing this exact name, previously
    existed a few lines above using the plain (un-wrapped) `conn` fixture
    with no principal and no allow-list data configured. Python silently
    let the second definition shadow the first, so the first NEVER RAN --
    and when un-shadowed and probed, it failed: with no allow-list data at
    all, the recipient denies on check 2 before the principal is ever
    checked, so its assertion ("no authenticated principal") did not match
    what actually happens ("recipient not on the supplier allow-list").
    That case is exactly the coverage
    test_send_denies_a_recipient_not_on_the_supplier_allowlist already
    provides, so the broken duplicate was deleted rather than kept under a
    new name.
    """
    from tests.guardrails.test_send_path_gate import engine as _guard_engine

    conn.row["supplier_email"] = "ap@techworld.example"
    wrapped = _AllowlistConn(conn)
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)
    with pytest.raises(vq.DispatchDenied) as excinfo:
        vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                      agent_nick=object(), policy_engine=_guard_engine(), conn=wrapped)
    assert "no authenticated principal" in excinfo.value.decision.reason
    assert not [s for s, _ in conn.log if "UPDATE" in s]


def test_send_denial_is_audited_with_the_strict_writer(conn, monkeypatch, _no_audit):
    """C1 requires record_action_or_fail, not the lenient writer, for this
    gate decision -- a failed audit write must abort the send rather than
    let it through unlogged."""
    monkeypatch.setattr(vq, "_send_email", lambda **kw: True)

    calls = []

    def _boom(**kw):
        calls.append(kw)
        raise vq.agent_actions.AuditWriteError("audit table is unreachable")

    monkeypatch.setattr(vq.agent_actions, "record_action_or_fail", _boom)

    with pytest.raises(vq.agent_actions.AuditWriteError):
        vq.send_query("disc:80", to="someone-else@random.example", subject="S", body="B",
                      agent_nick=object(), conn=conn)

    assert calls, "record_action_or_fail must be called for the gate decision"
    assert calls[0]["status"] == "denied"  # not on the allow-list -> the gate denies
    assert not [s for s, _ in conn.log if "UPDATE" in s]


def test_send_allowed_by_the_real_guard_is_audited_and_sent(conn, monkeypatch):
    """The positive case, end to end through the REAL guard: an Approver
    sending to an address that genuinely is on the supplier master, with
    routine (internal-clearance) content. `policy_engine` is injected
    explicitly (the same seam guardrail.authorize/check_dispatch already
    expose for tests) rather than depending on the process-wide rbac cache,
    which under pytest resolves to an in-memory stand-in with no rows and
    would fail this closed for reasons unrelated to what this test checks.
    """
    from tests.guardrails.test_send_path_gate import engine as _guard_engine, approver

    conn.row["supplier_email"] = "ap@techworld.example"
    wrapped = _AllowlistConn(conn)
    sent = {}
    monkeypatch.setattr(vq, "_send_email", lambda **kw: sent.update(kw) or True)

    out = vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                        agent_nick=object(), principal=approver(),
                        policy_engine=_guard_engine(), conn=wrapped)

    assert out["status"] == "sent"
    assert sent["to"] == "ap@techworld.example"


# --- the router ----------------------------------------------------------

def test_the_router_maps_refusals_and_delivery_failures_apart(monkeypatch):
    """A finding that cannot be queried and a query that could not be delivered are
    different problems: one is the caller's, one is ours. Retrying helps only the second,
    so they must not share a status code."""
    from fastapi import HTTPException
    from src.api.routers import value_summary as router

    class _App:
        class state:
            agent_nick = object()

    class _Req:
        app = _App()

    monkeypatch.setattr(router.value_query_service, "build_draft",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("not open")))
    with pytest.raises(HTTPException) as exc:
        router.get_query_draft("disc:80", _Req())
    assert exc.value.status_code == 409

    payload = router.QuerySend(to="a@b.example", subject="S", body="B")

    monkeypatch.setattr(router.value_query_service, "send_query",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("already queried")))
    with pytest.raises(HTTPException) as exc:
        router.post_query_send("disc:80", payload, _Req())
    assert exc.value.status_code == 409

    monkeypatch.setattr(router.value_query_service, "send_query",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("SES down")))
    with pytest.raises(HTTPException) as exc:
        router.post_query_send("disc:80", payload, _Req())
    assert exc.value.status_code == 502


def test_the_router_refuses_to_send_without_agent_nick(monkeypatch):
    from fastapi import HTTPException
    from src.api.routers import value_summary as router

    class _App:
        class state:
            agent_nick = None

    class _Req:
        app = _App()

    with pytest.raises(HTTPException) as exc:
        router.post_query_send("disc:80", router.QuerySend(to="a@b.example", subject="S",
                                                           body="B"), _Req())
    assert exc.value.status_code == 503
