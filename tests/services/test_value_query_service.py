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

def test_send_stamps_query_sent_and_audits(conn, monkeypatch, _no_audit):
    sent = {}
    monkeypatch.setattr(vq, "_send_email",
                        lambda **kw: sent.update(kw) or True)
    out = vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                        agent_nick=object(), conn=conn)
    assert out["status"] == "sent" and out["query_sent_at"]
    assert sent["to"] == "ap@techworld.example" and sent["subject"] == "S"

    updates = [s for s, _ in conn.log if "UPDATE proc.bp_extraction_discrepancy" in s]
    assert len(updates) == 1 and "query_sent_at" in updates[0]
    assert conn.committed == 1
    assert _no_audit and _no_audit[0]["agent"] == "value_found_query"
    assert _no_audit[0]["doc_pk"] == "INV-1042"


def test_send_failure_leaves_the_row_untouched(conn, monkeypatch):
    def _boom(**kw):
        raise RuntimeError("SES refused")
    monkeypatch.setattr(vq, "_send_email", _boom)

    with pytest.raises(RuntimeError, match="SES refused"):
        vq.send_query("disc:80", to="ap@techworld.example", subject="S", body="B",
                      agent_nick=object(), conn=conn)
    # Nothing stamped, nothing committed: an unsent query must never look sent.
    assert not [s for s, _ in conn.log if "UPDATE" in s]
    assert conn.committed == 0


def test_send_reports_a_refusal_without_raising_as_a_failure(conn, monkeypatch):
    # send_email returns success=False rather than raising — still a failure, still no stamp.
    monkeypatch.setattr(vq, "_send_email", lambda **kw: False)
    with pytest.raises(RuntimeError, match="not accepted"):
        vq.send_query("disc:80", to="a@b.example", subject="S", body="B",
                      agent_nick=object(), conn=conn)
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


# --- the router ----------------------------------------------------------

def test_the_router_maps_refusals_and_delivery_failures_apart(monkeypatch):
    """A finding that cannot be queried and a query that could not be delivered are
    different problems: one is the caller's, one is ours. Retrying helps only the second,
    so they must not share a status code."""
    from fastapi import HTTPException
    from src.api.routers import value_summary as router

    monkeypatch.setattr(router.value_query_service, "build_draft",
                        lambda fid: (_ for _ in ()).throw(ValueError("not open")))
    with pytest.raises(HTTPException) as exc:
        router.get_query_draft("disc:80")
    assert exc.value.status_code == 409

    class _App:
        class state:
            agent_nick = object()

    class _Req:
        app = _App()

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
