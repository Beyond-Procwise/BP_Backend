"""Every action the Action Centre can take must write a status the table allows.

proc.bp_extraction_discrepancy carries two CHECK constraints (scripts/migrations/
2026-05-16-extraction-discrepancy-hitl.sql):

    status            IN ('open', 'resolved', 'ignored', 'superseded')
    resolution_action IS NULL OR IN ('apply_value', 'keep_null', 'dismiss')

`DecisionEngine.execute` wrote 'flagged', 'on_hold' and 'escalated' into status, and
the human's verb ('confirm', 'approve', 'reject', 'flag', …) into resolution_action.
Neither is permitted, so 9 of the 11 actions raised CheckViolation, were swallowed by
the engine's `except Exception`, and came back to the user as "could not update the
finding". Verified against bp_testdb and bp_sqldb: only apply_value and dismiss worked.

The mapping asserted here is the one the Node gateway already uses for the SAME table
(beyond-procwaise-Api spendiq.service.ts RESOLUTION_STATUS / ALLOWED_RESOLUTION_ACTIONS),
so the two writers agree:

    apply_value / confirm / approve  -> resolved   the finding is settled
    dismiss / reject                 -> ignored    deliberately set aside, not a fix
    flag / hold / escalate / …       -> open       escalated; it MUST stay open

resolution_action is NOT "what the human clicked" — src/services/extraction/promotion.py
switches on it to decide what happens to the raw extracted value ('apply_value' writes
the corrected figure, 'keep_null' nulls the field, 'dismiss' touches nothing). A verb
with no honest fit records NULL. What the human clicked is recorded on proc.bp_decision
by _record_human_action, which already stores the real verb.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# tests/conftest.py owns sys.path (repo root + src/).
from engines.decision_engine import DecisionEngine

# The two CHECK constraints, as the database actually defines them.
LEGAL_STATUS = {"open", "resolved", "ignored", "superseded"}
LEGAL_RESOLUTION_ACTION = {None, "apply_value", "keep_null", "dismiss"}

# Column order of the SELECT in DecisionEngine._fetch_finding.
_FINDING_COLUMNS = [
    "discrepancy_id", "doc_type", "source_file", "doc_pk_candidate",
    "field_name", "raw_value", "expected_value", "computed_value",
    "issue_type", "severity", "status", "notes", "blocks_promotion",
    "evidence_page", "evidence_text",
]

# A benign, low-value finding: a warning that blocks nothing, with a signed delta so
# the variance is 5.00 rather than a headline number.
_FINDING_ROW = (
    4242, "invoice", "documents/invoice/INV-1.pdf", "INV-1",
    "invoice_amount", "105.00", "100.00", "+5.00",
    "amount_over_po", "warning", "open", "billed above the purchase order", False,
    2, "Total 105.00",
)


class FakeCursor:
    """Serves _fetch_finding's SELECT and records the UPDATE that follows."""

    def __init__(self):
        self.description = None
        self.updates: list[tuple] = []
        self.update_sql: list[str] = []
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        norm = " ".join(sql.split())
        if norm.startswith("SELECT") and "proc.bp_extraction_discrepancy" in norm:
            self.description = [(c,) for c in _FINDING_COLUMNS]
            self._last_result = _FINDING_ROW
        elif norm.startswith("UPDATE proc.bp_extraction_discrepancy"):
            # Both branches of execute() put status first and resolution_action second.
            self.updates.append(params)
            self.update_sql.append(norm)
            self._last_result = None
        elif norm.startswith("INSERT INTO proc.bp_decision"):
            self._last_result = (555,)
        else:  # pragma: no cover - an unexpected query would show up here
            self._last_result = None

    def fetchone(self):
        return self._last_result


class FakeConn:
    def __init__(self, cur):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur

    def commit(self):
        pass


def _engine():
    cur = FakeCursor()
    nick = SimpleNamespace(
        get_db_connection=lambda: FakeConn(cur),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    return DecisionEngine(nick), cur


@pytest.mark.parametrize("action", sorted(DecisionEngine.KNOWN_ACTIONS))
def test_every_action_writes_a_status_the_table_allows(action):
    """No action may write a status or resolution_action the CHECK rejects."""
    eng, cur = _engine()

    # override_reason is supplied so a closing action on escalate-worthy evidence
    # proceeds to the write rather than returning requires_override — this test is
    # about what gets written, not about the human-in-the-loop gate.
    result = eng.execute(4242, action, user_id="tester", override_reason="under test")

    assert result["applied"] is True, f"{action} did not apply: {result.get('error')}"
    assert cur.updates, f"{action} wrote nothing"
    status, resolution_action = cur.updates[-1][0], cur.updates[-1][1]
    assert status in LEGAL_STATUS, (
        f"{action} writes status={status!r}, which "
        f"bp_extraction_discrepancy_status_check rejects"
    )
    assert resolution_action in LEGAL_RESOLUTION_ACTION, (
        f"{action} writes resolution_action={resolution_action!r}, which "
        f"bp_extraction_discrepancy_resolution_action_check rejects"
    )


@pytest.mark.parametrize("action", ["flag", "hold", "escalate", "assign",
                                    "investigate", "query"])
def test_escalating_actions_leave_the_finding_open(action):
    """A flag is a request for a human, not a resolution: the finding stays open.

    This is the outcome the person clicking "Flag for review" is trying to produce;
    closing it would remove the finding from the very queue it was raised into.
    """
    eng, cur = _engine()
    eng.execute(4242, action, user_id="tester", override_reason="under test")
    assert cur.updates[-1][0] == "open"


def test_reopening_a_finding_clears_the_resolution_stamp():
    """Re-opening must not leave the timestamp from a previous resolution behind.

    Before this fix `flag` never reached the table at all, so resolved -> open was
    unreachable. Now that it works, a finding that was resolved and is later flagged
    would otherwise keep a resolved_at saying it was closed — a row that reads as
    both open and resolved. The gateway's resolveDiscrepancy nulls it for the same
    reason; the two writers agree.
    """
    eng, cur = _engine()
    eng.execute(4242, "flag", user_id="tester", override_reason="under test")
    assert "resolved_at = NULL" in cur.update_sql[-1], (
        "re-opening a finding must clear resolved_at, or the row reads as both "
        f"open and resolved: {cur.update_sql[-1]}"
    )


@pytest.mark.parametrize("action,expected", [
    ("apply_value", "resolved"), ("confirm", "resolved"), ("approve", "resolved"),
    ("dismiss", "ignored"), ("reject", "ignored"),
])
def test_settling_actions_close_the_finding(action, expected):
    """apply_value/confirm/approve settle it; dismiss/reject set it aside."""
    eng, cur = _engine()
    eng.execute(4242, action, user_id="tester", override_reason="under test")
    assert cur.updates[-1][0] == expected


_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
@pytest.mark.integration
def test_every_action_survives_the_real_constraints():
    """The pairs the engine writes are accepted by the live table, not just by a set.

    The unit tests above assert against a copy of the constraint. A copy can drift
    into proving a value the database does not accept, which is exactly the failure
    being fixed here — so this writes each pair to a real row and rolls back.
    """
    import psycopg2  # noqa: F401  (imported for the exception type below)
    from src.services.db import get_conn

    from engines.decision_engine import DecisionEngine as DE

    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute(
            "SELECT discrepancy_id FROM proc.bp_extraction_discrepancy "
            "WHERE status = 'open' LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            pytest.skip("no open finding to probe")
        finding_id = row[0]
        rejected = []
        for action in sorted(DE.KNOWN_ACTIONS):
            status, resolution_action = DE.STATUS_FOR_ACTION[action]
            try:
                cur.execute(
                    "UPDATE proc.bp_extraction_discrepancy "
                    "SET status = %s, resolution_action = %s WHERE discrepancy_id = %s",
                    (status, resolution_action, finding_id),
                )
            except Exception as exc:  # CheckViolation
                rejected.append(f"{action} -> ({status}, {resolution_action}): {exc}")
            finally:
                conn.rollback()
        assert not rejected, "the live table rejected:\n" + "\n".join(rejected)


class _Refused(Exception):
    """Stands in for the psycopg2 error the lifecycle trigger raises (SQLSTATE BP409)."""

    pgcode = "BP409"

    def __init__(self, message):
        super().__init__(message)
        self.diag = SimpleNamespace(message_primary=message)


def test_a_refused_move_tells_the_person_why():
    """Two people act on one finding: the second click is refused by the database.

    The person must hear that the finding already moved, not the generic "could not
    update the finding" that reads like an outage and invites them to click again.
    """
    eng, cur = _engine()
    message = "finding 4242 is resolved; it cannot move to ignored"

    def _execute(sql, params=None, _orig=cur.execute):
        if " ".join(sql.split()).startswith("UPDATE proc.bp_extraction_discrepancy"):
            raise _Refused(message)
        return _orig(sql, params)

    cur.execute = _execute
    result = eng.execute(4242, "dismiss", user_id="second", override_reason="under test")

    assert result["applied"] is False
    assert result["error"] == message
    assert result.get("conflict") is True
