"""T5: approve → versioned active hint; supersede prior; reject."""
import json

from src.services.db import get_conn
from src.services.extraction_feedback import apply
from src.services.extraction_feedback.hint_store import HINT_STORE


def _insert_proposal(doc_type, vendor_key, field_name, hint, dedup):
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_extraction_hint_proposal "
                "(doc_type, vendor_key, field_name, dedup_key, evidence, proposed_hint, status) "
                "VALUES (%s,%s,%s,%s,%s::jsonb,%s,'pending') RETURNING proposal_id",
                (doc_type, vendor_key, field_name, dedup, json.dumps({"sample_count": 4}), hint),
            )
            pid = cur.fetchone()[0]
        c.commit()
    return pid


def test_approve_creates_active_versioned_hint(cleanup_hint_names):
    names, dedups = cleanup_hint_names
    names.append("vhint::invoice::TESTCO::tax_amount")
    dedups.append("t-approve-1")
    pid = _insert_proposal("invoice", "TESTCO", "tax_amount", "capture VAT as tax_amount", "t-approve-1")

    res = apply.approve(pid, "tester")
    assert res["prompt_id"] and res["version"] == 1

    HINT_STORE.refresh()
    assert any("capture VAT" in h for h in HINT_STORE.hints_for("invoice", "TESTCO"))
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT status, resulting_prompt_id FROM proc.bp_extraction_hint_proposal WHERE proposal_id=%s", (pid,))
        status, rpid = cur.fetchone()
    assert status == "approved" and rpid == res["prompt_id"]


def test_approve_supersedes_prior_active(cleanup_hint_names):
    names, dedups = cleanup_hint_names
    names.append("vhint::invoice::TESTCO2::_")
    dedups.extend(["t-sup-1", "t-sup-2"])

    p1 = _insert_proposal("invoice", "TESTCO2", None, "hint one", "t-sup-1")
    apply.approve(p1, "tester")
    p2 = _insert_proposal("invoice", "TESTCO2", None, "hint two", "t-sup-2")
    r2 = apply.approve(p2, "tester")

    assert r2["version"] == 2
    HINT_STORE.refresh()
    assert HINT_STORE.hints_for("invoice", "TESTCO2") == ["hint two"]  # only latest active
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM proc.bp_prompt WHERE prompt_name='vhint::invoice::TESTCO2::_' AND prompts_status=1"
        )
        assert cur.fetchone()[0] == 1  # exactly one active


def test_reject_records_reason(cleanup_hint_names):
    names, dedups = cleanup_hint_names
    dedups.append("t-rej-1")
    pid = _insert_proposal("quote", "REJCO", "po_id", "x", "t-rej-1")
    apply.reject(pid, "tester", "not useful")
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT status, review_reason FROM proc.bp_extraction_hint_proposal WHERE proposal_id=%s", (pid,))
        status, reason = cur.fetchone()
    assert status == "rejected" and reason == "not useful"
