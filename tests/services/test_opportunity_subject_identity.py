"""An opportunity is what it is ABOUT, not the documents that supported it that day.

proc.bp_opportunity was keyed on opportunity_ref_id, which carries a fingerprint of
the supporting documents. One more invoice against the same order changed the
fingerprint, so:

  * an opportunity a person had REJECTED came back as a brand-new row at
    'identified', beside the rejected one, with nothing linking the two;
  * nothing could tell "the same thing again" from "the same thing, with news";
  * a rejected opportunity could never be re-opened at all.

Separately, the Reject button stores its feedback against the row's id, and the
miner looked feedback up by its own per-run counter (1, 2, 3...) -- so a rejection
could land on whichever unrelated finding was numbered the same in a later run.

The live tests insert their own probe rows and roll back. Nothing real is touched.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.services import opportunity_store as store

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


# ---------------------------------------------------------------------------
# The identity itself
# ---------------------------------------------------------------------------

def test_the_subject_does_not_depend_on_the_supporting_documents():
    base = {"detector_type": "Price Benchmark Variance", "supplier_id": "SUP-1",
            "item_id": "ITM-1"}
    then = store.subject_key({**base, "opportunity_ref_id": "p_d_aaaaaaaa_s_i",
                              "source_records": ["PO-1", "INV-1"]})
    now = store.subject_key({**base, "opportunity_ref_id": "p_d_bbbbbbbb_s_i",
                             "source_records": ["PO-1", "INV-1", "INV-2"]})
    assert then == now == "price benchmark variance|sup-1|itm-1"


def test_a_different_lever_on_the_same_item_is_a_different_subject():
    a = store.subject_key({"detector_type": "Price Benchmark Variance",
                           "supplier_id": "S", "item_id": "I"})
    b = store.subject_key({"detector_type": "Volume Consolidation",
                           "supplier_id": "S", "item_id": "I"})
    assert a != b


def test_with_no_item_the_deal_tells_two_findings_on_one_supplier_apart():
    """Overbilling is per deal, and one supplier has many deals."""
    a = store.subject_key({"detector_type": "Invoice Overbilling", "supplier_id": "S",
                           "calculation_details": {"deal_id": "DEAL-1"}})
    b = store.subject_key({"detector_type": "Invoice Overbilling", "supplier_id": "S",
                           "deal_id": "DEAL-2"})
    assert a == "invoice overbilling|s|deal:deal-1"
    assert b == "invoice overbilling|s|deal:deal-2"


def test_an_item_finding_ignores_the_deal_so_a_later_deal_link_cannot_split_it():
    """opportunity_linkage.py fills deal_id on the ROW after the fact; the finding
    itself still carries none. The identity must not move when that happens."""
    bare = store.subject_key({"detector_type": "D", "supplier_id": "S", "item_id": "I"})
    linked = store.subject_key({"detector_type": "D", "supplier_id": "S", "item_id": "I",
                                "deal_id": "DEAL-9"})
    assert bare == linked


def test_a_finding_about_nothing_in_particular_has_no_subject():
    assert store.subject_key({"detector_type": "D", "opportunity_ref_id": "r"}) is None


# ---------------------------------------------------------------------------
# What counts as news
# ---------------------------------------------------------------------------

def test_the_same_evidence_is_not_news():
    assert store.evidence_change(["PO-1", "INV-1"], 100.0, ["INV-1", "PO-1"], 100.004) is None


def test_a_new_record_is_news_and_is_named():
    change = store.evidence_change(["PO-1", "INV-1"], 100.0, ["PO-1", "INV-1", "INV-2"], 150.0)
    assert change == {"new_records": ["INV-2"], "impact_before": 100.0, "impact_after": 150.0}


def test_a_changed_figure_alone_is_news():
    """A new benchmark moves the saving without adding a single document."""
    change = store.evidence_change(["PO-1"], 100.0, ["PO-1"], 240.0)
    assert change["new_records"] == [] and change["impact_after"] == 240.0


def test_a_record_dropping_out_at_the_same_figure_is_not_news():
    assert store.evidence_change(["PO-1", "INV-1"], 100.0, ["PO-1"], 100.0) is None


def test_a_second_change_before_anyone_looked_adds_to_the_first():
    """Measured from the figure the person rejected, with everything that arrived."""
    first = store.evidence_change(["PO-1"], 100.0, ["PO-1", "INV-2"], 150.0)
    second = store.evidence_change(["PO-1", "INV-2"], 150.0, ["PO-1", "INV-2", "INV-3"],
                                   190.0, pending=first)
    assert second == {"new_records": ["INV-2", "INV-3"], "impact_before": 100.0,
                      "impact_after": 190.0}


# ---------------------------------------------------------------------------
# A rejection lands on the opportunity that was rejected, and no other
# ---------------------------------------------------------------------------

def test_feedback_is_not_matched_on_the_run_counter(monkeypatch):
    """Row '1' was rejected. The first finding of a later run is also numbered 1."""
    import agents.opportunity_miner_agent as miner_module

    feedback = {"opportunity_id": "1", "opportunity_ref_id": "1", "status": "rejected",
                "reason": "not relevant", "user_id": "u", "metadata": None,
                "updated_on": None}
    monkeypatch.setattr(miner_module, "load_opportunity_feedback",
                        lambda _nick: {"1": feedback})
    agent = object.__new__(miner_module.OpportunityMinerAgent)
    agent.agent_nick = None
    unrelated = miner_module.Finding(
        opportunity_id="1", opportunity_ref_id="pol_det_abc12345_sup_item",
        detector_type="D", supplier_id="SUP-X", category_id=None, item_id="ITM-X",
        financial_impact_gbp=10.0, calculation_details={}, source_records=[],
        detected_on=None)

    agent._apply_feedback_annotations([unrelated])

    assert unrelated.is_rejected is False
    assert unrelated.feedback_reason is None


def test_feedback_recorded_against_the_reference_still_applies(monkeypatch):
    import agents.opportunity_miner_agent as miner_module

    ref = "pol_det_abc12345_sup_item"
    feedback = {"opportunity_id": "row-7", "opportunity_ref_id": ref,
                "status": "rejected", "reason": "already contracted", "user_id": "u",
                "metadata": None, "updated_on": None}
    monkeypatch.setattr(miner_module, "load_opportunity_feedback",
                        lambda _nick: {"row-7": feedback, ref: feedback})
    agent = object.__new__(miner_module.OpportunityMinerAgent)
    agent.agent_nick = None
    finding = miner_module.Finding(
        opportunity_id="1", opportunity_ref_id=ref, detector_type="D",
        supplier_id="SUP-X", category_id=None, item_id="ITM-X",
        financial_impact_gbp=10.0, calculation_details={}, source_records=[],
        detected_on=None)

    agent._apply_feedback_annotations([finding])

    assert finding.is_rejected is True
    assert finding.feedback_reason == "already contracted"


# ---------------------------------------------------------------------------
# Live: what the table does
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn():
    """get_conn() is AUTOCOMMIT, on which rollback is a no-op -- switched off here, or
    every probe below would be a real write."""
    from src.services.db import get_conn

    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _finding(tag, sources, impact=100.0, counter="90001", **extra):
    """A price finding as the miner emits it: the reference changes with the sources."""
    import hashlib
    token = hashlib.sha1("|".join(sorted(sources)).encode()).hexdigest()[:8]
    return {"opportunity_id": counter,
            "opportunity_ref_id": f"probe_price_{token}_{tag}",
            "detector_type": "Subject Identity Probe",
            "supplier_id": f"SUP-{tag}", "item_id": f"ITM-{tag}",
            "financial_impact_gbp": impact, "source_records": list(sources),
            "calculation_details": {"actual_price": 12.0, "quantity": 40}, **extra}


def _rows(cur, tag):
    cur.execute("select opportunity_id, stage, review_requested_at is not null, "
                "review_detail, source_records, financial_impact_gbp "
                "from proc.bp_opportunity where supplier_id=%s order by created_at, "
                "opportunity_id", (f"SUP-{tag}",))
    # Parsed here, not left to the driver: importing scripts/testdata/db.py anywhere in
    # the same run tells psycopg2 to hand JSONB back as text, process-wide.
    return [(oid, stage, asked, store._as_dict(detail) or None, store._as_list(sources), impact)
            for oid, stage, asked, detail, sources, impact in cur.fetchall()]


@pytest.fixture()
def tag():
    return f"probe{uuid.uuid4().hex[:10]}"


@live
def test_rejected_and_found_again_on_the_same_evidence_stays_one_rejected_row(conn, tag):
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-1"]))
    store.set_stage(oid, "rejected", conn=conn)

    again = store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-1"], counter="5"))

    rows = _rows(cur, tag)
    assert again == oid and len(rows) == 1
    assert rows[0][1] == "rejected" and rows[0][2] is False


@live
def test_new_evidence_on_a_rejection_asks_for_review_instead_of_raising_it_again(conn, tag):
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-1"], impact=100.0))
    store.set_stage(oid, "rejected", conn=conn)

    again = store.upsert_opportunity(
        cur, _finding(tag, ["PO-1", "INV-1", "INV-2"], impact=150.0, counter="7"))

    rows = _rows(cur, tag)
    assert again == oid
    assert len(rows) == 1, "the same opportunity must not come back as a second row"
    _id, stage, asked, detail, sources, impact = rows[0]
    assert stage == "rejected", "new evidence asks a person; it does not overrule them"
    assert asked is True
    assert detail == {"new_records": ["INV-2"], "impact_before": 100.0, "impact_after": 150.0}
    assert sorted(sources) == ["INV-1", "INV-2", "PO-1"] and float(impact) == 150.0


@live
def test_a_person_can_reopen_after_new_evidence(conn, tag):
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1"]))
    store.set_stage(oid, "rejected", conn=conn)
    store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-2"], impact=150.0))

    assert store.resolve_review(oid, "reopen", decided_by="sub-1", conn=conn) == "identified"

    _id, stage, asked, detail, _s, _i = _rows(cur, tag)[0]
    assert stage == "identified" and asked is False
    assert detail["decision"] == "reopen" and detail["decided_by"] == "sub-1"


@live
def test_keeping_a_rejection_settles_it_until_something_else_changes(conn, tag):
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1"]))
    store.set_stage(oid, "rejected", conn=conn)
    store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-2"], impact=150.0))

    assert store.resolve_review(oid, "keep_rejected", conn=conn) == "rejected"
    store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-2"], impact=150.0))
    assert _rows(cur, tag)[0][1:3] == ("rejected", False), "the same evidence must not nag"

    store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-2", "INV-3"], impact=190.0))
    _id, stage, asked, detail, _s, _i = _rows(cur, tag)[0]
    assert stage == "rejected" and asked is True
    assert detail["new_records"] == ["INV-3"], "measured from what the person last saw"
    assert detail["impact_before"] == 150.0


@live
def test_there_is_nothing_to_review_on_an_opportunity_that_is_not_rejected(conn, tag):
    from src.services.lifecycle import IllegalTransition

    oid = store.upsert_opportunity(conn.cursor(), _finding(tag, ["PO-1"]))
    with pytest.raises(IllegalTransition):
        store.resolve_review(oid, "reopen", conn=conn)


@live
def test_new_evidence_on_an_open_opportunity_just_refreshes_it(conn, tag):
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1"], impact=100.0))
    store.set_stage(oid, "negotiation", conn=conn)

    store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-2"], impact=150.0))

    rows = _rows(cur, tag)
    assert len(rows) == 1
    assert rows[0][1] == "negotiation" and rows[0][2] is False
    assert float(rows[0][5]) == 150.0


@live
def test_new_records_after_a_realised_saving_are_a_new_opportunity(conn, tag):
    """The finished one keeps its history; what happened since is raised afresh."""
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1"], impact=100.0))
    store.set_stage(oid, "realised", conn=conn)

    fresh = store.upsert_opportunity(cur, _finding(tag, ["PO-1", "PO-2"], impact=60.0))

    stages = {r[0]: r[1] for r in _rows(cur, tag)}
    assert fresh != oid
    assert stages == {oid: "realised", fresh: "identified"}


@live
def test_a_new_finding_numbered_like_an_existing_row_is_saved(conn, tag):
    """Row '1' exists in this corpus; the miner numbers every run from 1."""
    cur = conn.cursor()
    cur.execute("insert into proc.bp_opportunity (opportunity_id, opportunity_ref_id, stage) "
                "values (%s, %s, 'identified') on conflict do nothing",
                ("1", f"counter-probe-{tag}"))

    stored = store.upsert_opportunity(cur, _finding(tag, ["PO-1"], counter="1"))

    assert stored != "1"
    assert len(_rows(cur, tag)) == 1


@live
def test_a_row_refreshed_under_new_evidence_is_not_then_retired(conn, tag):
    """The row keeps the reference it was first seen under, so retirement goes by row."""
    cur = conn.cursor()
    oid = store.upsert_opportunity(cur, _finding(tag, ["PO-1"]))
    seen = store.upsert_opportunity(cur, _finding(tag, ["PO-1", "INV-2"], impact=150.0))

    store.retire_missing(cur, [seen], ["Subject Identity Probe"])

    assert seen == oid
    assert _rows(cur, tag)[0][1] == "identified"


@live
def test_the_migration_and_the_code_agree_on_the_identity(conn):
    """deploy/sql/2026-10-13_opportunity_subject_review.sql backfills the column with
    a SQL copy of subject_key(). If the two drift, every backfilled row is orphaned
    and its opportunity comes back as new."""
    cur = conn.cursor()
    cur.execute("select detector_type, supplier_id, item_id, deal_id, category_id, "
                "subject_key from proc.bp_opportunity where subject_key is not null")
    rows = cur.fetchall()
    assert rows, "no backfilled rows to compare -- has the migration been applied?"
    for detector, supplier, item, deal, category, stored in rows:
        assert store.subject_key({
            "detector_type": detector, "supplier_id": supplier, "item_id": item,
            "deal_id": deal, "category_id": category}) == stored
