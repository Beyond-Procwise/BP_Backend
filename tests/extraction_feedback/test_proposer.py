"""T6: proposer aggregates recurring per-vendor failures into pending proposals."""
from src.services.db import get_conn
from src.services.extraction_feedback import proposer

V = "ZZTESTVENDOR"
V2 = "ZZTESTVENDOR2"


def _pending_for(vendor):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT field_name, status, proposed_hint FROM proc.bp_extraction_hint_proposal "
            "WHERE vendor_key=%s ORDER BY proposal_id",
            (vendor,),
        )
        return cur.fetchall()


def test_recurring_failure_creates_one_proposal_then_dedups(seed_telemetry):
    for i in range(4):  # 4 invoices, all with the same recurring discrepancy
        seed_telemetry("invoice", V, f"zz{i}", {"tax_percent_mismatch": 1})

    ids = proposer.propose_all(window_days=3650, min_docs=3, min_fail_rate=0.5,
                               draft=False, vendors={V})
    assert len(ids) == 1
    mine = _pending_for(V)
    assert len(mine) == 1
    assert mine[0][0] == "tax_percent_mismatch" and mine[0][1] == "pending"

    # Second run must NOT create a duplicate (dedup on scope+signal).
    ids2 = proposer.propose_all(window_days=3650, min_docs=3, min_fail_rate=0.5,
                                draft=False, vendors={V})
    assert ids2 == []
    assert len(_pending_for(V)) == 1


def test_below_min_docs_no_proposal(seed_telemetry):
    seed_telemetry("invoice", V2, "a", {"foo_mismatch": 1})
    seed_telemetry("invoice", V2, "b", {})  # clean doc → only 1/2 failing, < min_docs=3
    proposer.propose_all(window_days=3650, min_docs=3, min_fail_rate=0.5,
                         draft=False, vendors={V2})
    assert _pending_for(V2) == []
