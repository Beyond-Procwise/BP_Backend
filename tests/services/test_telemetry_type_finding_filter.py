"""proc.bp_extraction_telemetry must not count document-type findings.

Why this one was promoted out of the parked list while four other analytics
consumers stayed parked: _discrepancy_summary feeds n_discrepancies into a
PERSISTED per-document row, so an unfiltered count means a stored quality number
moves permanently because a reporting feature shipped. The other four
(corpus_facts, summary_agent, analysis_findings) compute on read, and each needs
its own product judgement about what its figure is for.

Two guards, deliberately. The live one is behavioural and is the real proof; the
non-live one runs in CI, where most runs happen. The earlier mistake in this plan
was having only the weak one, not having two.
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction import persistence  # noqa: E402
from src.services.extraction_telemetry import telemetry_service  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


def _without_sql_comments(text: str) -> str:
    """``text`` with SQL comments removed, lowercased — so a guard cannot pass
    on a clause that has been commented out."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return " ".join(line.split("--")[0] for line in text.splitlines()).lower()


def test_the_telemetry_query_excludes_every_type_finding_without_a_database():
    """NON-LIVE guard. The SQL the function sends must carry the exclusion, and
    the parameters must carry every type-finding issue type — which they do by
    importing TYPE_FINDING_ISSUE_TYPES rather than repeating the literals, so
    adding a fourth issue type cannot leave this consumer behind."""
    sent: dict = {}

    class Cur:
        def execute(self, sql, params=None):
            sent["sql"], sent["params"] = sql, params

        def fetchall(self):
            return []

    total, by_type = telemetry_service._discrepancy_summary(Cur(), "PK-1", "a.pdf")
    assert (total, by_type) == (0, {})
    flat = _without_sql_comments(sent["sql"])
    assert "<> all(" in flat, flat
    assert list(sent["params"][-1]) == list(persistence.TYPE_FINDING_ISSUE_TYPES)
    # the document's own identity is still matched on BOTH ways round, so the
    # added filter did not change which rows belong to the document
    assert "doc_pk_candidate = %s" in flat and "source_file = %s" in flat


@live
def test_the_telemetry_count_ignores_a_type_finding_but_keeps_a_real_one():
    """BEHAVIOURAL, against the real table. Two open findings on one probe
    document: one document-type finding, one genuine data problem. The telemetry
    count must see exactly one."""
    from src.services.db import get_conn

    pk = f"PROBE-TELEM-{uuid.uuid4().hex[:8]}"
    src_file = f"{pk}.pdf"
    try:
        persistence.write_discrepancies(
            doc_type="invoice", raw_id=1, source_file=src_file,
            doc_pk_candidate=pk,
            discrepancies=[
                persistence.Discrepancy(
                    field_name=persistence.TYPE_FINDING_FIELD,
                    issue_type=persistence.TYPE_FINDING_ISSUE_TYPES[0],
                    severity="warning", blocks_promotion=False,
                    raw_value="doctype.invoice", computed_value="doctype.order"),
                persistence.Discrepancy(
                    field_name="total_amount", issue_type="missing_required",
                    severity="critical", blocks_promotion=True),
            ])
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "select issue_type, count(*) from proc.bp_extraction_discrepancy "
                "where doc_pk_candidate = %s group by issue_type", (pk,))
            on_the_table = dict(cur.fetchall())
            total, by_type = telemetry_service._discrepancy_summary(cur, pk, src_file)
        assert len(on_the_table) == 2, on_the_table   # both rows really landed
        assert by_type == {"missing_required": 1}, by_type
        assert total == 1, total
    finally:
        with get_conn() as conn:
            conn.cursor().execute(
                "delete from proc.bp_extraction_discrepancy "
                "where doc_pk_candidate = %s", (pk,))
