"""The exec summary's "Saved (GBP)" fact and the ledger-sourced "Realised savings (GBP)".

Source-level pin, following the pattern of ``test_opportunity_dashboard.py``'s
``test_realised_comes_from_the_ledger``: this proves the builder was wired to the
ledger, not that a query returns the right number against live data. The live proof
is ``tests/services/rga/test_exec_summary_live.py`` plus the Task 10 demo -- the
builder takes no injected connection (``_fetch`` always opens its own via
``get_conn()``), so a rolled-back-transaction test cannot be added here without
adding a connection seam solely for the test, which the brief for this task rules
out.
"""

from datetime import date


def test_saved_fact_reads_the_ledger(monkeypatch):
    from src.services.rga.builders import exec_procurement_summary as ex

    src = open(ex.__file__).read()
    assert "proc.bp_value_outcome" in src
    assert "exec_summary.value_saved" in src
