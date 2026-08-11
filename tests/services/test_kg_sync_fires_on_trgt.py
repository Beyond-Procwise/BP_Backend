"""The KG sync fires when a row reaches _trgt, not when it reaches _stg.

The graph mirrors _trgt — the final, accepted state a document is approved and
transacted against. The sync used to run from process_monitor_watcher the moment
dispatch returned "promoted", which is _stg promotion. That put nodes in the
graph for documents still in flight, and the reconciling rebuild then swept them
as absent from _trgt, so the graph oscillated for exactly the unsettled
documents. Observed live on invoice 0526: in bp_invoice_stg, not in
bp_invoice_trgt, synced by the old path and removed by the next rebuild.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"


def test_kg_sync_reads_the_trgt_tier():
    from src.services.extraction.kg_sync import _TRGT_TABLE

    assert _TRGT_TABLE, "the table map is empty"
    for doc_type, table in _TRGT_TABLE.items():
        assert table.endswith("_trgt"), (
            f"{doc_type} reads {table}; the graph mirrors _trgt, so syncing "
            f"from _stg creates nodes the next rebuild deletes"
        )


def test_the_watcher_no_longer_triggers_the_sync():
    """process_monitor_watcher's 'promoted' means _stg. A call reinstated there
    would resurrect the oscillation."""
    source = (_SRC / "services" / "process_monitor_watcher.py").read_text()
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "sync_row_to_kg" not in called, (
        "process_monitor_watcher calls sync_row_to_kg again. At that point the "
        "row is in _stg, not _trgt."
    )


def test_the_trgt_promotion_job_triggers_the_sync():
    source = (_SRC / "services" / "backend_scheduler.py").read_text()
    assert "_sync_promoted_to_kg" in source
    assert "sync_row_to_kg" in source


class _Nick:
    pass


def _scheduler():
    from src.services.backend_scheduler import BackendScheduler
    s = BackendScheduler.__new__(BackendScheduler)
    s.agent_nick = _Nick()
    return s


def test_only_promoted_details_are_synced(monkeypatch):
    """A held document has not reached final state and must not be synced."""
    seen = []
    import src.services.extraction.kg_sync as ks
    monkeypatch.setattr(ks, "sync_row_to_kg",
                        lambda nick, dt, pk: seen.append((dt, pk)) or 1)

    result = {"details": [
        {"doc_type": "invoice", "doc_pk": "INV-1", "action": "promoted"},
        {"doc_type": "quote", "doc_pk": "Q-1", "action": "held"},
        {"doc_type": "purchase_order", "promoted": 3, "held": 0},   # aggregate
        {"doc_type": "invoice", "doc_pk": "INV-2", "action": "promoted"},
    ]}
    assert _scheduler()._sync_promoted_to_kg(result) == 2
    assert seen == [("invoice", "INV-1"), ("invoice", "INV-2")]


def test_a_sync_failure_never_breaks_the_promotion_job(monkeypatch):
    """The row is durable in _trgt before this runs. The graph is a downstream
    view and can be rebuilt; a KG problem must not fail promotion."""
    import src.services.extraction.kg_sync as ks

    def _boom(*a, **kw):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(ks, "sync_row_to_kg", _boom)
    result = {"details": [
        {"doc_type": "invoice", "doc_pk": "INV-1", "action": "promoted"},
    ]}
    assert _scheduler()._sync_promoted_to_kg(result) == 0   # no raise


@pytest.mark.parametrize("result", [None, {}, {"details": None}, {"details": []}])
def test_an_empty_result_is_harmless(result):
    assert _scheduler()._sync_promoted_to_kg(result) == 0
