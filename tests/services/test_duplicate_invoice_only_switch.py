"""D-29 (B) option 1: write ONE duplicate-invoice case and nothing else.

The detector and the backfill script take two narrowing switches: ``only`` (which kind of finding:
the pair scan over promoted invoices, or the same-number conflict over raw documents) and ``refs``
(which invoice numbers). The dry-run listing and the write obey the same filters, so what the
operator sees listed is exactly what ``--apply`` writes. Offline: fake cursor, no database.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.services import duplicate_invoice_detector as did
from tests.services.test_duplicate_invoice_same_reference import LOWER, MASTER, _Conn, _Cur, _doc

OTHER_A = _doc(100, ref="INV-777", file="x1.pdf", supplier="Acme", total=100.0, text="x1")
OTHER_B = _doc(101, ref="INV-777", file="x2.pdf", supplier="Acme", total=100.0, text="x2")

PAIR = {  # what find_duplicates returns for two promoted invoices the engine judges a pair
    "earlier": {"invoice_id": "P-1", "supplier_name": "Acme", "total_amount": 50.0, "currency": "GBP"},
    "later": {"invoice_id": "P-2", "supplier_name": "Acme", "total_amount": 50.0, "currency": "GBP"},
    "amount": 50.0, "score": 95.0, "band": "certain", "link": {"signals": []},
}


@pytest.fixture()
def run(monkeypatch):
    upserts = []
    monkeypatch.setattr("src.services.opportunity_store.upsert_opportunity", lambda cur, rec: upserts.append(rec))
    monkeypatch.setattr(did, "_get_rates", lambda: {"USD": 1.0, "GBP": 1.0})
    monkeypatch.setattr(did, "find_duplicates", lambda invoices, **k: [PAIR])
    monkeypatch.setattr(did, "_opportunity_record", lambda dup, rates=None: {"opportunity_ref_id": "pair:P-2"})

    def go(docs, **kw):
        cur = _Cur(docs)
        written = did._run(_Conn(cur), **kw)
        return written, cur.inserts, upserts
    return go


def _keys(inserts):
    return sorted(i[2] for i in inserts)


def test_with_no_switch_both_kinds_are_written(run):
    written, inserts, upserts = run([MASTER, LOWER, OTHER_A, OTHER_B])
    assert _keys(inserts) == ["01-2024-002", "INV-777", "P-2"] and written == 3
    assert len(upserts) == 3


def test_only_same_reference_skips_the_pair_scan_entirely(run):
    written, inserts, upserts = run([MASTER, LOWER, OTHER_A, OTHER_B], only=did.SAME_REFERENCE)
    assert _keys(inserts) == ["01-2024-002", "INV-777"] and written == 2
    assert all(u["opportunity_ref_id"] != "pair:P-2" for u in upserts)


def test_only_pairs_skips_the_same_number_conflicts(run):
    written, inserts, upserts = run([MASTER, LOWER], only=did.PAIRS)
    assert _keys(inserts) == ["P-2"] and written == 1
    assert [u["opportunity_ref_id"] for u in upserts] == ["pair:P-2"]


def test_refs_narrows_to_the_named_invoice_numbers_and_nothing_else_is_touched(run):
    written, inserts, upserts = run([MASTER, LOWER, OTHER_A, OTHER_B],
                                    only=did.SAME_REFERENCE, refs={"01-2024-002"})
    assert _keys(inserts) == ["01-2024-002"] and written == 1
    assert [u["opportunity_ref_id"] for u in upserts] == ["same_reference_01-2024-002"]
    assert "15877.50" in inserts[0][4], "impact is the lower total"


def test_refs_match_ignores_case_and_spacing_like_the_detector_does(run):
    written, inserts, _ = run([MASTER, LOWER], only=did.SAME_REFERENCE, refs={" 01-2024-002 "})
    assert _keys(inserts) == ["01-2024-002"]


def test_refs_applies_to_the_pair_scan_by_the_flagged_invoice(run):
    written, inserts, _ = run([MASTER, LOWER], refs={"P-2"})
    assert _keys(inserts) == ["P-2"]
    written, inserts, _ = run([MASTER, LOWER], refs={"P-1"})       # the earlier one is not the flagged doc
    assert inserts == []


def test_an_unknown_kind_is_refused_before_anything_is_read(run):
    with pytest.raises(ValueError):
        run([MASTER, LOWER], only="everything")


# ---- the operator script passes the switches through unchanged ---------------------------------

def _backfill():
    path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_duplicate_invoices.py"
    spec = importlib.util.spec_from_file_location("backfill_duplicate_invoices", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _NoDB:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _Cur([MASTER, LOWER])


def test_backfill_apply_passes_only_and_refs_to_the_detector(monkeypatch, capsys):
    bf = _backfill()
    seen = {}
    monkeypatch.setattr(bf, "get_conn", lambda: _NoDB())
    monkeypatch.setattr(bf, "run_detector", lambda conn, **kw: seen.update(kw) or 1)
    assert bf.main(["--apply", "--only", "same-reference", "--ref", "01-2024-002"]) == 0
    assert seen == {"only": did.SAME_REFERENCE, "refs": {"01-2024-002"}}
    out = capsys.readouterr().out
    assert "01-2024-002" in out and "wrote 1 finding(s)" in out


def test_backfill_dry_run_lists_only_what_apply_would_write(monkeypatch, capsys):
    bf = _backfill()
    monkeypatch.setattr(bf, "get_conn", lambda: _NoDB())
    monkeypatch.setattr(bf, "run_detector", lambda conn, **kw: pytest.fail("dry run must not write"))
    assert bf.main(["--only", "same-reference", "--ref", "INV-NOT-THERE"]) == 0
    out = capsys.readouterr().out
    assert "0 same-number conflict(s)" in out and "01-2024-002" not in out
    assert "pair scan skipped" in out
    assert "dry run" in out
