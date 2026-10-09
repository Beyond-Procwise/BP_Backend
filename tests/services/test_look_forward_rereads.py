"""An upload stands for its NEWEST read, including a re-read filed under a corrected reference
(deal_assignment_service._raw_index).

Orbis's V3 order form (upload 12) was first read as "ORB-Q-6612" -- V1's reference -- and re-read
by script on 2026-09-23, with no upload id, as "ORB-Q-6612 (V3)". The look-forward matched the
upload by its own id only, so it found the first read: on an established deal it would have
linked V1 again and never V3. The raw rows below are the live ones.
"""
from src.services import deal_assignment_service as das

ORBIS = [  # raw_id, quote_id, source_file, process_monitor_id
    (38456, "ORB-Q-6612", "documents/quote/Orbis_Platform_Solutions_Ltd_V1.xlsx", 10),
    (38461, "ORB-Q-6612", "documents/quote/Orbis_Platform_Solutions_Ltd_V3_BAFO_WINNER.xlsx", 12),
    (38462, "ORB-Q-6612", "documents/quote/Orbis_Platform_Solutions_Ltd_V3_BAFO_WINNER.xlsx", 12),
    (38755, "ORB-Q-6612", "documents/quote/Orbis_Platform_Solutions_Ltd_V1.xlsx", None),
    (38757, "ORB-Q-6612 (V3)", "documents/quote/Orbis_Platform_Solutions_Ltd_V3_BAFO_WINNER.xlsx", None),
]


def _index(rows, monkeypatch):
    seen = {}

    def fake_rows(cur, sql, params=()):
        seen["sql"] = sql
        return [{"raw_id": r, "quote_id": q, "source_file": f, "process_monitor_id": p} for r, q, f, p in rows]
    monkeypatch.setattr(das, "_rows", fake_rows)
    out = das._raw_index(None, "proc.bp_quote_raw", "quote_id")
    return out, seen["sql"]


def test_an_upload_resolves_to_the_newer_reread_of_its_own_file(monkeypatch):
    out, _ = _index(ORBIS, monkeypatch)
    assert out["by_pmid"][12] == "ORB-Q-6612 (V3)"
    assert out["by_pmid"][10] == "ORB-Q-6612"


def test_a_read_from_another_upload_of_the_file_never_counts(monkeypatch):
    # Upload 99 re-sent V3's file under another deal; its read is its own, not upload 12's.
    rows = ORBIS[:3] + [(38900, "ORB-Q-6612 (V3)-OTHER", "documents/quote/Orbis_Platform_Solutions_Ltd_V3_BAFO_WINNER.xlsx", 99)]
    out, _ = _index(rows, monkeypatch)
    assert out["by_pmid"][12] == "ORB-Q-6612"
    assert out["by_pmid"][99] == "ORB-Q-6612 (V3)-OTHER"


def test_a_reread_older_than_the_uploads_own_read_does_not_displace_it(monkeypatch):
    rows = [(100, "OLD-REREAD", "documents/quote/x.xlsx", None), (200, "OWN", "documents/quote/x.xlsx", 5)]
    out, _ = _index(rows, monkeypatch)
    assert out["by_pmid"][5] == "OWN"


def test_newest_is_decided_by_read_order_not_by_the_order_rows_arrive_in(monkeypatch):
    out, _ = _index(list(reversed(ORBIS)), monkeypatch)
    assert out["by_pmid"][12] == "ORB-Q-6612 (V3)"
    assert out["by_pmid"][10] == "ORB-Q-6612"
