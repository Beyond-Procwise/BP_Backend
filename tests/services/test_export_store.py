"""The exports table: save, get, list, delete, and the cap, with the statements seen."""
import datetime as dt

import pytest

from src.services import export_store as ES
from src.services.contract_tools import store


def _rows(found=None, outage=False):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if outage:
            raise store.StoreUnavailable("down")
        if sql.startswith("INSERT"):
            return [(params[0],)]
        if sql.startswith("DELETE"):
            return [(params[0],)] if found else []
        return list(found or [])
    rows.seen = seen
    return rows


def test_save_inserts_the_file_under_the_owner_and_returns_the_record_without_bytes():
    rows = _rows()
    rec = ES.save("xlsx", title="Spend by supplier", content=b"PK..", created_by="carol", source="skill:spend-analysis", rows=rows)
    sql, params = rows.seen[0]
    assert sql.startswith("INSERT INTO proc.bp_export") and params[1:5] == ["xlsx", "spend-by-supplier.xlsx", ES.MEDIA["xlsx"], "Spend by supplier"]
    assert params[5] == b"PK.." and params[6] == 4 and params[7:] == ["skill:spend-analysis", "carol"]
    assert rec["export_id"] == params[0] and len(rec["export_id"]) == 16 and "content" not in rec


def test_an_unknown_kind_and_a_file_over_the_cap_are_refused_before_any_statement():
    rows = _rows()
    with pytest.raises(ValueError):
        ES.save("pdf", title="t", content=b"x", created_by="c", rows=rows)
    with pytest.raises(ES.ExportTooLarge):
        ES.save("docx", title="t", content=b"x" * (ES.EXPORT_MAX_BYTES + 1), created_by="c", rows=rows)
    assert rows.seen == []


def test_get_returns_the_bytes_and_recent_does_not():
    row = ("abc", "docx", "note.docx", ES.MEDIA["docx"], "Note", memoryview(b"DOC"), 3, "agent", "carol", dt.datetime(2026, 10, 22, 18))
    got = ES.get("abc", rows=_rows(found=[row]))
    assert got["content"] == b"DOC" and got["created_date"] == "2026-10-22T18:00:00" and got["size_bytes"] == 3
    listed = ES.recent("carol", limit=500, rows=_rows(found=[row[:5] + (None,) + row[6:]]))
    assert "content" not in listed[0] and listed[0]["filename"] == "note.docx"
    assert ES.get("nope", rows=_rows(found=[])) is None


def test_delete_is_scoped_to_the_owner_and_an_outage_is_raised_not_hidden():
    rows = _rows(found=[("abc",)])
    assert ES.delete("abc", created_by="carol", rows=rows) is True
    assert rows.seen[0][1] == ["abc", "carol"]
    with pytest.raises(store.StoreUnavailable):
        ES.recent("carol", rows=_rows(outage=True))


def test_the_filename_is_a_safe_slug_of_the_title():
    assert ES._slug("Spend by supplier: Q3 (GBP)!") == "spend-by-supplier-q3-gbp"
    assert ES._slug("") == "export" and len(ES._slug("x" * 200)) == 60
