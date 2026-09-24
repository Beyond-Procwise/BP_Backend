"""Every translation that reaches a person, or is created, leaves an audit record.

The records go to proc.bp_agent_actions (phase='translation'), the same trail the rest of
the product reports from. These tests capture the writer instead of touching a database.
"""
from __future__ import annotations

import pytest

from src.services.i18n import audit


@pytest.fixture
def captured(monkeypatch):
    rows = []

    def fake(*, phase, action_type, **fields):
        rows.append({"phase": phase, "action_type": action_type, **fields})

    monkeypatch.setattr(audit, "_record_best_effort", fake)
    monkeypatch.setattr(audit, "_record_or_fail", fake)
    return rows


def test_served_records_who_what_and_exactly_what_was_shown(captured):
    audit.record_served(
        lang="de", requested_by="u1", model="m:1", prompt_version="v1",
        items=[{"source": "Hello", "shown": "Hallo", "status": "translated"},
               {"source": "Secret", "shown": "Secret", "status": "english_fallback"}],
    )
    (row,) = captured
    assert row["phase"] == "translation" and row["action_type"] == audit.SERVED
    assert row["agent"] == "translator" and "u1" in row["summary"]
    d = row["details"]
    assert d["requested_by"] == "u1" and d["lang"] == "de" and d["model"] == "m:1"
    assert d["items"][0]["shown"] == "Hallo" and d["items"][1]["status"] == "english_fallback"
    assert d["items"][0]["source_hash"]


def test_generated_is_best_effort_and_lists_hashes(captured):
    audit.record_generated(lang="ja", model="m:1", prompt_version="v1", hashes=["h1", "h2"])
    (row,) = captured
    assert row["action_type"] == audit.GENERATED and row["details"]["hashes"] == ["h1", "h2"]


def test_reviewed_import_keeps_old_and_new_text(captured):
    audit.record_reviewed_import(lang="es", imported_by="cli:muthu", added=3,
                                 changed=[("h", "Salvar", "Guardar")])
    (row,) = captured
    assert row["action_type"] == audit.REVIEWED_IMPORT
    assert row["details"]["changed"] == [{"source_hash": "h", "old": "Salvar", "new": "Guardar"}]


def test_served_audit_failure_raises(monkeypatch):
    def down(**_):
        raise audit.AuditWriteError("db down")

    monkeypatch.setattr(audit, "_record_or_fail", down)
    with pytest.raises(audit.AuditWriteError):
        audit.record_served(lang="de", requested_by="u1", model="m", prompt_version="v", items=[])


def test_public_key_list_changes_are_audited(captured):
    audit.record_public_keys(published_by="cli:muthu", total=2, added=["auth.signIn"], removed=["auth.old"])
    (row,) = captured
    assert row["action_type"] == audit.PUBLIC_KEYS
    assert row["details"] == {"published_by": "cli:muthu", "total": 2, "added": ["auth.signIn"], "removed": ["auth.old"]}
