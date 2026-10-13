"""Importing what is already in use must change nothing about what is in use."""
from __future__ import annotations

import json
import uuid

import pytest

from services.prompt_engine import legacy_import as L
from services.prompt_engine import registry as R


def _insert(db, name, desc, *, type_="message", status=1, version=1, by="system", linked=None):
    with db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, "
            "prompts_desc, prompts_status, version, created_by, last_modified_by) "
            "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)",
            (name, type_, linked, json.dumps(desc), status, version, by, by))


def _snapshot(db):
    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT prompt_id, to_jsonb(p)::text FROM proc.bp_prompt p ORDER BY prompt_id")
        return cur.fetchall()


@pytest.fixture()
def legacy(db):
    """A clean older table with one row of each shape the real one holds."""
    with db() as conn:
        conn.cursor().execute("TRUNCATE proc.bp_prompt")
    tag = uuid.uuid4().hex[:8]
    names = {k: f"{k}_{tag}" for k in ("object", "text", "list", "retired", "twin")}
    _insert(db, names["object"], {"prompt_template": "Rank {supplier}."}, version=5,
            by="response-style", linked="rag")
    _insert(db, names["text"], "Assess the document.", type_="agent_instructions", by="workspace")
    _insert(db, names["list"], {"templates": [{"id": "rank", "criteria": ["price"]}]})
    _insert(db, names["retired"], {"prompt_template": "old"}, status=0)
    _insert(db, names["twin"], {"prompt_template": "a"}, type_="email_prompt")
    _insert(db, names["twin"], {"prompt_template": "b"}, type_="message")
    _insert(db, "Has Spaces", {"prompt_template": "x"})
    return names


def test_each_active_prompt_becomes_version_one_in_use_exactly_as_it_was(db, legacy):
    report = L.run(apply=True)

    assert sorted(report["imported"]) == sorted(
        [legacy["object"], legacy["text"], legacy["list"]])
    live = R.resolve(legacy["object"])
    assert (live["version"], live["syntax"], live["template"], live["author"]) == (
        1, "legacy", "Rank {supplier}.", "response-style")
    assert live["settings"]["legacy_prompts_desc"] == {"prompt_template": "Rank {supplier}."}
    assert live["settings"]["legacy_linked_agents"] == "rag"
    assert R.resolve(legacy["text"])["template"] == "Assess the document."
    assert R.resolve(legacy["list"])["settings"]["legacy_prompts_desc"] == {
        "templates": [{"id": "rank", "criteria": ["price"]}]}
    assert R.pointer_history(legacy["object"])[0]["change"] == "imported"


def test_the_import_leaves_the_older_table_byte_for_byte_as_it_found_it(db, legacy):
    before = _snapshot(db)
    L.run(apply=True)
    assert _snapshot(db) == before


def test_a_retired_row_is_not_imported(db, legacy):
    L.run(apply=True)
    with pytest.raises(R.PointerUnresolved):
        R.resolve(legacy["retired"])


def test_a_shared_name_and_an_unusable_name_are_refused_and_said(db, legacy):
    report = L.run(apply=True)

    said = " | ".join(report["refused"])
    assert "Has Spaces" in said and "not a usable key" in said
    assert legacy["twin"] in said and "share this name" in said
    assert legacy["twin"] not in report["imported"]


def test_without_apply_it_only_says_what_it_would_do(db, legacy):
    report = L.run(apply=False)

    assert legacy["object"] in report["imported"] and report["applied"] is False
    with pytest.raises(R.PointerUnresolved):
        R.resolve(legacy["object"])


def test_running_it_again_imports_nothing_and_changes_nothing(db, legacy):
    L.run(apply=True)
    again = L.run(apply=True)

    assert again["imported"] == []
    assert sorted(again["already_there"]) == sorted(
        [legacy["object"], legacy["text"], legacy["list"]])
    assert [v["version"] for v in R.versions(legacy["object"])] == [1]


def test_an_older_row_edited_since_the_import_is_reported_not_overwritten(db, legacy):
    L.run(apply=True)
    with db() as conn:
        conn.cursor().execute(
            "UPDATE proc.bp_prompt SET prompts_desc = '{\"prompt_template\": \"edited\"}'::jsonb "
            "WHERE prompt_name = %s", (legacy["object"],))

    again = L.run(apply=True)

    assert again["drifted"] == [legacy["object"]]
    assert R.resolve(legacy["object"])["template"] == "Rank {supplier}."


def test_a_later_promotion_steps_the_older_tables_version_forward_never_back(db, legacy):
    """The imported row had reached version 5 in the older table. The registry
    calls the same wording version 1. Promoting registry version 2 must not
    tell the older readers the prompt went from 5 back to 2."""
    L.run(apply=True)
    R.save_version(legacy["object"], template="Rank {supplier} again.", author="ada",
                   syntax="legacy")
    R.promote(legacy["object"], 2, approved_by="bob")

    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT prompts_desc, version FROM proc.bp_prompt WHERE prompt_name = %s",
                    (legacy["object"],))
        assert cur.fetchone() == ({"prompt_template": "Rank {supplier} again."}, 6)
