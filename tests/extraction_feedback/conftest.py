"""Shared fixtures for extraction-feedback tests (live preprod DB, self-cleaning)."""
import json

import pytest

from src.services.db import get_conn


def _db_ok() -> bool:
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
            cur.fetchone()
        return True
    except Exception:
        return False


@pytest.fixture
def db_required():
    if not _db_ok():
        pytest.skip("no DB available")


@pytest.fixture
def seed_hint(db_required):
    """Insert an active extraction_vendor_hint bp_prompt row; clean up after."""
    created: list[int] = []

    def _seed(doc_type, vendor_key, hint_text, field_name=None, prompt_name=None):
        name = prompt_name or f"vhint::{doc_type}::{vendor_key}::{field_name or '_'}"
        desc = json.dumps({
            "scope": {"doc_type": doc_type, "vendor_key": vendor_key, "field_name": field_name},
            "hint_text": hint_text,
            "source_proposal_id": None,
        })
        with get_conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "INSERT INTO proc.bp_prompt "
                    "(prompt_name, prompt_type, prompts_desc, prompts_status, version, last_modified_by) "
                    "VALUES (%s, 'extraction_vendor_hint', %s::jsonb, 1, 1, 'test') "
                    "RETURNING prompt_id",
                    (name, desc),
                )
                pid = cur.fetchone()[0]
            c.commit()
        created.append(pid)
        return pid

    yield _seed

    with get_conn() as c:
        with c.cursor() as cur:
            for pid in created:
                cur.execute("DELETE FROM proc.bp_prompt WHERE prompt_id = %s", (pid,))
        c.commit()


@pytest.fixture
def cleanup_hint_names(db_required):
    """Delete any extraction_vendor_hint / proposal rows for given names/scopes after a test."""
    names: list[str] = []
    dedups: list[str] = []
    yield (names, dedups)
    with get_conn() as c:
        with c.cursor() as cur:
            for n in names:
                cur.execute(
                    "DELETE FROM proc.bp_prompt WHERE prompt_type='extraction_vendor_hint' AND prompt_name=%s",
                    (n,),
                )
            for d in dedups:
                cur.execute("DELETE FROM proc.bp_extraction_hint_proposal WHERE dedup_key=%s", (d,))
        c.commit()
