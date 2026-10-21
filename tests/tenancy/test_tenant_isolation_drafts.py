"""The tenant-isolation migration drafts (backlog 1.19, D-37): classification and generated files.

No database needed. The behaviour of the SQL is proven separately by scripts/tenancy/prove_on_scratch.sh
(a throwaway PostgreSQL 16 container). These tests stop the *inputs* drifting: a new table with no
isolation decision, a table in two groups, a hand-edited or stale generated file, or a migration that
touches a platform-shared table.
"""
from __future__ import annotations

import importlib.util
import os
import re

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.join(ROOT, rel)))
    spec.loader.exec_module(mod)
    return mod


cls = _load("tenancy_classification", "scripts/tenancy/classification.py")
gen = _load("tenancy_generate", "scripts/tenancy/generate_migrations.py")


def _lines(name):
    with open(os.path.join(HERE, name)) as fh:
        return [l.strip() for l in fh if l.strip()]


TABLES = _lines("proc_tables_2026-10-21.txt")
FILES = gen.files()


def test_every_table_has_exactly_one_decision():
    gen.check_classification(TABLES)   # raises with the names if not


def test_groups_do_not_overlap():
    assert not (cls.OWNED & cls.SHARED) and not (cls.OWNED & cls.REVIEW) and not (cls.SHARED & cls.REVIEW)


def test_a_new_table_without_a_decision_is_refused():
    with pytest.raises(SystemExit) as e:
        gen.check_classification(TABLES + ["bp_brand_new_table"])
    assert "bp_brand_new_table" in str(e.value)


def test_a_table_in_two_groups_is_refused(monkeypatch):
    monkeypatch.setattr(gen, "SHARED", frozenset(set(cls.SHARED) | {"bp_invoice_raw"}))
    with pytest.raises(SystemExit):
        gen.check_classification(TABLES)


def test_files_on_disk_are_what_the_generator_writes():
    for name, body in FILES.items():
        path = os.path.join(ROOT, "deploy", "sql", "tenancy", name)
        assert os.path.exists(path), f"{name} missing: run scripts/tenancy/generate_migrations.py"
        assert open(path).read() == body, f"{name} is stale or hand-edited: regenerate, do not edit"


def _tables_touched(sql):
    return set(re.findall(r"(?:ALTER TABLE|CALL pg_temp\.bp_relabel\('|ON) proc\.([a-z0-9_]+)", sql)) | \
        set(re.findall(r"CALL pg_temp\.bp_relabel\('proc\.([a-z0-9_]+)'", sql))


@pytest.mark.parametrize("part", ["1_add_label", "2_backfill_and_index", "3_not_null", "4_row_level_security"])
def test_migrations_touch_owned_tables_only(part):
    sql = FILES[f"2026-10-21_tenant_isolation_{part}.sql"]
    touched = _tables_touched(sql) - {"bp_support_access_log"}
    assert touched == set(cls.OWNED), (sorted(touched - cls.OWNED), sorted(cls.OWNED - touched))


def test_shared_and_review_tables_are_never_named_by_any_migration():
    protected = cls.SHARED | cls.REVIEW
    for name, sql in FILES.items():
        hit = {t for t in re.findall(r"proc\.([a-z0-9_]+)", sql) if t in protected}
        assert not hit, f"{name} touches {sorted(hit)}"


def test_every_owned_table_is_forced_not_just_enabled():
    sql = FILES["2026-10-21_tenant_isolation_4_row_level_security.sql"]
    assert sql.count("FORCE ROW LEVEL SECURITY") == len(cls.OWNED) == sql.count("ENABLE ROW LEVEL SECURITY")


def test_every_policy_fails_closed_when_no_tenant_is_set():
    sql = FILES["2026-10-21_tenant_isolation_4_row_level_security.sql"]
    # current_setting(..., true) returns NULL when unset; NULL = anything is not true -> no rows.
    assert sql.count("current_setting('app.tenant_id', true)") >= 2 * len(cls.OWNED)
    assert "app.tenant_id', false" not in sql    # the erroring form would break jobs instead of closing


def test_part_1_drops_the_default_it_adds():
    sql = FILES["2026-10-21_tenant_isolation_1_add_label.sql"]
    assert sql.count("ADD COLUMN IF NOT EXISTS tenant_id text DEFAULT 'default'") == len(cls.OWNED)
    assert sql.count("ALTER COLUMN tenant_id DROP DEFAULT") == len(cls.OWNED)


def test_identifiers_fit_and_are_unique():
    names = [gen.ident(p, t, s) for t in cls.OWNED for p, s in (("ix_bp", "tenant"), ("ck", "tenant_nn"))]
    assert all(len(n) <= 63 for n in names) and len(set(names)) == len(names)


def test_every_forward_file_has_a_rollback_where_one_can_exist():
    for part in ("1_add_label", "3_not_null", "4_row_level_security", "5_support_access"):
        assert f"2026-10-21_tenant_isolation_{part}_rollback.sql" in FILES


def test_every_file_says_it_is_a_draft():
    assert all(b.startswith("-- DRAFT FOR REVIEW. NOT APPLIED.") for b in FILES.values())
