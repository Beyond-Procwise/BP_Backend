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
ALREADY = set(_lines("proc_tables_already_labelled_2026-10-21.txt"))
WITH_DEFAULT = _lines("proc_tables_label_default_2026-10-21.txt")
WAS_REQUIRED = _lines("proc_tables_label_notnull_2026-10-21.txt")
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


@pytest.mark.parametrize("part", ["1_add_label", "2_backfill_and_index", "3_not_null", "4_row_level_security", "6_stamp_on_insert"])
def test_migrations_touch_owned_tables_only(part):
    sql = FILES[f"2026-10-21_tenant_isolation_{part}.sql"]
    touched = _tables_touched(sql) - {"bp_support_access_log", "bp_stamp_tenant"}   # the latter is a function
    expected = set(cls.OWNED) - ALREADY if part == "1_add_label" else set(cls.OWNED)   # part 1 leaves old columns alone
    assert touched == expected, (sorted(touched - expected), sorted(expected - touched))


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


def test_part_1_keeps_the_default_so_nothing_written_before_part_6_is_unlabelled():
    """Dropping it in part 1 left NULL rows in the tamper-proof tables, which refuse the fix-up UPDATE
    (found by practising on a copy of the real database)."""
    sql = FILES["2026-10-21_tenant_isolation_1_add_label.sql"]
    added = len(cls.OWNED) - len(set(cls.OWNED) & ALREADY)
    assert sql.count("ADD COLUMN IF NOT EXISTS tenant_id text DEFAULT 'default'") == added
    assert "DROP DEFAULT" not in sql


def test_part_1_never_touches_a_column_that_already_existed():
    """Seven tables already had tenant_id NOT NULL DEFAULT 'default'; dropping that default in part 1
    would make their next insert fail before the stamping trigger exists (found 2026-10-21)."""
    sql = FILES["2026-10-21_tenant_isolation_1_add_label.sql"]
    for t in set(cls.OWNED) & ALREADY:
        assert f"ALTER TABLE proc.{t} " not in sql, t


def test_the_tables_with_an_old_default_are_owned_and_already_labelled():
    assert WITH_DEFAULT and set(WITH_DEFAULT) <= set(cls.OWNED) & ALREADY


def test_part_6_puts_a_stamp_trigger_on_every_owned_table():
    sql = FILES["2026-10-21_tenant_isolation_6_stamp_on_insert.sql"]
    assert sql.count("CREATE TRIGGER tr_bp_stamp_tenant BEFORE INSERT") == len(cls.OWNED)


def test_part_6_drops_each_default_in_the_same_transaction_as_its_trigger():
    sql = FILES["2026-10-21_tenant_isolation_6_stamp_on_insert.sql"]
    dropped = set()
    for b in sql.split("BEGIN;")[1:]:
        m = re.search(r"ALTER TABLE proc\.([a-z0-9_]+) ALTER COLUMN tenant_id DROP DEFAULT", b)
        assert m, "every block must drop its table's default"
        assert "CREATE TRIGGER tr_bp_stamp_tenant" in b and "COMMIT;" in b
        dropped.add(m.group(1))
    assert dropped == set(cls.OWNED)


def test_the_stamp_trigger_never_overwrites_and_never_guesses():
    sql = FILES["2026-10-21_tenant_isolation_6_stamp_on_insert.sql"]
    fn = sql[sql.index("CREATE OR REPLACE FUNCTION"):sql.index("$$;")]
    assert "IF NEW.tenant_id IS NULL THEN" in fn           # a named tenant is kept
    assert "coalesce" not in fn.lower() and "'default'" not in fn   # nothing bound stays NULL


def test_part_6_rollback_restores_the_state_after_part_1_exactly():
    sql = FILES["2026-10-21_tenant_isolation_6_stamp_on_insert_rollback.sql"]
    expected = {t for t in cls.OWNED if t in set(WITH_DEFAULT) or t not in ALREADY}
    restored = set(re.findall(r"ALTER TABLE proc\.([a-z0-9_]+) ALTER COLUMN tenant_id SET DEFAULT", sql))
    assert restored == expected


def test_identifiers_fit_and_are_unique():
    names = [gen.ident(p, t, s) for t in cls.OWNED for p, s in (("ix_bp", "tenant"), ("ck", "tenant_nn"))]
    assert all(len(n) <= 63 for n in names) and len(set(names)) == len(names)


def test_every_forward_file_has_a_rollback_where_one_can_exist():
    for part in ("1_add_label", "3_not_null", "4_row_level_security", "5_support_access", "6_stamp_on_insert"):
        assert f"2026-10-21_tenant_isolation_{part}_rollback.sql" in FILES


def test_every_file_says_it_is_a_draft():
    assert all(b.startswith("-- DRAFT FOR REVIEW. NOT APPLIED.") for b in FILES.values())


def test_rolling_back_part_3_keeps_the_columns_that_were_mandatory_before():
    """Found by a full forward-and-back cycle on a copy of the real database: the rollback stripped the
    original NOT NULL from ten tables."""
    sql = FILES["2026-10-21_tenant_isolation_3_not_null_rollback.sql"]
    dropped = set(re.findall(r"ALTER TABLE proc\.([a-z0-9_]+) ALTER COLUMN tenant_id DROP NOT NULL", sql))
    assert WAS_REQUIRED and set(WAS_REQUIRED) <= set(cls.OWNED)
    assert dropped == set(cls.OWNED) - set(WAS_REQUIRED)
