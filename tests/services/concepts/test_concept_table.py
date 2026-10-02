"""The concept tables and seed.py must agree, and unconfirmed concepts must not resolve.

The vocabulary exists twice — as the seed in src/services/concepts/seed.py and as
rows in proc.bp_concept / proc.bp_document_type. Duplication between code and data
is only safe when something fails loudly the moment the two diverge. This is the
same contract tests/services/facts/test_uom_canonical_table.py holds for units.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/concepts/test_concept_table.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.seed import CONCEPTS, DOCUMENT_TYPES  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_DOMAINS = {
    "DOCUMENT_TYPE", "RELATIONSHIP_ROLE", "LINK_TYPE",
    "EXECUTION_MODE", "EVENT_KIND",
}


@pytest.fixture()
def concept_rows():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT concept_code, domain, definition, not_to_be_confused_with,
                   status, source, rejection_reason
              FROM proc.bp_concept
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@pytest.fixture()
def doc_type_rows():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT concept_code, role, default_parent_type, execution_mode,
                   aliases, identifiers, structural_signals, pipeline_doc_type,
                   status, source, requires_parent_evidence, parent_evidence_phrases
              FROM proc.bp_document_type
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def test_every_seeded_concept_exists_in_the_table(concept_rows):
    in_table = {r["concept_code"] for r in concept_rows}
    missing = set(CONCEPTS) - in_table
    assert not missing, f"concepts in seed.py with no row in bp_concept: {sorted(missing)}"


def test_every_table_concept_exists_in_the_seed(concept_rows):
    """The other direction. Without this, a concept could be added to the
    database and silently never resolve, which looks like the table works."""
    in_table = {r["concept_code"] for r in concept_rows}
    extra = in_table - set(CONCEPTS)
    assert not extra, f"concepts in bp_concept absent from seed.py: {sorted(extra)}"


def test_every_domain_is_one_of_the_five(concept_rows):
    bad = {r["concept_code"]: r["domain"] for r in concept_rows if r["domain"] not in _DOMAINS}
    assert not bad, f"rows with an unrecognised domain: {bad}"


def test_every_concept_code_carries_its_domain_prefix(concept_rows):
    prefix = {
        "DOCUMENT_TYPE": "doctype.",
        "RELATIONSHIP_ROLE": "role.",
        "LINK_TYPE": "link.",
        "EXECUTION_MODE": "exec.",
        "EVENT_KIND": "event.",
    }
    wrong = [
        r["concept_code"] for r in concept_rows
        if not r["concept_code"].startswith(prefix[r["domain"]])
    ]
    assert not wrong, (
        f"concept codes whose prefix does not match their domain: {sorted(wrong)} "
        "— the prefix is what keeps role.variation and doctype.variation on "
        "separate primary keys"
    )


def test_every_active_concept_has_a_definition(concept_rows):
    missing = [
        r["concept_code"] for r in concept_rows
        if r["status"] == "active" and not (r["definition"] or "").strip()
    ]
    assert not missing, f"active concepts with no definition: {sorted(missing)}"


def test_not_to_be_confused_with_points_at_real_concepts(concept_rows):
    known = {r["concept_code"] for r in concept_rows}
    dangling = {}
    for r in concept_rows:
        for other in r["not_to_be_confused_with"] or []:
            if other not in known:
                dangling.setdefault(r["concept_code"], []).append(other)
    assert not dangling, f"not_to_be_confused_with entries that name no concept: {dangling}"


def test_nothing_points_at_itself(concept_rows):
    selfref = [
        r["concept_code"] for r in concept_rows
        if r["concept_code"] in (r["not_to_be_confused_with"] or [])
    ]
    assert not selfref, f"concepts listed as not to be confused with themselves: {selfref}"


def test_every_rejected_concept_records_why(concept_rows):
    """'Fix the uploader' and 'this genuinely is not a document type' are
    different problems. Without the reason they look identical."""
    unexplained = [
        r["concept_code"] for r in concept_rows
        if r["status"] == "rejected" and not (r["rejection_reason"] or "").strip()
    ]
    assert not unexplained, f"rejected with no reason recorded: {sorted(unexplained)}"


def test_every_document_type_row_has_a_concept(concept_rows, doc_type_rows):
    doctypes = {r["concept_code"] for r in concept_rows if r["domain"] == "DOCUMENT_TYPE"}
    orphans = {r["concept_code"] for r in doc_type_rows} - doctypes
    assert not orphans, f"bp_document_type rows with no DOCUMENT_TYPE concept: {sorted(orphans)}"


def test_every_document_type_concept_has_attributes(concept_rows, doc_type_rows):
    doctypes = {
        r["concept_code"] for r in concept_rows
        if r["domain"] == "DOCUMENT_TYPE" and r["status"] != "rejected"
    }
    missing = doctypes - {r["concept_code"] for r in doc_type_rows}
    assert not missing, (
        f"DOCUMENT_TYPE concepts with no bp_document_type row: {sorted(missing)} "
        "— a type with no role or parent cannot take part in a relationship"
    )


def test_roles_parents_and_modes_resolve_to_concepts(concept_rows, doc_type_rows):
    by_domain = {}
    for r in concept_rows:
        by_domain.setdefault(r["domain"], set()).add(r["concept_code"])
    for r in doc_type_rows:
        assert r["role"] in by_domain.get("RELATIONSHIP_ROLE", set()), (
            f"{r['concept_code']}: role {r['role']!r} is not a RELATIONSHIP_ROLE concept"
        )
        if r["default_parent_type"]:
            assert r["default_parent_type"] in by_domain.get("DOCUMENT_TYPE", set()), (
                f"{r['concept_code']}: default_parent_type {r['default_parent_type']!r} "
                "is not a DOCUMENT_TYPE concept"
            )
        if r["execution_mode"]:
            assert r["execution_mode"] in by_domain.get("EXECUTION_MODE", set()), (
                f"{r['concept_code']}: execution_mode {r['execution_mode']!r} "
                "is not an EXECUTION_MODE concept"
            )


def test_seeded_aliases_are_recorded_in_the_table(doc_type_rows):
    table_aliases = {a.lower() for r in doc_type_rows for a in (r["aliases"] or [])}
    seeded = {a.lower() for dt in DOCUMENT_TYPES.values() for a in dt.aliases}
    missing = seeded - table_aliases
    assert not missing, f"aliases in seed.py absent from the table: {sorted(missing)}"


def test_pipeline_doc_type_is_one_the_pipeline_actually_has(doc_type_rows):
    """Four physical table families exist. A fifth value here would route a
    document at a table that is not there."""
    allowed = {"invoice", "purchase_order", "quote", "contract", None}
    bad = {
        r["concept_code"]: r["pipeline_doc_type"] for r in doc_type_rows
        if r["pipeline_doc_type"] not in allowed
    }
    assert not bad, f"pipeline_doc_type values with no physical pipeline: {bad}"


def test_no_alias_is_claimed_by_two_concepts(doc_type_rows):
    """The collision check. One alias on two rows is how a genuine ambiguity
    surfaces, and until the conflict register exists it must not be seeded —
    a resolver facing it can only answer UNRESOLVED."""
    owners: dict[str, list[str]] = {}
    for r in doc_type_rows:
        for alias in r["aliases"] or []:
            owners.setdefault(alias.strip().lower(), []).append(r["concept_code"])
    clashes = {a: sorted(o) for a, o in owners.items() if len(o) > 1}
    assert not clashes, f"aliases claimed by more than one concept: {clashes}"


def test_an_alias_never_equals_another_concepts_code(concept_rows, doc_type_rows):
    """'contract' as an alias of doctype.master_agreement while
    doctype.contract exists would make the resolver's answer depend on which
    table it looked at first.

    Scoped to DOCUMENT_TYPE codes, and the narrowing is the point. The alias
    index is built from bp_document_type rows ALONE (vocabulary.py's doc-type
    loop), so nothing ever resolves an alias against a role.*, link.*, exec.* or
    event.* local name — an alias equal to one of those creates no ambiguity of
    any kind. The wider reading cost the vocabulary two real aliases (bare
    'framework' against role.framework, bare 'notice' against role.notice) for
    a collision that cannot happen, and both are restored. The docstring's own
    example is two DOCUMENT_TYPE codes, which this still catches, with
    test_no_alias_is_claimed_by_two_concepts behind it.
    """
    codes = {
        r["concept_code"].split(".", 1)[-1].lower() for r in concept_rows
        if r["domain"] == "DOCUMENT_TYPE"
    }
    collisions = {}
    for r in doc_type_rows:
        own = r["concept_code"].split(".", 1)[-1].lower()
        for alias in r["aliases"] or []:
            a = alias.strip().lower().replace(" ", "_")
            if a in codes and a != own:
                collisions.setdefault(r["concept_code"], []).append(alias)
    assert not collisions, f"aliases that are another concept's own name: {collisions}"


def _jsonable(identifiers):
    """Seed identifiers as plain lists of dicts; key order is irrelevant to
    equality, list order is kept because it is part of the data."""
    return [dict(i) for i in identifiers]


def test_concept_rows_equal_the_seed_column_for_column(concept_rows):
    """Set membership proves a row exists, not that it says the same thing.
    A changed definition or status would otherwise pass green."""
    drift = []
    for r in concept_rows:
        seed = CONCEPTS.get(r["concept_code"])
        if seed is None:
            continue  # reported by test_every_table_concept_exists_in_the_seed
        pairs = [
            ("domain", r["domain"], seed.domain),
            ("definition", r["definition"], seed.definition),
            ("not_to_be_confused_with", list(r["not_to_be_confused_with"] or []),
             list(seed.not_to_be_confused_with)),
        ]
        # A person may have confirmed or rejected a seeded row since; that is
        # ownership, not drift. Only rows still marked source='seed' are held.
        if r["source"] == "seed":
            pairs.append(("status", r["status"], seed.status))
        for col, got, want in pairs:
            if got != want:
                drift.append(f"{r['concept_code']}.{col}: table={got!r} seed={want!r}")
    assert not drift, "bp_concept differs from seed.py:\n  " + "\n  ".join(drift)


def test_document_type_rows_equal_the_seed_column_for_column(doc_type_rows):
    drift = []
    for r in doc_type_rows:
        seed = DOCUMENT_TYPES.get(r["concept_code"])
        if seed is None:
            continue
        ident = r["identifiers"]
        if isinstance(ident, str):
            import json
            ident = json.loads(ident)
        pairs = [
            ("role", r["role"], seed.role),
            ("default_parent_type", r["default_parent_type"], seed.default_parent_type),
            ("execution_mode", r["execution_mode"], seed.execution_mode),
            ("aliases", list(r["aliases"] or []), list(seed.aliases)),
            ("identifiers", ident, _jsonable(seed.identifiers)),
            ("structural_signals", list(r["structural_signals"] or []),
             list(seed.structural_signals)),
            ("pipeline_doc_type", r["pipeline_doc_type"], seed.pipeline_doc_type),
            ("requires_parent_evidence", bool(r["requires_parent_evidence"]),
             seed.requires_parent_evidence),
            ("parent_evidence_phrases", list(r["parent_evidence_phrases"] or []),
             list(seed.parent_evidence_phrases)),
        ]
        if r["source"] == "seed":
            pairs.append(("status", r["status"], seed.status))
        for col, got, want in pairs:
            if got != want:
                drift.append(f"{r['concept_code']}.{col}: table={got!r} seed={want!r}")
    assert not drift, "bp_document_type differs from seed.py:\n  " + "\n  ".join(drift)
