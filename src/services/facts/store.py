"""Persistence for facts, provenance, constraints and finding links.

Ordering matters. A fact row is written before its provenance rows, because
provenance references the fact; the database's mandatory-provenance rule is a
DEFERRABLE INITIALLY DEFERRED constraint trigger precisely so that the pair can
be written in either order within one transaction and checked only at COMMIT.

This module never opens or commits a transaction of its own. The caller owns
the transaction boundary, so a partial write is the caller's to roll back —
and a fact whose provenance failed to insert must never be committed alone.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

from src.services.facts.models import CommercialFact, Constraint

logger = logging.getLogger(__name__)

_FACT_COLUMNS: Tuple[str, ...] = (
    "fact_id", "tenant_id", "fact_type", "concept_code",
    "source_doc_type", "source_doc_pk", "document_id", "document_version", "line_no",
    "measure_role", "basis_uom", "arithmetic_state",
    "unit_price", "quantity", "extended_value", "tax_amount", "discount_amount",
    "uom", "uom_normalised", "uom_dimension",
    "currency", "base_currency", "extended_value_base",
    "fx_rate", "fx_rate_date", "fx_rate_source",
    "supplier_id", "supplier_name", "buyer_id", "item_reference", "item_description",
    "category_l1", "category_l2", "category_l3", "category_l4",
    "contract_id", "term_start", "term_end", "term_months", "billing_frequency",
    "escalator_pct", "escalator_basis", "escalator_cap_pct",
    "cost_centre", "region", "country",
    "bundle_group_id", "deal_id",
    "value_basis", "validation_state", "reason_codes", "confidence",
)

_PROVENANCE_COLUMNS: Tuple[str, ...] = (
    "fact_id", "tenant_id", "document_id", "doc_type", "extraction_id",
    "field_path", "page", "locator", "verbatim_snippet", "model",
    "confidence", "extracted_at",
)


def _enum(value):
    return getattr(value, "value", value)


def _require_transaction(cur) -> None:
    """Refuse to write on an autocommit connection.

    ``src.services.db.get_conn()`` returns a connection with autocommit ON. On
    such a connection every statement is its own transaction, so the deferred
    provenance trigger is evaluated the instant the fact row is inserted —
    before its provenance can possibly exist — and every write fails with a
    confusing CheckViolation naming a fact that was about to be evidenced.

    Worse is the case where it does not fail: a fact and its provenance written
    as two separate transactions are momentarily committed apart, and a crash
    between them leaves a fact the trigger would have rejected. The guarantee
    only holds inside one transaction, so this refuses rather than pretends.
    """
    conn = getattr(cur, "connection", None)
    if conn is not None and getattr(conn, "autocommit", False):
        raise RuntimeError(
            "persist_facts requires a transaction: set conn.autocommit = False "
            "before writing. The mandatory-provenance trigger is DEFERRABLE "
            "INITIALLY DEFERRED and is only meaningful at COMMIT."
        )


def _fact_values(fact: CommercialFact) -> list:
    row = []
    for column in _FACT_COLUMNS:
        row.append(_enum(getattr(fact, column, None)))
    return row


def persist_facts(cur, facts: Sequence[CommercialFact]) -> int:
    """Upsert facts and replace their provenance. Returns the number written.

    Re-running the assembler over the same document must not duplicate: fact_id
    is deterministic, so this upserts. Provenance is replaced rather than
    appended for the same reason — otherwise a second run doubles the evidence
    rows behind an unchanged fact.
    """
    if not facts:
        return 0

    _require_transaction(cur)

    fact_cols = ", ".join(_FACT_COLUMNS)
    fact_ph = ", ".join(["%s"] * len(_FACT_COLUMNS))
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in _FACT_COLUMNS if c != "fact_id"
    )
    fact_sql = (
        f"INSERT INTO proc.bp_commercial_fact ({fact_cols}) VALUES ({fact_ph}) "
        f"ON CONFLICT (fact_id) DO UPDATE SET {updates}, recorded_at = now()"
    )

    prov_cols = ", ".join(_PROVENANCE_COLUMNS)
    prov_ph = ", ".join(["%s"] * len(_PROVENANCE_COLUMNS))
    prov_sql = (
        f"INSERT INTO proc.bp_fact_provenance ({prov_cols}) VALUES ({prov_ph})"
    )

    written = 0
    for fact in facts:
        cur.execute(fact_sql, _fact_values(fact))
        # Replace, do not append. The deferred trigger tolerates the momentary
        # gap because it is only evaluated at COMMIT, by which point the new
        # rows are in place.
        cur.execute(
            "DELETE FROM proc.bp_fact_provenance WHERE fact_id = %s", (fact.fact_id,)
        )
        for prov in fact.provenance:
            cur.execute(prov_sql, [
                fact.fact_id, fact.tenant_id, prov.document_id, prov.doc_type,
                prov.extraction_id, prov.field_path, prov.page, prov.locator,
                prov.verbatim_snippet, prov.model, prov.confidence, prov.extracted_at,
            ])
        written += 1

    return written


def link_facts_to_finding(
    cur, opportunity_ref_id: str, facts: Iterable[CommercialFact], role: str,
    tenant_id: str = "default",
) -> int:
    """Attach facts to an opportunity with a stated role.

    Many-to-many by design: one invoice line can support both an overbilling
    finding and a duplicate finding, and 301 of 308 opportunities draw on three
    source documents.
    """
    sql = (
        "INSERT INTO proc.bp_finding_fact "
        "(opportunity_ref_id, fact_id, role, tenant_id) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (opportunity_ref_id, fact_id, role) DO NOTHING"
    )
    linked = 0
    for fact in facts:
        cur.execute(sql, (opportunity_ref_id, fact.fact_id, role, tenant_id))
        linked += 1
    return linked


_CONSTRAINT_COLUMNS: Tuple[str, ...] = (
    "constraint_id", "tenant_id", "constraint_type", "concept_code",
    "bound_value", "bound_uom", "bound_direction", "bound_currency",
    "bound_basis", "measurement_period", "applies_to_entities",
    "applies_to_documents", "testability_state",
    "source_doc_type", "source_doc_pk", "document_id", "contract_id",
    "effective_from", "effective_to",
    "validation_state", "reason_codes", "confidence",
)


def persist_constraints(cur, constraints: Sequence[Constraint]) -> int:
    """Upsert constraints. Provenance for constraints is not yet stored: the
    corpus contains zero contracts, so there is nothing to write and a table
    written by nothing would look like a working path."""
    if not constraints:
        return 0

    cols = ", ".join(_CONSTRAINT_COLUMNS)
    ph = ", ".join(["%s"] * len(_CONSTRAINT_COLUMNS))
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in _CONSTRAINT_COLUMNS if c != "constraint_id"
    )
    sql = (
        f"INSERT INTO proc.bp_constraint ({cols}) VALUES ({ph}) "
        f"ON CONFLICT (constraint_id) DO UPDATE SET {updates}, recorded_at = now()"
    )

    written = 0
    for c in constraints:
        cur.execute(sql, [_enum(getattr(c, col, None)) for col in _CONSTRAINT_COLUMNS])
        written += 1
    return written
