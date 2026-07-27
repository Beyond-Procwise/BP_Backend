"""Copy reference data verbatim from live into the test databases.

These tables are not synthesised. If FX rates, policies, prompts or the category
taxonomy differed from production, a passing test would say nothing about how the
product behaves against real configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.testdata.db import connect, copy_rows
from scripts.testdata.guards import assert_safe_target

REFERENCE_TABLES: dict[str, tuple[str, ...]] = {
    "bp_sqldb": (
        "bp_fx_rates",
        "bp_policy",
        "bp_prompt",
        "bp_admin_config",
        "bp_vendor_extraction_profiles",
        "bp_complaince_metric_prty_lkup",
        "procurement_patterns",
    ),
    "uicanvas": (
        "bp_category",
        "category",
        "category_mapping",
        "bp_policy",
        "bp_prompt",
    ),
}


@dataclass(frozen=True)
class TaxonomyLeaf:
    l1: str | None
    l2: str | None
    l3: str | None
    l4: str | None
    l5: str | None
    # The real per-level identifiers. cost_centre.linked_category_level_5_id and
    # item.category_id both want l5_id -- note it is NOT unique: 121 distinct
    # values across 246 leaves, because it identifies a node within its branch.
    l1_id: str | None
    l2_id: str | None
    l3_id: str | None
    l4_id: str | None
    l5_id: str | None
    unspsc_code: str | None
    esg_impact: str | None
    category_status: str | None
    spend_classification: str | None
    category_risk_rating: str | None
    audit_frequency: str | None
    policy_coverage: str | None

    @property
    def path(self) -> str:
        parts = [self.l1, self.l2, self.l3, self.l4, self.l5]
        return " > ".join(part for part in parts if part)


def load_taxonomy(dbname: str = "uicanvas") -> list[TaxonomyLeaf]:
    """The real 5-level taxonomy. Read-only against the source database."""
    conn = connect(dbname)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select category_level_1, category_level_2, category_level_3,
                       category_level_4, category_level_5,
                       category_level_1_id, category_level_2_id, category_level_3_id,
                       category_level_4_id, category_level_5_id,
                       unspsc_code, esg_impact,
                       category_status, spend_classification, category_risk_rating,
                       audit_frequency, policy_coverage
                from proc.bp_category
                order by 1, 2, 3, 4, 5
                """
            )
            return [TaxonomyLeaf(*row) for row in cur.fetchall()]
    finally:
        conn.close()


def _columns(conn, schema: str, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "select column_name from information_schema.columns "
            "where table_schema = %s and table_name = %s order by ordinal_position",
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def load_fx_rates(source_db: str) -> dict[str, float]:
    """Currency code -> USD-based rate, from the newest snapshot only.

    bp_fx_rates accumulates snapshots; mixing two of them would make the same
    currency convert two ways in one build. The build log records which
    snapshot was used, because a different snapshot means a different checksum.
    """
    conn = connect(source_db)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select currency, rate from proc.bp_fx_rates
                 where fetched_at = (select max(fetched_at) from proc.bp_fx_rates)
                   and base_currency = 'USD'
                """
            )
            return {code: float(rate) for code, rate in cur.fetchall()}
    finally:
        conn.close()


def copy_reference(source_db: str, target_db: str) -> dict[str, int]:
    """Copy every reference table for source_db into target_db. Returns row counts."""
    assert_safe_target(target_db)
    tables: Sequence[str] = REFERENCE_TABLES.get(source_db, ())

    source = connect(source_db)
    target = connect(target_db)
    written: dict[str, int] = {}
    try:
        for table in tables:
            columns = _columns(source, "proc", table)
            if not columns:
                continue
            column_list = ", ".join(f'"{c}"' for c in columns)
            with source.cursor() as cur:
                cur.execute(f'select {column_list} from proc."{table}"')
                rows = cur.fetchall()
            with target.cursor() as cur:
                cur.execute(f'truncate proc."{table}"')
            written[table] = copy_rows(target, "proc", table, columns, rows)
        target.commit()
    finally:
        source.close()
        target.close()
    return written
