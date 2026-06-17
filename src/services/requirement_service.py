from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.services.db import get_conn

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_FIELDS: Tuple[str, ...] = (
    "title", "category", "quantity", "needed_by_date", "delivery_location",
)

# Columns persisted to proc.bp_requirement (excludes DB-defaulted timestamps).
_PERSIST_COLUMNS: Tuple[str, ...] = (
    "requirement_id", "session_id", "status", "created_by", "title",
    "category", "description", "quantity", "unit", "target_budget",
    "currency", "needed_by_date", "delivery_location", "priority",
    "specifications", "constraints", "completeness_score",
    "missing_fields", "seed_context",
)
_JSONB_COLUMNS = {"specifications", "constraints", "missing_fields", "seed_context"}


def mint_requirement_id(created_by: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"REQ-{stamp}-{uuid.uuid4().hex[:8]}"


def evaluate_completeness(
    requirement: Dict[str, Any], required_fields: Iterable[str]
) -> Tuple[float, List[str]]:
    """Return (score, missing_fields). A field counts as filled when present
    and not None/blank. No fabrication: only genuinely-filled fields score."""
    required = list(required_fields) or list(DEFAULT_REQUIRED_FIELDS)
    missing: List[str] = []
    for name in required:
        value = requirement.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    filled = len(required) - len(missing)
    score = filled / len(required) if required else 1.0
    return score, missing


def seed_context(category: str) -> Dict[str, Any]:
    """Best-effort history summary for a category from final (_trgt) tables.
    Returns {} on any failure so a turn never breaks on infra issues."""
    if not category:
        return {}
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "select supplier_name, count(*) as n, avg(total_amount) as avg_amount "
                "from proc.bp_purchase_order_trgt "
                "where lower(coalesce(category, '')) = lower(%s) "
                "group by supplier_name order by n desc limit 5",
                (category,),
            )
            cols = [d[0] for d in (cur.description or [])]
            suppliers = [dict(zip(cols, r)) for r in cur.fetchall()]
        return {"category": category, "recent_suppliers": suppliers}
    except Exception:
        logger.debug("seed_context failed for category=%s", category, exc_info=True)
        return {}


def persist(record: Dict[str, Any]) -> None:
    """Upsert one proc.bp_requirement row keyed by requirement_id."""
    values = []
    for col in _PERSIST_COLUMNS:
        val = record.get(col)
        if col in _JSONB_COLUMNS and val is not None and not isinstance(val, str):
            val = json.dumps(val)
        values.append(val)
    placeholders = ", ".join(["%s"] * len(_PERSIST_COLUMNS))
    update_cols = [c for c in _PERSIST_COLUMNS if c != "requirement_id"]
    set_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
    sql = (
        f"insert into proc.bp_requirement ({', '.join(_PERSIST_COLUMNS)}) "
        f"values ({placeholders}) "
        f"on conflict (requirement_id) do update set {set_clause}, updated_at = now()"
    )
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(values))
        conn.commit()


def get_requirement(requirement_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "select * from proc.bp_requirement where requirement_id = %s",
            (requirement_id,),
        )
        cols = [d[0] for d in (cur.description or [])]
        row = cur.fetchone()
    if not row:
        return None
    return dict(zip(cols, row))


def list_requirements(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "select * from proc.bp_requirement order by created_at desc "
            "limit %s offset %s",
            (limit, offset),
        )
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]
