from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
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

# The only fields an LLM may set from a buyer's message: their own scalar facts.
# Everything else on the row — status, completeness_score, the JSONB scope in
# specifications — is the agent's to write, and must not be overwritable by
# whatever the model happens to emit. Mirrors the "Allowed fields" list in the
# governing elicitation prompt; test_requirement_field_coercion keeps them in step.
ELICITABLE_FIELDS: Tuple[str, ...] = (
    "title", "category", "description", "quantity", "unit", "target_budget",
    "currency", "needed_by_date", "delivery_location", "priority",
)

# Columns the DB will reject a free-text value for. Everything else on
# bp_requirement is text/varchar with no length limit, so it needs no coercion.
_DATE_COLUMNS = {"needed_by_date"}
_NUMERIC_COLUMNS = {"quantity", "target_budget"}

# Explicit formats only — deliberately NOT dateutil. A permissive parser turns
# "3" into the third of this month and "next year" into a real date, which is
# fabrication dressed up as parsing. Anything not in this list is rejected and
# the field stays missing, so the agent asks the buyer for it.
_DATE_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
)
_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
# Currency symbols and thousands separators are formatting, not content.
_MONEY_STRIP_RE = re.compile(r"[£$€¥,\s]")


def _coerce_date(value: Any) -> Optional[str]:
    """Return an ISO date string, or None when the value is not a date."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if not text:
        return None
    text = _ORDINAL_RE.sub(r"\1", text).replace(",", " ")
    text = " ".join(text.split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _coerce_number(value: Any) -> Optional[Any]:
    """Return a number, or None when the value is prose.

    Strict on purpose: "10 units" is rejected rather than read as 10, because the
    unit belongs in ``unit`` and a partial read of a sentence is a guess. The one
    thing stripped is money formatting ("£480,000" → 480000).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return value
    text = _MONEY_STRIP_RE.sub("", str(value or "").strip())
    if not text or not _NUMERIC_RE.match(text):
        return None
    return float(text) if "." in text else int(text)


# Fields worth grounding: the ones that get ACTED on. A fabricated delivery
# address or deadline goes out to suppliers; a paraphrased title does not.
_GROUNDED_COLUMNS = _DATE_COLUMNS | _NUMERIC_COLUMNS | {"delivery_location"}

_MONTHS: Tuple[str, ...] = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
)
_MONTH_ALT = "|".join(_MONTHS)
# A month only counts as a date when a number sits beside it. "may" is a month and
# also an everyday verb, so a brief saying "we may need extra licences" must not
# ground an invented May date.
_MONTH_DATE_RE = re.compile(
    r"\b\d{1,4}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:" + _MONTH_ALT + r")\b"
    r"|\b(?:" + _MONTH_ALT + r")\b\s+\d{1,4}",
    re.I,
)
# Numeric dates need either slashes ("01/10", "01/10/2026") or a full
# dot/hyphen triple. Two components joined by a dot or hyphen are far more often
# a decimal or a range: "99.95% SLA" and "3-5 years" are not dates, and treating
# them as ones let a fabricated deadline through.
_NUMERIC_DATE_RE = re.compile(
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}\b"
)
_SEPARATORS_RE = re.compile(r"[,\s_]")
_WORD_RE = re.compile(r"[a-z]{4,}")


def _is_grounded(field: str, value: Any, source: str) -> bool:
    """Can this value be traced to something the buyer wrote?

    Deliberately format-tolerant, because legitimate normalisation changes the
    bytes: "01/10/2026" becomes 2026-10-01 and "£4,500" becomes 4500. So the
    check is for evidence of the fact, not a substring match — a year or a month
    named in the text for a date, the digits for a number, a shared word for a
    place. Same principle as the extraction pipeline's grounding guard.
    """
    src = (source or "").lower()
    if not src:
        return True  # nothing to check against; caller opted out
    if field in _NUMERIC_COLUMNS:
        digits = re.sub(r"\D", "", str(value))
        return bool(digits) and digits in _SEPARATORS_RE.sub("", src)
    if field in _DATE_COLUMNS:
        iso = str(value)
        if iso[:4] in src:                                  # the year is stated
            return True
        if _MONTH_DATE_RE.search(src):                      # "1 May", "July 2026"
            return True
        return bool(_NUMERIC_DATE_RE.search(src))           # "01/10", "01.10.2026"
    # Place names: at least one substantial word must come from the buyer.
    words = _WORD_RE.findall(str(value).lower())
    if not words:
        return True  # short codes ("UK HQ") carry nothing checkable
    return any(word in src for word in words)


def coerce_updates(
    updates: Any,
    allowed: Optional[Iterable[str]] = None,
    source_text: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Split LLM-proposed field updates into (persistable, rejected).

    Two things are enforced here, both learned the hard way on live data:

    * **Which fields** a model may set. Only ``ELICITABLE_FIELDS`` — the buyer's
      own scalar facts. An elicitation call once returned a ``specifications``
      value of its own and overwrote a scope the buyer had just accepted, along
      with the provenance recording that they accepted it. Structural columns
      (status, requirement_id, completeness_score, the JSONB scope) are owned by
      the agent, not by whatever the model felt like emitting.
    * **What type** the value can be. bp_requirement types its date and numeric
      columns, so "3 years from contract start" in ``needed_by_date`` made the
      INSERT raise and took the whole turn down with it.

    When ``source_text`` is given, dates, numbers and places are additionally
    checked for grounding in it — a robotics brief with no address once yielded
    delivery_location "Main Distribution Center, London, UK", which scored the
    requirement complete and handed it to sourcing.

    Bad values are dropped and reported, never repaired or guessed: a rejected
    field simply stays missing, and the agent asks the buyer for it.
    """
    if not isinstance(updates, dict):
        return {}, []
    permitted = set(ELICITABLE_FIELDS if allowed is None else allowed)
    clean: Dict[str, Any] = {}
    rejected: List[Dict[str, Any]] = []

    def _reject(field: str, value: Any, reason: str) -> None:
        rejected.append({"field": field, "value": value, "reason": reason})
        logger.warning("requirement field %r rejected (%s): %r", field, reason, value)

    for field, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if field not in permitted:
            _reject(field, value, "not an elicitable field")
            continue
        if field in _DATE_COLUMNS:
            iso = _coerce_date(value)
            if iso is None:
                _reject(field, value, "not a date")
                continue
            clean[field] = iso
        elif field in _NUMERIC_COLUMNS:
            number = _coerce_number(value)
            if number is None:
                _reject(field, value, "not a number")
                continue
            clean[field] = number
        else:
            clean[field] = value
        if field in _GROUNDED_COLUMNS and field in clean:
            if not _is_grounded(field, clean[field], source_text):
                del clean[field]
                _reject(field, value, "not grounded in the buyer's message")
    return clean, rejected


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


def get_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent bp_requirement row for a session_id, or None.

    This is the durable backing store for multi-turn elicitation when Redis is
    unavailable: each turn upserts the (gathering) row and the next turn reloads
    it here."""
    if not session_id:
        return None
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "select * from proc.bp_requirement where session_id = %s "
            "order by created_at desc limit 1",
            (session_id,),
        )
        cols = [d[0] for d in (cur.description or [])]
        row = cur.fetchone()
    if not row:
        return None
    return dict(zip(cols, row))


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
