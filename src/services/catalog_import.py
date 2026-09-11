"""Import a distributor catalog feed into proc.bp_catalog_item.

Deliberately NOT part of the extraction pipeline. The extraction stack answers
"did the model read this document correctly", and that question earns its
confidence scores, its grounding guard, its judge and its promotion gate,
because a PDF is evidence and a model's reading of it is a claim.

A column in a distributor's price file is not a claim. The distributor is
telling us their cost. There is nothing to ground it against and no second
opinion to weigh, so this module is a deterministic column map with no model
call anywhere in it. It reuses only the parser, which already returns
structured cells for a spreadsheet (``Page.tables[0].rows`` of ``Cell``).

Three rules the code exists to enforce:

1. **The map is a row a human owns.** ``proc.bp_catalog_mapping`` says which of
   a distributor's headings feeds which of our columns. An unmapped required
   column rejects the import; it is never guessed from a similar heading.
2. **Absence stays absent.** A feed with no cost column produces NULL, never 0.
   A feed with no lifecycle column produces NULL, never ``'active'``.
3. **A price is a version, not an update.** A repriced row closes the current
   version and opens a new one, so a quote sent in March keeps reporting
   March's margin. An unchanged row produces no version at all.

**This function manages its own transaction and must not be called inside one.**
It commits the receipt before touching any item row, precisely so a crash
mid-import leaves a record, and it commits again at the end. A caller that wraps
it in a transaction of its own does not get one -- the first internal commit
ends the caller's transaction too, taking any earlier work with it. Verifying
this against a live database therefore needs a scratch schema or a throwaway
database, not a rolled-back transaction.

Not in this module, and not stubbed here either: cost-tier and relation feeds
(``bp_catalog_cost_tier``, ``bp_catalog_item_relation``). Those arrive as
separate files with their own shapes, and a placeholder that looked finished
would be worse than their absence.

Spec: docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2.extras

logger = logging.getLogger(__name__)


# The insert column order for proc.bp_catalog_item. Callers and tests read this
# rather than restating it; a column added here must be added to _row_params.
ITEM_COLUMNS: Tuple[str, ...] = (
    "source_id",
    "tenant_id",
    "distributor_id",
    "distributor_sku",
    "mpn",
    "manufacturer",
    "brand",
    "item_description",
    "unspsc_code",
    "unit_of_measure",
    "pack_size",
    "pack_uom",
    "currency",
    "list_price",
    "cost_price",
    "cost_basis",
    "availability_status",
    "stock_qty",
    "lead_time_days",
    "lifecycle_status",
    "end_of_sale_date",
    "end_of_life_date",
)

# What a mapping profile may target. source_id/tenant_id/distributor_id come
# from the import call, not from the file.
MAPPABLE_COLUMNS = frozenset(ITEM_COLUMNS) - {"source_id", "tenant_id", "distributor_id"}

# NOT NULL in the schema, so a row missing one of these cannot be written and is
# rejected rather than defaulted.
_REQUIRED_COLUMNS = ("distributor_sku", "item_description", "currency")

_DECIMAL_COLUMNS = frozenset({"pack_size", "list_price", "cost_price", "stock_qty"})
_INT_COLUMNS = frozenset({"lead_time_days"})
_DATE_COLUMNS = frozenset({"end_of_sale_date", "end_of_life_date"})

# The columns whose change makes a new version. source_id is excluded on
# purpose: re-sending the same prices in a differently-named file is not a
# price change, and versioning on it would churn history for nothing.
_MATERIAL_COLUMNS = tuple(
    c for c in ITEM_COLUMNS if c not in ("source_id", "tenant_id", "distributor_id")
)

_STATUS_IMPORTED = "imported"
_STATUS_PARTIAL = "partial"
_STATUS_FAILED = "failed"
_STATUS_DUPLICATE = "duplicate"

_DUP_STATUSES = (_STATUS_IMPORTED, _STATUS_PARTIAL)


@dataclass(frozen=True)
class RowReject:
    """One row we would not write, and why. Reported, never silently dropped."""

    sheet: int
    row_no: int
    reason: str
    sku: Optional[str] = None


@dataclass
class ImportResult:
    status: str
    source_id: Optional[int] = None
    rows_seen: int = 0
    rows_loaded: int = 0
    rows_rejected: int = 0
    rows_versioned: int = 0
    rows_unchanged: int = 0
    sheets_matched: int = 0
    sheets_skipped: int = 0
    rejects: List[RowReject] = field(default_factory=list)
    error: Optional[str] = None


# --- value coercion ---------------------------------------------------------

_NUM_KEEP = re.compile(r"[^0-9.\-]")
_ALL_INTS = re.compile(r"\d+")


def _decimal(raw: str) -> Decimal:
    return Decimal(raw)


def _apply_transform(target: str, raw: str, transform: Optional[str]) -> Any:
    """Return the value to write, or raise ValueError with a readable reason.

    A blank cell is always None -- for every target, transform and type. That is
    rule 2, and it is one branch rather than a special case per column.
    """
    text = (raw or "").strip()
    if not text:
        return None

    if transform == "pence_to_major":
        digits = _NUM_KEEP.sub("", text)
        if not digits:
            raise ValueError(f"{target}: {text!r} is not a pence amount")
        return _decimal(digits) / Decimal(100)

    if transform == "trim_currency":
        stripped = _NUM_KEEP.sub("", text)
        if not stripped:
            raise ValueError(f"{target}: {text!r} has no number in it")
        return _decimal(stripped)

    if transform == "pack_split":
        # The ONLY number in the cell, or nothing. "Box of 10" and "10 per pack"
        # both mean ten; "Box of 10 x 5" means something this transform cannot
        # settle, and picking one of the two would be a guess that reaches a
        # margin calculation. Ambiguity is refused, not resolved.
        found = _ALL_INTS.findall(text)
        if len(found) != 1:
            return None
        return _decimal(found[0])

    if transform:
        raise ValueError(f"{target}: unknown transform {transform!r}")

    if target in _DECIMAL_COLUMNS:
        try:
            return _decimal(text)
        except InvalidOperation:
            raise ValueError(f"{target}: {text!r} is not a number") from None

    if target in _INT_COLUMNS:
        try:
            return int(_decimal(text))
        except (InvalidOperation, ValueError):
            raise ValueError(f"{target}: {text!r} is not a whole number") from None

    if target in _DATE_COLUMNS:
        # psycopg2 casts an ISO string; anything else is the feed's problem to fix.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            raise ValueError(f"{target}: {text!r} is not an ISO date")
        return text

    return text


# --- mapping ----------------------------------------------------------------

def _norm_header(value: str) -> str:
    return " ".join((value or "").split()).casefold()


@dataclass(frozen=True)
class _MappingEntry:
    target_column: str
    source_header: str
    transform: Optional[str]
    is_required: bool


def _load_mapping(cur, mapping_profile: str) -> List[_MappingEntry]:
    cur.execute(
        "SELECT target_column, source_header, transform, is_required "
        "FROM proc.bp_catalog_mapping WHERE mapping_profile = %s",
        (mapping_profile,),
    )
    entries: List[_MappingEntry] = []
    for row in cur.fetchall() or []:
        target = row["target_column"]
        if target not in MAPPABLE_COLUMNS:
            raise ValueError(
                f"mapping profile {mapping_profile!r} targets {target!r}, "
                "which is not a column a feed may set"
            )
        entries.append(
            _MappingEntry(
                target_column=target,
                source_header=row["source_header"],
                transform=row["transform"],
                is_required=bool(row["is_required"]) or target in _REQUIRED_COLUMNS,
            )
        )
    return entries


def _load_unspsc(cur) -> frozenset:
    """The live category codes. A soft reference the importer has to police,
    because bp_category_master is a view over a foreign table and Postgres
    cannot declare a foreign key to one."""
    cur.execute(
        "SELECT unspsc_code FROM proc.bp_category_master WHERE unspsc_code IS NOT NULL"
    )
    return frozenset(r["unspsc_code"] for r in (cur.fetchall() or []))


# --- sheet reading ----------------------------------------------------------

def _sheet_rows(table) -> List[List[str]]:
    return [[c.text for c in row] for row in table.rows]


def _header_index(header: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, h in enumerate(header):
        key = _norm_header(h)
        if key and key not in out:
            out[key] = i
    return out


def _sheet_matches(index: Dict[str, int], mapping: Iterable[_MappingEntry]) -> bool:
    return all(
        _norm_header(m.source_header) in index for m in mapping if m.is_required
    )


# --- versioning -------------------------------------------------------------

_SELECT_CURRENT = (
    "SELECT catalog_item_id, " + ", ".join(_MATERIAL_COLUMNS) + " "
    "FROM proc.bp_catalog_item "
    "WHERE distributor_id = %s AND distributor_sku = %s AND valid_to IS NULL"
)

_CLOSE_CURRENT = (
    "UPDATE proc.bp_catalog_item SET valid_to = now() WHERE catalog_item_id = %s"
)

_INSERT_ITEM = (
    "INSERT INTO proc.bp_catalog_item (" + ", ".join(ITEM_COLUMNS) + ") "
    "VALUES (" + ", ".join(["%s"] * len(ITEM_COLUMNS)) + ")"
)


def _same_as_current(values: Dict[str, Any], current: Dict[str, Any]) -> bool:
    for col in _MATERIAL_COLUMNS:
        new, old = values.get(col), current.get(col)
        if isinstance(new, Decimal) or isinstance(old, Decimal):
            if (new is None) != (old is None):
                return False
            if new is not None and Decimal(str(new)) != Decimal(str(old)):
                return False
            continue
        if (new or None) != (old or None):
            return False
    return True


# --- the receipt ------------------------------------------------------------

_FIND_SOURCE = (
    "SELECT source_id, status FROM proc.bp_catalog_source "
    "WHERE distributor_id = %s AND content_sha256 = %s"
)

# Upserted, not inserted: a retry of a file that failed must be able to reuse
# its receipt row, and UNIQUE (distributor_id, content_sha256) forbids a second.
_UPSERT_SOURCE = """
INSERT INTO proc.bp_catalog_source (
    tenant_id, distributor_id, feed_name, file_name, content_sha256,
    mapping_profile, price_effective, status, imported_by
) VALUES (%s, %s, %s, %s, %s, %s, %s, 'failed', %s)
ON CONFLICT (distributor_id, content_sha256) DO UPDATE SET
    feed_name       = EXCLUDED.feed_name,
    file_name       = EXCLUDED.file_name,
    mapping_profile = EXCLUDED.mapping_profile,
    price_effective = EXCLUDED.price_effective,
    status          = 'failed',
    imported_by     = EXCLUDED.imported_by,
    attempt_count   = proc.bp_catalog_source.attempt_count + 1,
    created_date    = now()
RETURNING source_id
"""

_FINISH_SOURCE = (
    "UPDATE proc.bp_catalog_source SET status = %s, rows_seen = %s, "
    "rows_loaded = %s, rows_rejected = %s, error = %s WHERE source_id = %s"
)


def _dict_cursor(conn: Any):
    try:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except TypeError:  # a fake cursor that takes no kwargs
        return conn.cursor()


# --- entry point ------------------------------------------------------------

def import_catalog(
    *,
    distributor_id: str,
    feed_name: str,
    mapping_profile: str,
    price_effective: Any,
    imported_by: str,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    file_path: Optional[str] = None,
    parsed: Any = None,
    tenant_id: Optional[str] = None,
    conn: Any = None,
) -> ImportResult:
    """Load one catalog feed. Returns what happened; raises only on programmer error.

    ``file_bytes`` is hashed for idempotency -- of the FILE, not of the parse,
    so a parser change never makes an already-loaded feed look new. Pass
    ``parsed`` to skip parsing, or ``file_path`` to have it parsed here.
    """
    if conn is None:
        from src.services.db import get_conn

        with get_conn() as owned:
            return import_catalog(
                distributor_id=distributor_id, feed_name=feed_name,
                mapping_profile=mapping_profile, price_effective=price_effective,
                imported_by=imported_by, file_bytes=file_bytes, file_name=file_name,
                file_path=file_path, parsed=parsed, tenant_id=tenant_id, conn=owned,
            )

    if file_bytes is None and file_path is None:
        raise ValueError("import_catalog needs file_bytes or file_path")
    if file_bytes is None:
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()

    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    cur = _dict_cursor(conn)

    # 1. Already loaded? Only a SUCCESSFUL prior attempt counts -- otherwise a
    #    failure would be permanent and the file could never be retried.
    cur.execute(_FIND_SOURCE, (distributor_id, content_sha256))
    prior = cur.fetchone()
    if prior and prior.get("status") in _DUP_STATUSES:
        logger.info(
            "catalog feed already loaded for %s (source_id=%s); nothing to do",
            distributor_id, prior["source_id"],
        )
        return ImportResult(status=_STATUS_DUPLICATE, source_id=prior["source_id"])

    # 2. The receipt exists BEFORE any item work and starts as 'failed'. That is
    #    the honest initial state, and it is why a crash mid-import can never
    #    look like a distributor who simply sent us nothing.
    cur.execute(
        _UPSERT_SOURCE,
        (tenant_id, distributor_id, feed_name, file_name, content_sha256,
         mapping_profile, price_effective, imported_by),
    )
    source_id = cur.fetchone()["source_id"]
    conn.commit()

    result = ImportResult(status=_STATUS_FAILED, source_id=source_id)
    try:
        if parsed is None:
            from src.services.extraction_v3.parsers.router import parse

            parsed = parse(file_path)

        mapping = _load_mapping(cur, mapping_profile)
        if not mapping:
            raise ValueError(
                f"no mapping profile {mapping_profile!r} in proc.bp_catalog_mapping; "
                "a distributor's headings are declared, not inferred"
            )

        valid_unspsc = (
            _load_unspsc(cur)
            if any(m.target_column == "unspsc_code" for m in mapping)
            else frozenset()
        )

        _read_sheets(
            cur, parsed, mapping, valid_unspsc, result,
            source_id=source_id, tenant_id=tenant_id, distributor_id=distributor_id,
        )

        if result.sheets_matched == 0:
            raise ValueError(
                "no sheet in this file carries every required heading for profile "
                f"{mapping_profile!r}; nothing was loaded"
            )

        result.status = _STATUS_PARTIAL if result.rows_rejected else _STATUS_IMPORTED

    except Exception as exc:  # noqa: BLE001 - the receipt must record any failure
        conn.rollback()
        result.status = _STATUS_FAILED
        result.error = str(exc)
        logger.exception("catalog import failed for %s", distributor_id)

    cur.execute(
        _FINISH_SOURCE,
        (result.status, result.rows_seen, result.rows_loaded,
         result.rows_rejected, result.error, source_id),
    )
    conn.commit()
    return result


def _read_sheets(
    cur, parsed, mapping, valid_unspsc, result, *,
    source_id: int, tenant_id: Optional[str], distributor_id: str,
) -> None:
    for page in parsed.pages:
        for table in page.tables:
            rows = _sheet_rows(table)
            if not rows:
                result.sheets_skipped += 1
                continue
            index = _header_index(rows[0])
            if not _sheet_matches(index, mapping):
                # A cover sheet, or a sheet for another profile. Counted, so an
                # operator can see the file was not read the way they expected.
                result.sheets_skipped += 1
                continue
            result.sheets_matched += 1
            for row_no, raw_row in enumerate(rows[1:], start=2):
                if not any((c or "").strip() for c in raw_row):
                    continue
                result.rows_seen += 1
                _read_row(
                    cur, raw_row, index, mapping, valid_unspsc, result,
                    sheet=page.index, row_no=row_no, source_id=source_id,
                    tenant_id=tenant_id, distributor_id=distributor_id,
                )


def _read_row(
    cur, raw_row, index, mapping, valid_unspsc, result, *,
    sheet: int, row_no: int, source_id: int, tenant_id: Optional[str],
    distributor_id: str,
) -> None:
    values: Dict[str, Any] = {c: None for c in ITEM_COLUMNS}
    values.update(
        source_id=source_id, tenant_id=tenant_id, distributor_id=distributor_id
    )

    def reject(reason: str) -> None:
        result.rows_rejected += 1
        result.rejects.append(
            RowReject(sheet=sheet, row_no=row_no, reason=reason,
                      sku=values.get("distributor_sku"))
        )

    for m in mapping:
        col = index.get(_norm_header(m.source_header))
        raw = raw_row[col] if col is not None and col < len(raw_row) else ""
        try:
            values[m.target_column] = _apply_transform(
                m.target_column, raw, m.transform
            )
        except ValueError as exc:
            reject(str(exc))
            return

    for col in _REQUIRED_COLUMNS:
        if values.get(col) in (None, ""):
            reject(f"{col} is empty, and the column is NOT NULL")
            return

    code = values.get("unspsc_code")
    if code and code not in valid_unspsc:
        reject(f"unspsc_code {code!r} is not in the live category master")
        return

    cur.execute(_SELECT_CURRENT, (distributor_id, values["distributor_sku"]))
    current = cur.fetchone()

    if current and _same_as_current(values, current):
        result.rows_unchanged += 1
        return

    if current:
        cur.execute(_CLOSE_CURRENT, (current["catalog_item_id"],))
        result.rows_versioned += 1

    cur.execute(_INSERT_ITEM, tuple(values[c] for c in ITEM_COLUMNS))
    result.rows_loaded += 1
