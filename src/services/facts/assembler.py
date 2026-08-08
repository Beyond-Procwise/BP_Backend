"""The Fact Assembler — _trgt rows in, provenanced facts out.

Deterministic. No LLM, no inference, no defaulting. A line that cannot be
evidenced produces no fact at all.

JOIN DIRECTION (F1). The assembler drives from the _trgt line row and looks
provenance up by (doc_type, doc_pk, field_path). It never enumerates provenance
and joins forward. Measured on bp_sqldb, joining provenance -> _trgt looks
catastrophic — of 125 quote doc_pks in bp_extraction_provenance_v3 only 45
match a bp_quote_trgt.quote_id — but the unmatched keys are values like '10'
and '048597': failed extraction attempts whose primary key was garbage and
which never promoted. Provenance records every attempt. Driven the other way,
coverage is near-perfect, and mandatory provenance is achievable precisely
because of that direction.

LINE INDEX (verified against bp_sqldb, not assumed). Provenance writes
``line_items[0].unit_price`` for the FIRST line; the _trgt line tables number
lines from 1. Matching on ``line_no - 1`` reproduces the _trgt row's own
unit_price, while matching on ``line_no`` returns a different line's price. An
off-by-one here would silently attach the wrong line's evidence to a price,
which is worse than carrying no evidence at all.

FIELD NAMES. Provenance field paths use the extraction schema's field NAME
(``line_amount``), while the _trgt column may differ (``line_total`` on quotes
and purchase orders). That mapping is read from ``extraction_schemas/*.yaml``
rather than restated here, so it cannot drift into a second vocabulary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence

import yaml

from src.services.facts.arithmetic import check_line_arithmetic
from src.services.facts.concept_codes import SCHEMA_DIR
from src.services.facts.fx import FX_UNAVAILABLE, resolve_fx
from src.services.facts.models import (
    ArithmeticState,
    CommercialFact,
    FactProvenance,
    MeasureRole,
    ValidationState,
    ValueBasis,
)
from src.services.facts.uom import UOM_UNMAPPED, ensure_vocabulary, normalise_uom

logger = logging.getLogger(__name__)

#: The document stated no unit at all. Distinct from UOM_UNMAPPED, which means
#: the document stated something that is not a unit.
UOM_ABSENT = "UOM_ABSENT"

#: What a unit rate is "per" when the document never said.
#:
#: Measured on bp_sqldb: unit_of_measure is NULL on 152/152 invoice lines,
#: 171/171 purchase order lines and 205/316 quote lines. F5 enumerated the
#: distinct non-null values and so never surfaced this. Three options existed
#: and only one is honest:
#:
#:   - default to 'each'  -> fabricates a unit for the entire corpus
#:   - call it an extended_line -> restates a unit price as a total, which is
#:     precisely the bug F6 exists to prevent
#:   - record explicitly that the document did not say
#:
#: The sentinel is deliberately not a unit name, so nothing can mistake it for
#: one, and it must never normalise to a canonical unit. A comparison layer is
#: expected to refuse to compare two rates whose basis is unstated.
BASIS_UOM_UNSTATED = "unstated"

#: Reporting currency. Facts stamp their own rate so a re-render cannot move
#: yesterday's number.
BASE_CURRENCY = "GBP"


@dataclass(frozen=True)
class _DocConfig:
    schema: str
    lines_table: str
    pk_column: str
    line_no_column: str
    header_table: Optional[str]
    header_pk: Optional[str]


# bp_po_trgt does not exist in proc (the surviving header table is
# bp_po_trgt_june12), so purchase order lines carry their own currency and have
# no header to enrich from. Recorded rather than worked around.
_DOC_CONFIG: Dict[str, _DocConfig] = {
    "invoice": _DocConfig(
        schema="invoice.yaml",
        lines_table="proc.bp_invoice_line_items_trgt",
        pk_column="invoice_id",
        line_no_column="line_no",
        header_table="proc.bp_invoice_trgt",
        header_pk="invoice_id",
    ),
    "quote": _DocConfig(
        schema="quote.yaml",
        lines_table="proc.bp_quote_line_items_trgt",
        pk_column="quote_id",
        line_no_column="line_number",
        header_table="proc.bp_quote_trgt",
        header_pk="quote_id",
    ),
    "purchase_order": _DocConfig(
        schema="purchase_order.yaml",
        lines_table="proc.bp_po_line_items_trgt",
        pk_column="po_id",
        line_no_column="line_number",
        header_table=None,
        header_pk=None,
    ),
}

# Provenance rows for the same (doc_pk, field_path) exist once per extraction
# attempt. Pick one deterministically, best first, or two runs could stamp
# different evidence onto the same fact.
_PROVENANCE_SQL = """
    SELECT provenance_id, doc_type, doc_pk, field_path, value, page,
           bbox_x0, bbox_y0, bbox_x1, bbox_y1, evidence_text,
           model, final_confidence, extracted_at
      FROM proc.bp_extraction_provenance_v3
     WHERE doc_type = %s AND doc_pk = %s
       AND field_path LIKE 'line_items[%%'
     ORDER BY final_confidence DESC NULLS LAST,
              extracted_at DESC NULLS LAST,
              provenance_id DESC
"""

#: The economic fields a line fact is built from, by schema field name.
_ECONOMIC_FIELDS = ("unit_price", "quantity", "line_amount", "total_amount",
                    "unit_of_measure", "item_description")

_schema_cache: Dict[str, Dict[str, str]] = {}


def _line_field_map(schema_file: str) -> Dict[str, str]:
    """schema field name -> _trgt column name, read from the extraction schema."""
    if schema_file in _schema_cache:
        return _schema_cache[schema_file]

    mapping: Dict[str, str] = {}
    path = SCHEMA_DIR / schema_file
    try:
        doc = yaml.safe_load(path.read_text()) or {}
        line_items = doc.get("line_items") or {}
        if isinstance(line_items, dict):
            for field in line_items.get("fields") or []:
                name = (field or {}).get("name")
                if name:
                    mapping[str(name)] = str(field.get("db_column") or name)
    except Exception:
        logger.exception("could not read line-item field map from %s", path)

    _schema_cache[schema_file] = mapping
    return mapping


def _rows(cur) -> List[Dict[str, Any]]:
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in (cur.fetchall() or [])]


def _dec(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _locator(row: Dict[str, Any]) -> str:
    """A pointer into the page. Never blank — the model rejects a blank one."""
    box = [row.get("bbox_x0"), row.get("bbox_y0"), row.get("bbox_x1"), row.get("bbox_y1")]
    if all(v is not None for v in box):
        return "bbox:" + ",".join(str(v) for v in box)
    page = row.get("page")
    if page is not None:
        return f"page:{page}"
    return f"field:{row.get('field_path')}"


def _rank(row: Dict[str, Any]) -> tuple:
    """Best-attempt ordering: confidence, then recency, then row id."""
    confidence = row.get("final_confidence")
    extracted_at = row.get("extracted_at")
    return (
        float(confidence) if confidence is not None else float("-inf"),
        extracted_at.timestamp() if hasattr(extracted_at, "timestamp") else float("-inf"),
        int(row.get("provenance_id") or 0),
    )


def _provenance_index(cur, doc_type: str, doc_pk: str) -> Dict[str, Dict[str, Any]]:
    """field_path -> the single best provenance row for it.

    The same (doc_pk, field_path) exists once per extraction attempt, so a pick
    is unavoidable. It is made here in Python rather than relying on the SQL
    ORDER BY alone: the in-memory store this codebase substitutes under pytest
    does not implement ordering, and determinism that only holds against real
    PostgreSQL is not determinism. The ORDER BY stays as an optimisation.
    """
    cur.execute(_PROVENANCE_SQL, (doc_type, doc_pk))
    index: Dict[str, Dict[str, Any]] = {}
    for row in _rows(cur):
        path = row.get("field_path")
        if not path:
            continue
        incumbent = index.get(path)
        if incumbent is None or _rank(row) > _rank(incumbent):
            index[path] = row
    return index


def _header(cur, cfg: _DocConfig, doc_pk: str) -> Dict[str, Any]:
    if not cfg.header_table:
        return {}
    try:
        cur.execute(
            f"SELECT * FROM {cfg.header_table} WHERE {cfg.header_pk} = %s LIMIT 1",
            (doc_pk,),
        )
        rows = _rows(cur)
        return rows[0] if rows else {}
    except Exception:
        logger.exception("could not read header for %s %s", doc_type_safe(cfg), doc_pk)
        return {}


def doc_type_safe(cfg: _DocConfig) -> str:
    return cfg.lines_table


def _to_provenance(row: Dict[str, Any], document_id: str) -> Optional[FactProvenance]:
    try:
        return FactProvenance(
            document_id=document_id,
            doc_type=row.get("doc_type"),
            extraction_id=(str(row["provenance_id"])
                           if row.get("provenance_id") is not None else None),
            field_path=row.get("field_path") or "",
            page=row.get("page"),
            locator=_locator(row),
            verbatim_snippet=row.get("evidence_text") or row.get("value"),
            extracted_at=row.get("extracted_at"),
            model=row.get("model"),
            confidence=row.get("final_confidence"),
        )
    except Exception:
        logger.warning("discarding malformed provenance row %s", row.get("provenance_id"))
        return None


def assemble_line_facts(cur, doc_type: str, doc_pk: str) -> List[CommercialFact]:
    """Build the facts for one document's lines.

    Returns an empty list rather than an unprovenanced fact whenever a line has
    no evidence behind it. That is F2's fail-closed behaviour: on a seeded
    corpus it means almost nothing is produced, which is correct, not a bug.
    """
    cfg = _DOC_CONFIG.get(doc_type)
    if cfg is None:
        raise ValueError(
            f"unsupported doc_type {doc_type!r}: expected one of "
            f"{sorted(_DOC_CONFIG)}"
        )

    # Refresh the unit vocabulary from proc.bp_uom_canonical, reusing this
    # cursor. Cached behind a TTL, so this is a no-op on all but the first call
    # in a run -- normalise_uom is called once per line and must not carry a
    # query with it.
    ensure_vocabulary(cur)

    field_map = _line_field_map(cfg.schema)
    amount_column = field_map.get("line_amount", "line_amount")
    total_column = field_map.get("total_amount", "total_amount")

    cur.execute(
        f"SELECT * FROM {cfg.lines_table} WHERE {cfg.pk_column} = %s "
        f"ORDER BY {cfg.line_no_column}",
        (doc_pk,),
    )
    lines = _rows(cur)
    if not lines:
        logger.info("no %s lines for %s", doc_type, doc_pk)
        return []

    provenance_index = _provenance_index(cur, doc_type, doc_pk)
    header = _header(cur, cfg, doc_pk)

    facts: List[CommercialFact] = []
    for line in lines:
        fact = _assemble_one(
            cur, cfg, doc_type, doc_pk, line, provenance_index, header,
            amount_column, total_column,
        )
        if fact is not None:
            facts.append(fact)
    return facts


def _assemble_one(
    cur,
    cfg: _DocConfig,
    doc_type: str,
    doc_pk: str,
    line: Dict[str, Any],
    provenance_index: Dict[str, Dict[str, Any]],
    header: Dict[str, Any],
    amount_column: str,
    total_column: str,
) -> Optional[CommercialFact]:
    line_no = line.get(cfg.line_no_column)
    if line_no is None:
        logger.info("skipping %s %s line with no line number", doc_type, doc_pk)
        return None

    # THE off-by-one. Provenance is 0-based, the _trgt tables are 1-based.
    index = int(line_no) - 1

    document_id = str(line.get("document_id") or doc_pk or "").strip() or doc_pk

    matched: List[FactProvenance] = []
    for field in _ECONOMIC_FIELDS:
        row = provenance_index.get(f"line_items[{index}].{field}")
        if row is None:
            continue
        prov = _to_provenance(row, document_id)
        if prov is not None:
            matched.append(prov)

    if not matched:
        # Fail closed. A fact with no evidence is exactly what this phase
        # exists to make impossible.
        logger.info(
            "no provenance for %s %s line %s (field_path line_items[%s].*) "
            "- no fact produced",
            doc_type, doc_pk, line_no, index,
        )
        return None

    reason_codes: List[str] = []

    quantity = _dec(line.get("quantity"))
    unit_price = _dec(line.get("unit_price"))
    extended = _dec(line.get(amount_column))
    if extended is None:
        extended = _dec(line.get(total_column))

    # --- unit of measure -----------------------------------------------------
    raw_uom = line.get("unit_of_measure")
    uom_result = normalise_uom(raw_uom)
    has_raw_uom = isinstance(raw_uom, str) and raw_uom.strip() != ""

    if uom_result.canonical is not None:
        basis_uom = uom_result.canonical
    elif has_raw_uom:
        # Unmappable, so carry the document's own string forward untouched.
        basis_uom = raw_uom
        reason_codes.append(UOM_UNMAPPED)
    else:
        basis_uom = BASIS_UOM_UNSTATED
        reason_codes.append(UOM_ABSENT)
    reason_codes.extend(c for c in uom_result.reason_codes if c != UOM_UNMAPPED)

    # --- role ----------------------------------------------------------------
    if unit_price is not None:
        measure_role = MeasureRole.UNIT_RATE
    elif extended is not None:
        # A lump-sum services line: stating "this is a total" is truthful,
        # where a NULL unit_price would merely read as missing data.
        measure_role = MeasureRole.EXTENDED_LINE
        basis_uom = None
    else:
        measure_role = None
        basis_uom = None

    arithmetic_state: Optional[ArithmeticState] = None
    if measure_role is not None:
        arithmetic_state = check_line_arithmetic(quantity, unit_price, extended)

    # --- currency ------------------------------------------------------------
    currency = line.get("currency") or header.get("currency")
    fx = resolve_fx(cur, currency, BASE_CURRENCY) if currency else None
    fx_rate = fx.rate if fx else None
    if fx is not None and FX_UNAVAILABLE in fx.reason_codes:
        reason_codes.append(FX_UNAVAILABLE)

    extended_base = extended * fx_rate if (extended is not None and fx_rate is not None) else None

    try:
        return CommercialFact(
            fact_id=f"{doc_type}|{doc_pk}|L{line_no}",
            fact_type="line_unit_price" if measure_role is MeasureRole.UNIT_RATE else "line_amount",
            concept_code="unit_price" if measure_role is MeasureRole.UNIT_RATE else "line_amount",
            source_doc_type=doc_type,
            source_doc_pk=str(doc_pk),
            document_id=document_id,
            line_no=int(line_no),
            measure_role=measure_role,
            basis_uom=basis_uom,
            arithmetic_state=arithmetic_state,
            unit_price=unit_price,
            quantity=quantity,
            extended_value=extended,
            tax_amount=_dec(line.get("tax_amount")),
            uom=raw_uom,
            uom_normalised=uom_result.canonical,
            uom_dimension=uom_result.dimension,
            currency=currency,
            base_currency=BASE_CURRENCY if fx_rate is not None else None,
            extended_value_base=extended_base,
            fx_rate=fx_rate,
            fx_rate_date=fx.rate_date if fx else None,
            fx_rate_source=fx.source if fx else None,
            supplier_id=header.get("supplier_id"),
            buyer_id=header.get("buyer_id"),
            item_reference=line.get("item_id"),
            item_description=line.get("item_description"),
            contract_id=header.get("contract_id"),
            deal_id=line.get("deal_id") or header.get("deal_id"),
            region=line.get("region") or header.get("region"),
            country=line.get("country") or header.get("country"),
            value_basis=ValueBasis.AS_SUPPLIED,
            validation_state=(
                ValidationState.INVALID
                if arithmetic_state is ArithmeticState.INCONSISTENT
                else ValidationState.UNVERIFIED
            ),
            reason_codes=reason_codes,
            provenance=matched,
        )
    except Exception:
        # A line the model refuses is reported, never silently dropped.
        logger.exception(
            "could not construct a fact for %s %s line %s", doc_type, doc_pk, line_no
        )
        return None


def assemble_document_facts(cur, doc_type: str, doc_pks: Sequence[str]) -> List[CommercialFact]:
    """Convenience sweep over several documents of one type."""
    out: List[CommercialFact] = []
    for pk in doc_pks:
        out.extend(assemble_line_facts(cur, doc_type, pk))
    return out
