"""Single-flow extraction dispatch.

Orchestrates L0 (parse) → L1 (regex) → L3 grounding + invariants → persist.
L2 engineered modules (table, NER, address, date, bbox proximity) and the
LLM judge are wired into the gaps that this MVP path leaves open — added
incrementally as live testing exposes them.
"""
from __future__ import annotations

import logging
import re
import subprocess
from typing import Any
from uuid import uuid4

from src.services.agent_actions import record_action, PHASE_EXTRACTION
from src.services.extraction import completeness as _completeness
from src.services.extraction import persistence, promotion
from src.services.extraction.engineered.ner_validator import fill_ner_gaps
from src.services.extraction.engineered.table_extractor import extract_line_items
from src.services.extraction.judge_gate import run_grounded_judge_for_gaps
from src.services.extraction.parser import parse as parse_document
from src.services.extraction.pattern_extractor import run_pattern_extractor
from src.services.extraction.pattern_registry import get_registry
from src.services.extraction.persistence import Discrepancy
from src.services.extraction_v3.binding.invariants_runner import run_invariants

log = logging.getLogger(__name__)


# What we have learned about each reader, refreshed at most every 15 minutes. A rate
# computed over a 180-day window does not move minute to minute, and a DB round trip
# per document would otherwise be paid on every upload.
_ACCURACY_CACHE: dict = {}
_ACCURACY_CACHED_AT: float = 0.0
_ACCURACY_TTL_SECONDS = 900


def _cached_accuracy() -> dict:
    """Measured accuracy, refreshed at most every 15 minutes."""
    global _ACCURACY_CACHE, _ACCURACY_CACHED_AT
    import time
    from src.services.extraction_feedback.accuracy import load_accuracy
    now = time.monotonic()
    if now - _ACCURACY_CACHED_AT > _ACCURACY_TTL_SECONDS:
        _ACCURACY_CACHE = load_accuracy()
        _ACCURACY_CACHED_AT = now
    return _ACCURACY_CACHE


# What corrections have settled about each supplier's currency, refreshed at most every 15
# minutes — same rationale and TTL as _cached_accuracy above. default_for()'s learned-currency
# query is a full scan of bp_extraction_verdict; this block runs per document (any invoice
# with a bare "$"), so paying for that scan on every one of them would be the same mistake
# Task 4 already fixed once for reader accuracy.
_LEARNED_CCY_CACHE: dict = {}
_LEARNED_CCY_CACHED_AT: float = 0.0
_LEARNED_CCY_TTL_SECONDS = 900


def _cached_learned_currency(cur) -> dict:
    """learned_currency(), refreshed at most every 15 minutes."""
    global _LEARNED_CCY_CACHE, _LEARNED_CCY_CACHED_AT
    import time
    from src.services.extraction_feedback.supplier_currency import (
        learned_currency, _LOAD_SQL as _CCY_LOAD_SQL,
    )
    now = time.monotonic()
    if now - _LEARNED_CCY_CACHED_AT > _LEARNED_CCY_TTL_SECONDS:
        cur.execute(_CCY_LOAD_SQL)
        cols = [d[0] for d in (cur.description or [])]
        _LEARNED_CCY_CACHE = learned_currency([dict(zip(cols, r)) for r in cur.fetchall()])
        _LEARNED_CCY_CACHED_AT = now
    return _LEARNED_CCY_CACHE


def _resolve_bare_dollar_currency_hint(columns: dict) -> None:
    """Populate columns["supplier_default_currency"], the last-resort hint
    resolve_dollar_currency consults, from a supplier already on file.

    `columns` has no "supplier_id" at this point in the pipeline — supplier_name only
    resolves to a canonical supplier_id later, in promotion.promote() (supplier_resolver),
    after _raw is even written — so re-running that full resolver here (fuzzy match,
    auto-create) would duplicate its side effects (review-queue rows, possibly a new
    supplier row) before a doc_pk even exists. This instead recognises only a supplier
    already on file EXACTLY by name or by a human-confirmed alias: cheap, indexed, creates
    nothing.

    A name that matches more than one supplier row is deliberately treated as no match at
    all — bp_supplier has no uniqueness constraint on supplier_name/trading_name, so
    picking LIMIT 1 of an ambiguous pair would silently carry the WRONG company's currency
    through to columns["currency"]. Same principle this whole area already applies: when
    the evidence does not settle the question, fall through to "unresolved" and let the
    existing ambiguity discrepancy stop the document for a human, rather than guess.

    Non-fatal: any failure here (a name matching nothing, a name matching more than one
    supplier, or a DB error) just leaves the hint unset, exactly as if this function did
    not run — resolve_dollar_currency's other two resorts (an explicit code or country the
    document itself states) are unaffected either way.
    """
    try:
        from src.services.extraction_feedback.supplier_currency import default_for
        from src.services.db import get_conn as _sc_conn
        _sup_name = str(columns.get("supplier_name") or "").strip()
        if not _sup_name:
            return
        with _sc_conn() as _c:
            _cur = _c.cursor()
            _cur.execute(
                "SELECT supplier_id FROM proc.bp_supplier_alias "
                "WHERE LOWER(alias_name) = LOWER(%s) LIMIT 1",
                (_sup_name,),
            )
            _sup_row = _cur.fetchone()
            if not _sup_row:
                _cur.execute(
                    "SELECT supplier_id FROM proc.bp_supplier "
                    "WHERE LOWER(supplier_name) = LOWER(%s) "
                    "   OR LOWER(trading_name) = LOWER(%s) LIMIT 2",
                    (_sup_name, _sup_name),
                )
                _sup_rows = _cur.fetchall()
                # Exactly one match required — a name shared by two real suppliers must
                # NOT silently inherit either one's currency.
                _sup_row = _sup_rows[0] if len(_sup_rows) == 1 else None
            if _sup_row:
                columns["supplier_default_currency"] = default_for(
                    _cur, _sup_row[0], learned=_cached_learned_currency(_cur),
                )
    except Exception:
        log.exception("dispatch: supplier currency lookup failed (non-fatal)")


def _apply_currency_resolution(columns: dict, full_text: str, registry, discrepancies: list) -> None:
    """Resolve/validate columns["currency"] in place; append a blocking Discrepancy to
    ``discrepancies`` when it can't be settled.

    The number on a line is extracted whether or not a symbol sits beside it; the currency
    is the part that can be silently wrong, and a wrong currency rescales every figure
    derived from it. Neither rule below guesses: each either resolves the currency from
    something the document actually states, or raises a blocking discrepancy so the
    document waits on the Action page for a person instead of being promoted on a guess.

    Extracted out of dispatch_document as its own function so the guarantee below is
    directly testable without driving the full pipeline.

    supplier_default_currency is a hint for resolve_dollar_currency, not a column — the
    try/finally guarantees it is gone from ``columns`` before persistence on every exit
    from this function, including an exception raised anywhere inside it (Task 2's review
    found exactly this class of bug live: a stray key reaching an INSERT and taking down a
    whole promotion with "column ... does not exist").
    """
    from src.services.extraction import context_layer as _ccy_rules
    _ccy_col = "currency"
    try:
        if _ccy_col not in {f.db_column for f in registry.schema.fields}:
            return
        _conflict = _ccy_rules.currency_conflict(columns, full_text)
        if _conflict:
            discrepancies.append(Discrepancy(
                field_name=_ccy_col,
                issue_type="currency_ambiguous",
                severity="critical",
                blocks_promotion=True,
                raw_value=str(columns.get(_ccy_col) or ""),
                computed_value=",".join(_conflict["symbols"]),
                evidence_text=_conflict["evidence"] or None,
                notes=_conflict["detail"] + " — confirm which currency the totals are in.",
            ))
        elif "$" in (full_text or ""):
            # A bare "$" is not necessarily USD: the schema stores CAD, AUD, SGD, HKD and
            # NZD too, and every one of them prints the same symbol.
            _resolve_bare_dollar_currency_hint(columns)

            _resolved = _ccy_rules.resolve_dollar_currency(columns, full_text)
            if _resolved:
                _code, _why = _resolved
                if columns.get(_ccy_col) != _code:
                    log.info("dispatch: currency %r → %r (%s)",
                             columns.get(_ccy_col), _code, _why)
                    columns[_ccy_col] = _code
            elif not re.search(r"\b(?:GBP|EUR|JPY|INR|CHF)\b", full_text or ""):
                discrepancies.append(Discrepancy(
                    field_name=_ccy_col,
                    issue_type="currency_ambiguous",
                    severity="critical",
                    blocks_promotion=True,
                    raw_value=str(columns.get(_ccy_col) or ""),
                    computed_value=",".join(sorted(_ccy_rules.DOLLAR_CURRENCIES)),
                    notes=("the document prices in '$' but never says which dollar, and "
                           "nothing on it or in the supplier record settles it — confirm "
                           "the currency rather than assume USD."),
                ))
    finally:
        columns.pop("supplier_default_currency", None)


# Leading quote-identifier prefix ("QUT136586", "QUOTE-2025-051", ...). Stripped
# only when an identifier (something containing a digit) remains, so a quote
# referenced bare in a sibling PO doc and the same quote extracted from its own
# quote document resolve to one canonical PK (no duplicate _stg row).
_QUOTE_ID_PREFIX = re.compile(r"^(?:quotation|quote|qut|qte)[\s\-\.\/:#]*(?=\d)", re.I)

# Map each doc type → its per-line amount db_column for recovered line items.
# The line-index column (line_no / line_number) is injected by
# persistence.write_line_items_raw, so recovered rows must NOT include it.
_RECOVERED_LINE_AMOUNT_COL = {
    "invoice": "line_amount",
    "purchase_order": "line_total",
    "quote": "line_total",
}


def _map_recovered_lines(doc_type: str, recovered: list[dict]) -> list[dict[str, Any]]:
    """Map generic recovered items -> per-doc-type line-item db_column rows.
    The line-index column is intentionally OMITTED — persistence.write_line_items_raw
    injects it; including it would cause a duplicate-column INSERT."""
    amt_col = _RECOVERED_LINE_AMOUNT_COL[doc_type]
    mapped: list[dict[str, Any]] = []
    for it in recovered:
        row: dict[str, Any] = {
            "item_description": it["description"],
            amt_col: it["amount"],
        }
        if it.get("quantity") is not None:
            row["quantity"] = it["quantity"]
        if it.get("unit_price") is not None:
            row["unit_price"] = it["unit_price"]
        mapped.append(row)
    return mapped


def normalize_doc_pk(doc_type: str, value):
    """Canonicalize a document primary key to its persisted form.

    Quote IDs are stored without their QUT/Quote prefix; other doc types keep
    their PK verbatim. Returns the input unchanged when no normalization
    applies (including None and empty strings)."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return v
    if doc_type == "quote":
        return _QUOTE_ID_PREFIX.sub("", v)
    return v


def _pipeline_version() -> str:
    """Short git SHA + 'renov' suffix, e.g. 'e8c8563-renov'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "unknown"
    return f"{sha}-renov"


def _serialize_parsed(parsed) -> dict[str, Any]:
    """Build the parser_snapshot JSONB payload. Keep it tight — full_text,
    backend, confidence. Pages/tokens are heavy and not needed for re-grounding
    at the field level (full_text is enough)."""
    return {
        "source_path": getattr(parsed, "source_path", None),
        "file_format": getattr(parsed, "file_format", None),
        "parser_backend": getattr(parsed, "parser_backend", None),
        "parser_confidence": getattr(parsed, "parser_confidence", None),
        "full_text": parsed.full_text,
        "page_count": len(getattr(parsed, "pages", []) or []),
    }


def dispatch_document(
    *, process_monitor_id: int | None, file_path: str, doc_type: str,
) -> dict[str, Any]:
    """Run a single document through the renovation pipeline end-to-end.

    Returns a small result summary: {status, raw_id, doc_pk, n_fields,
    n_discrepancies, trace_id}.
    """
    trace_id = uuid4()
    pipeline_version = _pipeline_version()
    log.info("dispatch start trace=%s doc_type=%s file=%s",
             trace_id, doc_type, file_path)

    # L0
    parsed = parse_document(file_path)

    # get_registry(doc_type) is a process-wide singleton — run_pattern_extractor (L1,
    # next) resolves this SAME instance internally and reads registry.patterns_for(field)
    # to decide which reader wins per field. Learning must be applied here, BEFORE that
    # first read, or the demotion has no effect on the document in front of us: it would
    # only take hold for the *next* document processed after this one. A learning-lookup
    # failure must never fail extraction — fall back to the static priors.
    registry = get_registry(doc_type)
    try:
        _n = registry.apply_observed(_cached_accuracy())
        if _n:
            log.info("dispatch: %d pattern prior(s) replaced by measured accuracy", _n)
    except Exception:
        log.exception("dispatch: could not apply learned accuracy (using static priors)")

    # L1
    candidates = run_pattern_extractor(parsed, doc_type)

    # L2 — engineered fallbacks for NER-typed fields the L1 regex missed.
    # Only fires for fields with judge.ner_type_check != 'none'.
    l1_fields = {c.field for c in candidates}
    try:
        ner_candidates = fill_ner_gaps(
            parsed=parsed,
            schema=registry.schema,
            existing_fields=l1_fields,
        )
        candidates.extend(ner_candidates)
    except Exception as exc:
        log.warning("NER gap-fill failed: %s", exc)

    # L2 — line item extraction from ParsedDocument.tables
    try:
        line_candidates = extract_line_items(parsed, registry.schema)
        candidates.extend(line_candidates)
    except Exception as exc:
        log.warning("table_extractor failed: %s", exc)

    # L3 — grounded last-resort judge fills required fields that L1+L2 missed.
    # Each judge candidate's evidence_text is verified to be a verbatim
    # substring of parsed.full_text by call_grounded_last_resort itself; the
    # substring gate below re-validates as defence in depth.
    try:
        judge_candidates = run_grounded_judge_for_gaps(
            parsed=parsed,
            registry=registry,
            existing_candidates=candidates,
            file_path=getattr(parsed, "source_path", None),
        )
        candidates.extend(judge_candidates)
    except Exception as exc:
        log.warning("grounded judge gate failed: %s", exc)

    # L3 grounding gate — defence in depth. L1 candidates come from regex
    # over full_text (always substrings); L2 NER spans come from token
    # iteration over the parse (also substrings); L3 judge candidates have
    # already been through call_grounded_last_resort's safety check which
    # uses the same progressive normalisation as below. We replicate the
    # check here so any future candidate source has to honour it too.
    import html as _html
    from src.services.extraction_v3.judge.grounded_last_resort import (
        _collapse_letter_spacing, _norm_ws, _WS_RE,
    )
    full_text = parsed.full_text or ""
    # Docling exports markdown pipe-tables — and escapes any `|` characters
    # inside cell text as `&#124;`. The cell.text we recover from the
    # structural table preserves the literal `|`. To keep these two views
    # comparable for the grounding gate, decode HTML entities on the
    # full_text side before substring checks.
    full_text_dec = _html.unescape(full_text)
    full_text_ws = _norm_ws(full_text_dec)
    full_text_sq = _WS_RE.sub("", _collapse_letter_spacing(full_text_dec))

    def _grounded(c) -> bool:
        t = c.span.text
        if not t:
            return False
        t_dec = _html.unescape(t)
        if t_dec in full_text_dec:
            return True
        if _norm_ws(t_dec) in full_text_ws:
            return True
        # Letter-spacing collapse + all-whitespace strip — last resort, only
        # accepts evidence whose non-whitespace tokens all appear in the doc.
        if _WS_RE.sub("", _collapse_letter_spacing(t_dec)) in full_text_sq:
            return True
        return False

    grounded = [c for c in candidates if _grounded(c)]
    record_action(
        phase=PHASE_EXTRACTION,
        action_type="grounding_gate",
        doc_type=doc_type,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        agent="grounding_gate",
        status="ok" if grounded else "warn",
        summary=f"{len(grounded)} of {len(candidates)} candidates grounded",
        details={"candidates": len(candidates), "grounded": len(grounded)},
    )

    # Build header record + provenance picks + line items
    columns, picked, bind_errors = persistence.build_header_record(grounded, registry)
    line_items = persistence.build_line_items(grounded, registry)

    # Context understanding layer — single Ollama (BeyondProcwise/AgentNick)
    # pass over the full document text. The fine-tuned procurement model
    # uses the L1/L2/L3 candidates as HINTS but treats full_text as ground
    # truth, disambiguating supplier vs buyer, dropping layout noise, and
    # producing typed/grounded values. Its output REPLACES the regex/judge
    # picks for every schema field — None means "could not be grounded in
    # the document" (no fabrication), which translates to a missing-required
    # discrepancy below if the field is required.
    full_text = parsed.full_text or ""
    if full_text.strip():
        try:
            from src.services.extraction.context_layer import synthesize as _ctx_synth
            synthesized = _ctx_synth(
                doc_type, full_text, columns, file_path=file_path,
            )
            # Build the schema's set of valid db_columns. context_layer's
            # output may include derived fields (exchange_rate_to_usd,
            # converted_amount_usd) that exist on the invoice/PO _raw
            # tables but not the quote _raw table; filtering on the schema
            # keeps the INSERT statement honest.
            # Also include `field.name` for fields with resolves_to_db_column
            # set (e.g. invoice.supplier_name → resolved to supplier_id by
            # supplier_resolver during promote()): the field has db_column=null
            # but the _raw table DOES have a column by that name.
            valid_cols: set[str] = set()
            for f in registry.schema.fields:
                if f.db_column:
                    valid_cols.add(f.db_column)
                if getattr(f, "resolves_to_db_column", None):
                    valid_cols.add(f.name)
            for k, v in synthesized.items():
                if k not in valid_cols:
                    continue
                if v is not None:
                    columns[k] = v
                else:
                    # context_layer says "couldn't ground this" — drop any
                    # regex noise we had. Required fields will surface as
                    # missing_required below; non-required stay NULL.
                    columns.pop(k, None)
            record_action(
                phase=PHASE_EXTRACTION,
                action_type="context_synthesize",
                doc_type=doc_type,
                trace_id=trace_id,
                pipeline_version=pipeline_version,
                agent="AgentNick",
                status="ok",
                summary=f"context_layer synthesized {len(synthesized)} fields",
                details={"fields": sorted(synthesized.keys())},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("context_layer synthesis failed: %s", exc)

    # Canonicalize the primary key before it is used for _raw, provenance and
    # promotion (e.g. quote 'QUT136586' → '136586') so the same quote referenced
    # across documents resolves to a single _stg row.
    _pk_field = persistence._DOC_PK_FIELD[doc_type]
    if columns.get(_pk_field) not in (None, ""):
        columns[_pk_field] = normalize_doc_pk(doc_type, columns[_pk_field])

    # --- Bounded gap-control recovery (one attempt) ---
    # If the structural/table extractor produced no lines or lines that don't
    # reconcile to the header subtotal, ask AgentNick to re-enumerate line
    # items from full_text. Accept the recovered set only when it reconciles
    # (or sums closer to the header total than what we had) — never fabricate.
    has_line_schema = bool(
        registry.schema.line_items and registry.schema.line_items.fields
    )
    pre = _completeness.assess(
        doc_type, columns, line_items, has_line_schema=has_line_schema,
    )
    if has_line_schema and full_text.strip() and pre.status in (
        "no_line_items", "line_sum_mismatch",
    ):
        try:
            header_total = _completeness.header_subtotal(doc_type, columns)
            from src.services.extraction import context_layer as _ctx
            recovered = _ctx.synthesize_line_items(doc_type, full_text, header_total)
        except Exception as exc:  # noqa: BLE001
            log.warning("line-item recovery failed: %s", exc)
            recovered = []

        if recovered:
            mapped = _map_recovered_lines(doc_type, recovered)
            header_total = _completeness.header_subtotal(doc_type, columns)
            old_sum = _completeness.line_sum(doc_type, line_items)
            new_sum = _completeness.line_sum(doc_type, mapped)
            accept = False
            if header_total:  # truthy float; 0.0 subtotal treated as unverifiable (see completeness._reconciles)
                old_err = abs((old_sum or 0) - header_total)
                new_err = abs((new_sum or 0) - header_total)
                accept = new_err < old_err
            elif not line_items:
                accept = True  # had nothing; grounded recovery is strictly better
            if accept:
                log.info(
                    "line-item recovery: replaced %d lines with %d (sum %.2f->%.2f, header=%s)",
                    len(line_items), len(mapped), old_sum or 0, new_sum or 0, header_total,
                )
                line_items = mapped

    # Subtotal recovery from line items. When the required header subtotal
    # (invoice_amount / total_amount) could not be grounded from the text but
    # line items are present, derive it from the lines — subtotal-closure aware,
    # so a mis-captured Subtotal/Tax row (table parsers emit them as extra
    # "line items") is not double-counted. Pure arithmetic over grounded line
    # amounts, never fabrication; closes the garbled-summary-label gap that left
    # invoice_amount NULL and blocked promotion as missing_required.
    _sub_col = _completeness._SUBTOTAL_COL.get(doc_type)
    if _sub_col and columns.get(_sub_col) in (None, "") and line_items:
        _sub, _cut = _completeness.derive_subtotal_from_lines(doc_type, line_items)
        if _sub is not None:
            columns[_sub_col] = _sub
            if _cut is not None and 0 < _cut < len(line_items):
                log.info(
                    "dispatch: derived %s=%s and trimmed %d summary line(s) "
                    "after subtotal-closure",
                    _sub_col, _sub, len(line_items) - _cut,
                )
                line_items = line_items[:_cut]
            else:
                log.info("dispatch: derived %s=%s from %d line items",
                         _sub_col, _sub, len(line_items))

    discrepancies: list[Discrepancy] = []
    # Re-bind any synthesized values that conflict with column types is
    # already handled by context_layer's _validate_and_bind. Bind errors
    # from the L1 layer are stale once synthesis ran — drop them.
    if not full_text.strip():
        discrepancies.extend(bind_errors)

    # Missing required fields (post-synthesis). columns now reflects the
    # context layer's truth; missing column == AgentNick could not find it.
    for f in registry.schema.fields:
        if f.required and f.db_column and (
            f.db_column not in columns or columns.get(f.db_column) in (None, "")
        ):
            discrepancies.append(Discrepancy(
                field_name=f.name,
                issue_type="missing_required",
                severity="critical",
                blocks_promotion=True,
                notes="context_layer (AgentNick) could not ground a value in the document",
            ))

    # Which currency is this money in?
    #
    # The number on a line is extracted whether or not a symbol sits beside it; the currency
    # is the part that can be silently wrong, and a wrong currency rescales every figure
    # derived from it. Neither rule below guesses: each either resolves the currency from
    # something the document actually states, or raises a blocking discrepancy so the
    # document waits on the Action page for a person instead of being promoted on a guess.
    _apply_currency_resolution(columns, full_text, registry, discrepancies)

    # Invariants
    try:
        invariant_results = run_invariants(
            header=columns, line_items=[], schema=registry.schema,
        )
    except Exception as exc:
        log.warning("invariants_runner failed: %s", exc)
        invariant_results = []

    for ir in invariant_results:
        if ir.severity == "CRITICAL":
            discrepancies.append(Discrepancy(
                field_name=ir.name,  # the invariant name; fields_involved not exposed
                issue_type="invariant_failed",
                severity="critical",
                blocks_promotion=True,
                notes=ir.message or f"{ir.name} failed",
            ))
        elif ir.severity == "WARNING":
            discrepancies.append(Discrepancy(
                field_name=ir.name,
                issue_type="invariant_failed",
                severity="warning",
                blocks_promotion=False,
                notes=ir.message or f"{ir.name} warning",
            ))

    # Line-item warnings: surface silent failures into the discrepancy
    # queue so HITL can see them. Non-blocking — these don't stop
    # promotion (we still want the header data in _stg).
    if registry.schema.line_items and registry.schema.line_items.fields:
        if not line_items:
            discrepancies.append(Discrepancy(
                field_name="line_items",
                issue_type="missing_line_items",
                severity="warning",
                blocks_promotion=False,
                notes=(
                    "no line items were extracted from this document — "
                    "table_extractor + text-fallback both returned 0 rows"
                ),
            ))
        else:
            # Flag lines where the numeric essentials are all NULL.
            numeric_keys = {
                "quantity", "unit_price", "line_amount", "line_total",
                "total_amount", "total_amount_incl_tax",
            }
            from src.services.extraction.three_way_match import is_non_charge_line
            for li_idx, li in enumerate(line_items):
                if not any(li.get(k) not in (None, "", 0) for k in numeric_keys):
                    # Terms/footer furniture legitimately has no numbers —
                    # warning on it buried the real findings 201-deep.
                    if is_non_charge_line(li.get("item_description")):
                        continue
                    discrepancies.append(Discrepancy(
                        field_name=f"line_items[{li_idx}]",
                        issue_type="line_missing_numbers",
                        severity="warning",
                        blocks_promotion=False,
                        notes=(
                            f"line {li_idx + 1} has no quantity / unit_price / "
                            f"line_amount — only the description was captured"
                        ),
                    ))

            # A line that HAS quantity/unit_price but NO amount is the dangerous
            # case: the check above passes (some numerics are present) and the
            # qty×unit reconciliation below skips on a NULL amount, so the line
            # sails through with its money missing. That is exactly how
            # Invoice_INV618706 booked its UNIT PRICE (£584.79) as the invoice
            # total instead of £1,169.58, silently, at 94% confidence.
            _amt_col_missing = _RECOVERED_LINE_AMOUNT_COL.get(doc_type, "line_amount")
            for li_idx, li in enumerate(line_items):
                has_qty_or_price = any(
                    li.get(k) not in (None, "", 0) for k in ("quantity", "unit_price")
                )
                if has_qty_or_price and li.get(_amt_col_missing) in (None, ""):
                    discrepancies.append(Discrepancy(
                        field_name=f"line_items[{li_idx}].{_amt_col_missing}",
                        issue_type="line_missing_amount",
                        severity="warning",
                        blocks_promotion=False,
                        notes=(
                            f"line {li_idx + 1} printed a quantity/unit price but its "
                            f"{_amt_col_missing} was not captured — the line's money is "
                            f"missing, so header totals derived from these lines are unsafe"
                        ),
                    ))

            # Source-data reconciliation: when a line prints quantity,
            # unit_price AND a line total that don't agree (qty × unit_price
            # ≠ line total), the DOCUMENT itself is internally inconsistent.
            # Keep the printed values verbatim (no fabrication) and flag the
            # mismatch as a discrepancy so the source error surfaces.
            _amt_col = _RECOVERED_LINE_AMOUNT_COL.get(doc_type, "line_amount")
            for li_idx, li in enumerate(line_items):
                q = _completeness._to_float(li.get("quantity"))
                up = _completeness._to_float(li.get("unit_price"))
                amt = _completeness._to_float(li.get(_amt_col))
                if q and up and amt is not None and q > 0 and up > 0:
                    computed = round(q * up, 2)
                    if abs(computed - amt) > max(0.02, 0.01 * computed):
                        discrepancies.append(Discrepancy(
                            field_name=f"line_items[{li_idx}].{_amt_col}",
                            issue_type="line_total_mismatch",
                            severity="warning",
                            blocks_promotion=False,
                            raw_value=str(amt),
                            computed_value=str(computed),
                            notes=(
                                f"line {li_idx + 1}: quantity {q:g} × unit_price {up:g} "
                                f"= {computed} but line total printed as {amt} "
                                f"(source-data inconsistency)"
                            ),
                        ))

    # The completeness verdict must become a discrepancy, not just a log line.
    # It was previously computed AFTER the discrepancy list was written (see the
    # `completeness_status` assignment further down), so a document whose line
    # items don't reconcile to its own header subtotal promoted with
    # n_discrepancies=0. Invoice_INV618706 promoted at 94.44% confidence with
    # status 'line_sum_mismatch' and nothing in the queue — the 2x money error
    # was detected and then dropped on the floor.
    if has_line_schema and line_items:
        _post = _completeness.assess(
            doc_type, columns, line_items, has_line_schema=has_line_schema,
        )
        if _post.status == "line_sum_mismatch":
            _hdr = _completeness.header_subtotal(doc_type, columns)
            _lsum = _completeness.line_sum(doc_type, line_items)
            discrepancies.append(Discrepancy(
                field_name="line_items",
                issue_type="line_sum_mismatch",
                severity="warning",
                blocks_promotion=False,
                raw_value=str(_lsum) if _lsum is not None else None,
                expected_value=str(_hdr) if _hdr is not None else None,
                notes=(
                    f"line items sum to {_lsum} but the document's header subtotal is "
                    f"{_hdr} — the lines and the header disagree. Values kept verbatim; "
                    f"flagged for review."
                ),
            ))

    # Three-way match: what does this document say that its purchase order does not?
    #
    # Everything above is the document arguing with ITSELF (its lines don't sum to its
    # header, a required field is missing). None of it is what a buyer actually needs,
    # which is the document arguing with the PO: a line that was never ordered, a price
    # above the one agreed, a PO number that does not exist. That comparison was being
    # computed in linking_engine, reduced to a confidence score, and discarded.
    try:
        from src.services.extraction.three_way_match import check_against_po

        po_findings = check_against_po(doc_type, columns, line_items)
        if po_findings:
            log.info(
                "three-way match: %d finding(s) against the purchase order (%s)",
                len(po_findings),
                ", ".join(sorted({f.issue_type for f in po_findings})),
            )
        discrepancies.extend(po_findings)
    except Exception:  # noqa: BLE001 — a match failure must not lose the extraction
        log.exception("three-way match failed; extraction stands, no PO findings raised")

    blocking = any(d.blocks_promotion for d in discrepancies)
    promotion_status = "discrepancy" if blocking else "pending"

    # `persistence.write_raw` derives doc_pk_candidate from columns[pk_field]
    # internally — since context_layer now writes the authoritative
    # identifier into that schema column (e.g. columns["invoice_id"]),
    # nothing extra is needed here.
    #
    # Freeze who produced each column value NOW, while `candidates` is still in
    # memory, and thread it through parser_snapshot. promotion.promote() — the one
    # funnel every promotion path (this inline call, HITL-triggered promotion, the
    # pending catch-up sweep) goes through — runs long after this and only ever
    # sees the persisted _raw row, never the original Candidate objects. Without
    # this, promote() would have nothing to attribute a value to and every field
    # on every path would fall back to "context_layer" regardless of what actually
    # produced it.
    _parser_snapshot = _serialize_parsed(parsed)
    try:
        from src.services.extraction import provenance as _prov
        _parser_snapshot["_field_provenance"] = _prov.snapshot(columns, candidates)
    except Exception:
        log.exception("dispatch: provenance snapshot failed (non-fatal)")

    raw_id = persistence.write_raw(
        doc_type=doc_type,
        file_path=file_path,
        process_monitor_id=process_monitor_id,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        columns=columns,
        parser_snapshot=_parser_snapshot,
        promotion_status=promotion_status,
    )
    record_action(
        phase=PHASE_EXTRACTION,
        action_type="persist",
        doc_type=doc_type,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        agent="persistence",
        status="ok",
        summary=f"persisted raw_id={raw_id}",
        details={"raw_id": raw_id, "n_fields": len(columns), "n_lines": len(line_items)},
    )

    # Write line items (if any)
    if line_items:
        try:
            persistence.write_line_items_raw(
                doc_type=doc_type, raw_id=raw_id, line_items=line_items,
            )
        except Exception as exc:
            log.warning("line_items_raw write failed: %s", exc)

    if discrepancies:
        persistence.write_discrepancies(
            doc_type=doc_type, raw_id=raw_id,
            source_file=file_path,
            doc_pk_candidate=columns.get(persistence._DOC_PK_FIELD[doc_type]),
            discrepancies=discrepancies,
        )

    # Provenance (only when we have a doc_pk — required by the table's NOT NULL)
    persistence.write_provenance(
        doc_type=doc_type,
        doc_pk=columns.get(persistence._DOC_PK_FIELD[doc_type]),
        pipeline_version=pipeline_version,
        picked=picked,
        registry=registry,
    )

    doc_pk = columns.get(persistence._DOC_PK_FIELD[doc_type])

    # Auto-promote when clean (no blocking discrepancy) and we have a doc_pk.
    # Discrepancy rows wait for HITL → trigger → listener.
    #
    # NOTE: field-provenance recording lives inside promotion.promote() itself,
    # not here — it is the one seam every promotion path (this inline call, the
    # HITL NOTIFY listener's apply_hitl_fixes_and_promote, and the promote_pending
    # catch-up sweep) runs through, so writing it there is the only way a
    # HITL-resolved document ends up with provenance too.
    final_status = promotion_status
    if not blocking and doc_pk:
        prom = promotion.promote(raw_id, doc_type)
        if prom.get("ok"):
            final_status = "promoted"
        else:
            log.warning("inline promote failed: %s", prom.get("reason"))
            final_status = "pending"  # _raw kept; manual retry possible
    elif not blocking and not doc_pk:
        # Missing PK is a soft failure — keep the _raw row for inspection,
        # mark status='no_pk' so it doesn't sit in 'pending' indefinitely.
        persistence.update_promotion_status(
            doc_type=doc_type, raw_id=raw_id, status="no_pk",
        )
        final_status = "no_pk"

    missing_required_fields = sorted({
        d.field_name for d in discrepancies
        if d.issue_type == "missing_required" and d.blocks_promotion
    })
    completeness_status = _completeness.assess(
        doc_type, columns, line_items,
        has_line_schema=has_line_schema,
        missing_required=missing_required_fields,
    ).status
    # Surface the post-promotion confidence (computed by promote()'s
    # _compute_confidence_score) so the watcher's training-data collector
    # can decide whether this extraction is good enough to record as a
    # gold-standard example. Without this, the watcher's confidence>=0.90
    # gate is always against 0 and no example is ever recorded.
    confidence_score: float = 0.0
    if final_status == "promoted" and doc_pk:
        try:
            cs = _read_persisted_confidence(doc_type, doc_pk)
            if cs is not None:
                confidence_score = float(cs)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not read persisted confidence: %s", exc)

    # _source_text feeds the training-example collector. Cap so the JSONL
    # line stays a few KB — the model only needs the structural shape of
    # the source, not the whole multi-page text.
    source_text = (parsed.full_text or "")[:6000]
    result = {
        "status": final_status,
        "raw_id": raw_id,
        "doc_pk": doc_pk,
        "pk": doc_pk,  # alias for the watcher's legacy reader
        "doc_type": doc_type,
        "n_fields": len(columns),
        # Watcher reads this key (legacy v3 shape) to decide whether to log
        # the ZERO_LINE_ITEMS warning. Surface the count we actually wrote
        # so the warning fires only when extraction genuinely missed lines,
        # not when this newer pipeline simply doesn't include the key.
        "line_items": len(line_items),
        "header_fields": len(columns),
        "n_discrepancies": len(discrepancies),
        "discrepancies": len(discrepancies),
        "missing_required": missing_required_fields,
        "completeness_status": completeness_status,
        "trace_id": str(trace_id),
        "pipeline_version": pipeline_version,
        "raw_persisted": True,
        # New keys for the training-data collector. Defaults are safe
        # (empty / zero) so non-collector callers are unaffected.
        "_source_text": source_text,
        "confidence": confidence_score / 100.0 if confidence_score else 0.0,
        "confidence_score": confidence_score,
        "errors": 0 if final_status in ("promoted", "pending") else 1,
    }
    # Avoid logging the full source_text — keep the log line tidy.
    log_result = {k: v for k, v in result.items() if k != "_source_text"}
    log.info("dispatch end %s", log_result)
    return result


def _read_persisted_confidence(doc_type: str, doc_pk: str) -> float | None:
    """Read confidence_score from the freshly-promoted _stg row."""
    from src.services.db import get_conn

    stg_table_map = {
        "invoice": ("proc.bp_invoice_stg", "invoice_id"),
        "purchase_order": ("proc.bp_purchase_order_stg", "po_id"),
        "quote": ("proc.bp_quote_stg", "quote_id"),
    }
    info = stg_table_map.get(doc_type)
    if not info:
        return None
    stg_table, pk_col = info
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT confidence_score FROM {stg_table} WHERE {pk_col} = %s",
            (doc_pk,),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None
