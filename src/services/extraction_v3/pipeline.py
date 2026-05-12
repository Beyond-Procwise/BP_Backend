"""Pipeline V3 orchestrator. Sequence:
   1. Layer 1: parse the doc → ParsedDocument
   2. Layer 2: single Qwen-VL call (qwen_vlm extractor) → candidates
   3. Layer 3: type-bind, run invariants, judge-orchestrate (coherence only), assemble result.

   The 6-extractor parallel stack has been replaced by one Qwen2.5-VL-7B-Instruct
   call per document. There are no disagreements to break — the VLM is the sole
   extraction source. The judge orchestrator still runs schema-coherence as a
   final cross-field validation pass.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.result import (
    ExtractionResult, CommittedField, ResidualField, JudgeAction, ResidualReason,
)
from src.services.extraction_v3.parsers.router import parse as parse_document
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema, DocSchema
from src.services.extraction_v3.binding.type_binder import bind_typed
from src.services.extraction_v3.binding.invariants_runner import run_invariants
from src.services.extraction_v3.judge.orchestrator import run_judge_orchestrator
from src.services.extraction_v3.judge.contracts import InvariantResultSummary
# Register the VLM extractor in the registry
import src.services.extraction_v3.extractors.vlm  # noqa

log = logging.getLogger(__name__)

PIPELINE_VERSION = "v3.2.1"
HEADER_LINE_ITEM_PREFIX = "line_items["


def _group_candidates_by_field(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for c in candidates:
        out.setdefault(c.field, []).append(c)
    return out


class PipelineV3:

    def run(self, doc_path: str | Path, doc_type: str) -> ExtractionResult:
        path = Path(doc_path)
        log.info("Pipeline V3: parsing %s as %s", path, doc_type)

        # 1. Parse (Layer 1)
        parsed: ParsedDocument = parse_document(path)
        schema: DocSchema = load_doc_schema(doc_type)

        # 2. Single Qwen-VL extraction call (Layer 2)
        # One model call per document — no parallel extractor stack needed.
        from src.services.extraction_v3.extractors.vlm import extract_with_vlm
        all_candidates: list[Candidate] = extract_with_vlm(parsed, schema, source_file=str(path))
        log.info("Pipeline V3: %d candidates from qwen_vlm for %s", len(all_candidates), path.name)

        # Split header candidates from line-item candidates
        header_cands = [c for c in all_candidates if not c.field.startswith(HEADER_LINE_ITEM_PREFIX)]
        line_cands = [c for c in all_candidates if c.field.startswith(HEADER_LINE_ITEM_PREFIX)]

        # 3a. Header candidates → judge orchestrator
        # With a single VLM source there are no disagreements, so the orchestrator
        # mostly passes through candidates and runs schema-coherence as a final pass.
        header_by_field = _group_candidates_by_field(header_cands)

        # 3b. Run invariants on a preliminary record (just the candidate values).
        # We do this BEFORE the judge so the coherence pass can see the invariants.
        prelim_record = {
            fname: cands[0].value
            for fname, cands in header_by_field.items() if cands
        }
        prelim_lines = self._build_prelim_lines(line_cands)
        invariant_results = run_invariants(prelim_record, prelim_lines, schema)
        invariant_summaries = [
            InvariantResultSummary(name=r.name, passed=(r.severity in ("info", "INFO", "PASS", "ok")),
                                    severity=r.severity, message=r.message)
            for r in invariant_results
        ]

        # 3c. Run judge orchestrator on header
        orch = run_judge_orchestrator(
            header_by_field,
            schema,
            parsed.full_text,
            invariant_summaries,
            file_path=str(path),
        )

        # 4. Build the ExtractionResult
        committed: list[CommittedField] = []
        residuals: list[ResidualField] = []
        for fname, outcome in orch.field_outcomes.items():
            if outcome.chosen is not None:
                # Type-bind the chosen value
                fspec = next(f for f in schema.fields if f.name == fname)
                bound = bind_typed(outcome.chosen, fspec.type)
                final_conf = outcome.chosen.confidence
                # Demote on coherence='incoherent'
                if orch.coherence is not None and orch.coherence.verdict == "incoherent":
                    final_conf = max(0.0, final_conf - 0.20)
                if bound.bind_error:
                    # Type-coercion failure → residual
                    residuals.append(ResidualField(
                        field_path=fname, reason="bind_error_no_resolution",
                        candidates=outcome.candidates_seen,
                    ))
                else:
                    committed.append(CommittedField(
                        field_path=fname,
                        value=str(bound.coerced_value),
                        page=outcome.chosen.page,
                        bbox=outcome.chosen.bbox,
                        evidence_text=outcome.chosen.evidence_text,
                        model=outcome.chosen.model,
                        model_confidence=outcome.chosen.confidence,
                        judge_actions=outcome.judge_actions,
                        final_confidence=final_conf,
                    ))
            elif outcome.residual_reason is not None:
                residuals.append(ResidualField(
                    field_path=fname, reason=outcome.residual_reason,
                    candidates=outcome.candidates_seen,
                ))

        # 4b.5. Invoice-id label-word rejection: if the extracted invoice_id is a
        # common label word (pure alphabetic, ≤2 words, e.g. "Document", "Invoice",
        # "Number", "Reference") it is an OCR artefact where the anchor word was
        # captured instead of the actual identifier. Demote it to residual.
        _ID_LABEL_WORDS = frozenset({
            "document", "invoice", "number", "reference", "order", "id",
            "ref", "no", "none", "null", "n/a", "na", "date", "voucher",
        })
        cleaned_committed_id: list[CommittedField] = []
        for cf in committed:
            if cf.field_path == "invoice_id":
                val_lower = cf.value.strip().lower()
                words = val_lower.split()
                if len(words) <= 2 and all(w in _ID_LABEL_WORDS for w in words):
                    log.debug(
                        "Rejecting invoice_id=%r — looks like a label word, not an ID",
                        cf.value,
                    )
                    residuals.append(ResidualField(
                        field_path="invoice_id",
                        reason="bind_error_no_resolution",
                        candidates=[],
                    ))
                    continue
            cleaned_committed_id.append(cf)
        committed = cleaned_committed_id

        # 4b.6. Invoice-id regex recovery: if invoice_id was rejected or is missing,
        # try a direct regex scan over the raw text for label:value patterns covering
        # the canonical invoice-id labels. This catches OCR layouts where the label
        # and value are on adjacent lines ("Document\nNumber: 0526") so the standard
        # anchor extractors mis-capture the label word.
        committed_field_paths = {cf.field_path for cf in committed}
        if "invoice_id" not in committed_field_paths:
            _INV_ID_RE = re.compile(
                r"(?:Invoice\s+(?:Number|No\.?|#)|Document\s+Number|"
                r"Document\s+No\.?|Inv\.?\s+No\.?|Reference\s+No\.?)"
                r"[\s:]+([#A-Za-z0-9][A-Za-z0-9\-/\.\#]{1,30})",
                re.IGNORECASE,
            )
            m = _INV_ID_RE.search(parsed.full_text)
            if m:
                inv_id_val = m.group(1).strip().rstrip(".")
                evidence = m.group(0)
                log.debug(
                    "Invoice-id regex recovery: found %r → %r",
                    evidence, inv_id_val,
                )
                # Remove any existing residual for invoice_id
                residuals = [rf for rf in residuals if rf.field_path != "invoice_id"]
                committed.append(CommittedField(
                    field_path="invoice_id",
                    value=inv_id_val,
                    page=1,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    evidence_text=evidence,
                    model="pipeline_recovery",
                    model_confidence=0.70,
                    judge_actions=[],
                    final_confidence=0.70,
                ))

        # 4b.7. Invoice-date regex recovery: if invoice_date is missing, scan the text
        # for a date pattern near a DATE label. This catches documents where Qwen
        # reformats the date (e.g. "10/20/2025" → "2025-10-20" ISO) so the substring
        # check rejects the normalised value. We extract the raw date string so it IS
        # a literal substring of the document text.
        if schema.doc_type == "invoice":
            committed_field_paths = {cf.field_path for cf in committed}
            if "invoice_date" not in committed_field_paths:
                # Match: DATE<sep><date> or Invoice Date<sep><date>
                _DATE_LABEL_RE = re.compile(
                    r"(?:Invoice\s*Date|Date|Dated|Issue\s*Date|Billing\s*Date)"
                    r"[\s:–-]*"
                    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
                    r"|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"
                    r"|\d{1,2}[-/](?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
                    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-/]\d{2,4}"
                    r"|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
                    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}"
                    r"|\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})",
                    re.IGNORECASE,
                )
                m = _DATE_LABEL_RE.search(parsed.full_text)
                if m is None:
                    # Fallback: OCR-fuzzy date pattern — captures DD <3-letter-word>, YYYY
                    # where the month token may be OCR-corrupted (e.g. "Jon" for "Jan",
                    # "Jol" for "Jul"). Anchored to a date label so false positives are rare.
                    _DATE_FUZZY_RE = re.compile(
                        r"(?:Invoice\s*Date|Date|Dated|Issue\s*Date|Billing\s*Date)"
                        r"[\s:–-]*"
                        r"(\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4})",
                        re.IGNORECASE,
                    )
                    m = _DATE_FUZZY_RE.search(parsed.full_text)
                    if m:
                        log.info(
                            "Invoice-date fuzzy OCR recovery: found %r for %s",
                            m.group(0), path.name,
                        )
                if m:
                    date_raw = m.group(1).strip()
                    evidence = m.group(0)
                    log.info(
                        "Invoice-date regex recovery: found %r → %r for %s",
                        evidence, date_raw, path.name,
                    )
                    residuals = [rf for rf in residuals if rf.field_path != "invoice_date"]
                    committed.append(CommittedField(
                        field_path="invoice_date",
                        value=date_raw,
                        page=1,
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        evidence_text=evidence,
                        model="pipeline_recovery",
                        model_confidence=0.72,
                        judge_actions=[],
                        final_confidence=0.72,
                    ))

        # 4b.8. Invoice-amount regex recovery: if invoice_amount is missing, scan for
        # a Subtotal or Net Amount label followed by a monetary value in the text.
        # Extracts the raw substring so the value passes the substring guarantee.
        if schema.doc_type == "invoice":
            committed_field_paths = {cf.field_path for cf in committed}
            if "invoice_amount" not in committed_field_paths:
                _AMT_LABEL_RE = re.compile(
                    r"(?:Subtotal|Sub-total|Net\s*Amount|Amount\s*(?:Before\s*Tax|Excl\.?\s*Tax|Excl\.?\s*VAT|Net))"
                    r"[\s:|\|]*([£€\$¥₹]?\s*[\d,\.]+(?:\s*[£€\$¥₹])?)",
                    re.IGNORECASE,
                )
                m = _AMT_LABEL_RE.search(parsed.full_text)
                _amt_raw_override: str | None = None
                _amt_evidence_override: str | None = None
                if m is None:
                    # Fallback: three consecutive currency amounts on separate lines
                    # (subtotal / tax / grand-total pattern, as seen in DOCX python-docx extras).
                    # We take the FIRST amount as the subtotal (invoice_amount).
                    _AMT_TRIPLET_RE = re.compile(
                        r"([£€\$¥₹][\d,\.]+)\n([£€\$¥₹][\d,\.]+)\n([£€\$¥₹][\d,\.]+)"
                    )
                    mt = _AMT_TRIPLET_RE.search(parsed.full_text)
                    if mt:
                        log.info(
                            "Invoice-amount triplet recovery (no label): %r / %r / %r for %s",
                            mt.group(1), mt.group(2), mt.group(3), path.name,
                        )
                        _amt_raw_override = mt.group(1)  # first = subtotal
                        _amt_evidence_override = mt.group(1)
                if m or _amt_raw_override:
                    amt_raw = (_amt_raw_override if _amt_raw_override is not None else m.group(1)).strip()
                    # Strip leading/trailing currency symbols, keep numeric+comma+period
                    amt_numeric = re.sub(r"^[£€\$¥₹\s]+|[£€\$¥₹\s]+$", "", amt_raw).strip()
                    evidence = _amt_evidence_override if _amt_evidence_override is not None else m.group(0)
                    if amt_numeric and any(ch.isdigit() for ch in amt_numeric):
                        # Normalise through parse_amount so the DB gets a plain decimal
                        # ("129,200" → "129200.0"). The evidence_text keeps the raw match
                        # so provenance is traceable.
                        try:
                            from src.services.extraction_v2.parsers.amounts import parse_amount as _parse_amt
                            _parsed_amt = _parse_amt(amt_numeric)
                            if _parsed_amt is not None:
                                amt_numeric = str(_parsed_amt)
                        except Exception:
                            pass
                        log.info(
                            "Invoice-amount regex recovery: found %r → %r for %s",
                            evidence, amt_numeric, path.name,
                        )
                        residuals = [rf for rf in residuals if rf.field_path != "invoice_amount"]
                        committed.append(CommittedField(
                            field_path="invoice_amount",
                            value=amt_numeric,
                            page=1,
                            bbox=(0.0, 0.0, 0.0, 0.0),
                            evidence_text=evidence,
                            model="pipeline_recovery",
                            model_confidence=0.72,
                            judge_actions=[],
                            final_confidence=0.72,
                        ))

        # 4c. Currency symbol recovery: if currency is missing or in residuals but
        # the document contains an unambiguous currency symbol (£, €, ¥, etc.),
        # derive the ISO code from the symbol. The symbol IS in the text (evidence
        # is the matched character) so this is not fabrication.
        #
        # Priority order: explicit ISO code in text (e.g. "USD", "GBP") first,
        # then fallback to symbol. We only add recovery if currency is absent.
        _CURRENCY_SYMBOL_MAP = {
            "£": "GBP",
            "€": "EUR",
            "¥": "JPY",
            "₹": "INR",
            "₩": "KRW",
            "₣": "CHF",
            "A$": "AUD",
            "C$": "CAD",
            "NZ$": "NZD",
            "S$": "SGD",
            "HK$": "HKD",
            "R$": "BRL",
            "₽": "RUB",
            "฿": "THB",
            "₺": "TRY",
            "Rp": "IDR",
            "$": "USD",  # plain dollar — lowest priority (checked after multi-char variants)
        }
        _CURRENCY_TEXT_MAP = {
            "british pound": "GBP", "pounds sterling": "GBP", "sterling": "GBP",
            "euro": "EUR", "euros": "EUR",
            "us dollar": "USD", "u.s. dollar": "USD", "american dollar": "USD",
            "canadian dollar": "CAD",
            "australian dollar": "AUD",
            "swiss franc": "CHF",
            "japanese yen": "JPY",
            "chinese yuan": "CNY",
            "indian rupee": "INR",
        }
        # Step 4c checks two cases:
        # (a) currency field missing entirely → recover from symbol / text
        # (b) currency field committed but value is a raw symbol (e.g. "$", "£") not an ISO code
        #     → replace with the ISO code (the grounded judge may have returned the symbol verbatim)
        _ISO_CURRENCY_RE = re.compile(r'^[A-Z]{3}$')

        committed_fields_set = {cf.field_path for cf in committed}
        _existing_currency_is_raw_symbol = False
        _existing_currency_cf: CommittedField | None = None
        for _cf in committed:
            if _cf.field_path == "currency":
                _existing_currency_cf = _cf
                if not _ISO_CURRENCY_RE.match((_cf.value or "").strip()):
                    # Value is not a 3-letter ISO code (e.g. "$", "£", "USD$", etc.)
                    _existing_currency_is_raw_symbol = True
                break

        _need_currency_recovery = (
            "currency" not in committed_fields_set
            or _existing_currency_is_raw_symbol
        )

        if _need_currency_recovery and parsed is not None:
            _text = parsed.full_text
            _recovered_currency: str | None = None
            _recovered_evidence: str | None = None

            # Try named currency first (higher confidence)
            _text_lower = _text.lower()
            for phrase, iso in _CURRENCY_TEXT_MAP.items():
                if phrase in _text_lower:
                    _recovered_currency = iso
                    _recovered_evidence = phrase
                    break

            # Fall back to symbol scan if no text match
            if _recovered_currency is None:
                # Multi-char symbols first (A$, C$, etc.) to avoid $ ambiguity
                for symbol in sorted(_CURRENCY_SYMBOL_MAP, key=len, reverse=True):
                    if symbol in _text:
                        _recovered_currency = _CURRENCY_SYMBOL_MAP[symbol]
                        _recovered_evidence = symbol
                        break

            if _recovered_currency is not None:
                log.info(
                    "Currency symbol recovery: %r → %s for %s",
                    _recovered_evidence, _recovered_currency, path.name,
                )
                # Remove any existing currency committed field and residuals
                committed = [cf for cf in committed if cf.field_path != "currency"]
                residuals = [rf for rf in residuals if rf.field_path != "currency"]
                committed.append(CommittedField(
                    field_path="currency",
                    value=_recovered_currency,
                    page=1,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    evidence_text=_recovered_evidence or "",
                    model="pipeline_recovery",
                    model_confidence=0.85,
                    judge_actions=[],
                    final_confidence=0.85,
                ))

        # 4d. Tax-percent range guard: tax_percent must be in [0, 50].
        # Real-world tax rates never exceed 50% (most are 0–25%).
        # If an extractor returns 87.0 or 95.41, it captured a monetary amount
        # (e.g. qa_roberta answering "What is the Tax Rate?" with "$87.00" → 87.0).
        # The previous upper bound of 100 was too permissive; tightened to 50
        # to eliminate these monetary-value false positives while accepting all
        # legitimate tax rates (GST 10%, VAT 20%, HST 13%, etc.).
        _TAX_PCT_MAX = 50.0
        cleaned_committed3: list[CommittedField] = []
        for cf in committed:
            if cf.field_path == "tax_percent":
                try:
                    pct_val = float(cf.value)
                    if not (0.0 <= pct_val <= _TAX_PCT_MAX):
                        log.debug(
                            "Rejecting tax_percent=%s (out of [0,%s] range) from %s",
                            cf.value, _TAX_PCT_MAX, cf.model,
                        )
                        continue
                except (ValueError, TypeError):
                    pass
            cleaned_committed3.append(cf)
        committed = cleaned_committed3

        # 4e. Supplier-name garbage rejection.
        # The supplier_name field uses broad canonical labels and multiple
        # extractors, which frequently pick up footer text, email addresses,
        # address blocks, label words, and other non-company-name content.
        # Apply the resolver's own _is_garbage_name logic here so that garbage
        # values don't make it into provenance at all (the resolver would reject
        # them, leaving supplier_id NULL — but provenance would still show the
        # bad extraction). Removing them here means provenance is clean and the
        # next extraction attempt has a fresh chance.
        cleaned_committed_sup: list[CommittedField] = []
        for cf in committed:
            if cf.field_path in ("supplier_name", "buyer_id"):
                try:
                    from src.services.extraction_v3.supplier_resolver import _is_garbage_name
                    val = (cf.value or "").strip()
                    if val and _is_garbage_name(val):
                        log.info(
                            "Pipeline: dropping garbage supplier/buyer name %r (field=%s model=%s)",
                            val, cf.field_path, cf.model,
                        )
                        continue  # drop — leave field as residual/absent
                except Exception:
                    pass  # if import fails, keep the candidate unchanged
            cleaned_committed_sup.append(cf)
        committed = cleaned_committed_sup

        # 5. Demote on critical invariant failures
        critical = [r for r in invariant_results if r.severity in ("CRITICAL", "critical", "fail")]
        if critical:
            # Already in residuals if it's about a specific field; otherwise just demote confidences
            for cf in committed:
                cf.final_confidence = max(0.0, cf.final_confidence - 0.15)

        # 6. Line items: emit committed CommittedFields per (row_idx, field_name)
        # Deduplicate: multiple extractors may emit candidates for the same
        # field path (e.g. table_transformer + layoutlmv3 both find the same cell).
        # Keep the highest-confidence candidate per field_path.
        line_cands_by_field: dict[str, Candidate] = {}
        for cand in line_cands:
            existing = line_cands_by_field.get(cand.field)
            if existing is None or cand.confidence > existing.confidence:
                line_cands_by_field[cand.field] = cand
        for cand in line_cands_by_field.values():
            committed.append(CommittedField(
                field_path=cand.field,
                value=cand.value,
                page=cand.page,
                bbox=cand.bbox,
                evidence_text=cand.evidence_text,
                model=cand.model,
                model_confidence=cand.confidence,
                judge_actions=[],
                final_confidence=cand.confidence,
            ))

        # 6e. Tax-percent regex recovery: run BEFORE 6b so that invoice_amount
        # recovery has access to the tax rate when computing the pre-tax subtotal.
        # If tax_percent was not extracted by any extractor but the document
        # contains a pattern like "Tax (10%)" or "GST (15%)", extract it from raw text.
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            if "tax_percent" not in committed_fields and parsed is not None:
                _TAX_PCT_RE = re.compile(
                    r"(?:Tax|VAT|GST|HST|PST)\s*[:\(]\s*(\d+(?:\.\d+)?)\s*%",
                    re.IGNORECASE,
                )
                m = _TAX_PCT_RE.search(parsed.full_text)
                if m:
                    pct_str = m.group(1)
                    evidence = m.group(0)
                    log.debug(
                        "Tax-percent regex recovery (early): found %r → %s%%",
                        evidence, pct_str,
                    )
                    committed.append(CommittedField(
                        field_path="tax_percent",
                        value=pct_str,
                        page=1,
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        evidence_text=evidence,
                        model="pipeline_recovery",
                        model_confidence=0.75,
                        judge_actions=[],
                        final_confidence=0.75,
                    ))

        # 6f. Tax-amount derivation (early): run BEFORE 6b so invoice_amount
        # recovery can use tax_amount to compute the pre-tax subtotal.
        # If tax_amount is missing but BOTH tax_percent and invoice_total_incl_tax
        # are committed, derive tax_amount = total - (total / (1 + pct/100)).
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            if (
                "tax_amount" not in committed_fields
                and "tax_percent" in committed_fields
                and "invoice_total_incl_tax" in committed_fields
            ):
                try:
                    pct = float(committed_fields["tax_percent"].value)
                    total = float(committed_fields["invoice_total_incl_tax"].value)
                    if 0 < pct <= 50.0 and total > 0:
                        pretax = total / (1.0 + pct / 100.0)
                        tax_derived = round(total - pretax, 2)
                        evidence_str = committed_fields["tax_percent"].evidence_text
                        log.debug(
                            "Tax-amount derivation (early): total=%s pct=%s%% → tax_amount=%s",
                            total, pct, tax_derived,
                        )
                        committed.append(CommittedField(
                            field_path="tax_amount",
                            value=str(tax_derived),
                            page=committed_fields["invoice_total_incl_tax"].page,
                            bbox=(0.0, 0.0, 0.0, 0.0),
                            evidence_text=evidence_str,
                            model="pipeline_recovery",
                            model_confidence=0.70,
                            judge_actions=[],
                            final_confidence=0.70,
                        ))
                        residuals = [rf for rf in residuals if rf.field_path != "tax_amount"]
                except (ValueError, TypeError, KeyError):
                    pass

        # 6b. Invoice-specific recovery: if invoice_amount is a residual but
        # invoice_total_incl_tax is committed, derive invoice_amount.
        #
        # Runs AFTER 6e/6f so that tax data is available for accurate subtotal derivation.
        # Two cases:
        #   (a) No-tax invoice (no tax_amount and no tax_percent committed):
        #       invoice_amount = invoice_total_incl_tax (grand total = net amount).
        #   (b) Taxed invoice (tax_amount OR tax_percent committed):
        #       invoice_amount = invoice_total_incl_tax - tax_amount (net subtotal).
        #       If tax_amount is not committed but tax_percent is, derive:
        #       invoice_amount = total / (1 + pct/100).
        #
        # This ensures invoice_amount always reflects the PRE-TAX subtotal on
        # invoices that have tax, not the grand total.
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            residual_fields = {rf.field_path for rf in residuals}
            if (
                "invoice_amount" in residual_fields
                and "invoice_total_incl_tax" in committed_fields
            ):
                total_cf = committed_fields["invoice_total_incl_tax"]
                _tax_amt_cf = committed_fields.get("tax_amount")
                _tax_pct_cf = committed_fields.get("tax_percent")

                derived_value = total_cf.value  # default: use total (no-tax case)
                recovery_conf = 0.60

                if _tax_amt_cf is not None:
                    # Case (b1): subtract known tax_amount from total
                    try:
                        subtotal = round(float(total_cf.value) - float(_tax_amt_cf.value), 2)
                        if subtotal > 0:
                            derived_value = str(subtotal)
                            recovery_conf = 0.70
                            log.debug(
                                "invoice_amount recovery (tax-aware): %s - %s = %s",
                                total_cf.value, _tax_amt_cf.value, derived_value,
                            )
                    except (ValueError, TypeError):
                        pass
                elif _tax_pct_cf is not None:
                    # Case (b2): back-calculate subtotal from tax rate
                    try:
                        pct = float(_tax_pct_cf.value)
                        total = float(total_cf.value)
                        if 0 < pct <= 50.0 and total > 0:
                            subtotal = round(total / (1.0 + pct / 100.0), 2)
                            derived_value = str(subtotal)
                            recovery_conf = 0.65
                            log.debug(
                                "invoice_amount recovery (tax-pct): %s / (1+%s%%) = %s",
                                total, pct, derived_value,
                            )
                    except (ValueError, TypeError):
                        pass
                else:
                    log.debug(
                        "invoice_amount recovery (no-tax): invoice_amount ← invoice_total_incl_tax (%s)",
                        total_cf.value,
                    )

                committed.append(CommittedField(
                    field_path="invoice_amount",
                    value=derived_value,
                    page=total_cf.page,
                    bbox=total_cf.bbox,
                    evidence_text=total_cf.evidence_text,
                    model="pipeline_recovery",
                    model_confidence=recovery_conf,
                    judge_actions=[],
                    final_confidence=recovery_conf,
                ))
                # Remove invoice_amount from residuals
                residuals = [rf for rf in residuals if rf.field_path != "invoice_amount"]

        # 6c. Spurious-tax suppression: if tax_amount == invoice_amount (or
        # tax_amount == invoice_total_incl_tax) AND tax_percent is absent,
        # the model extracted the grand total line as tax. Remove it.
        # This commonly happens when a "TOTAL: $9085" line is the only amount
        # and layoutlmv3 fires both invoice_total_incl_tax and tax_amount on it.
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            _tax = committed_fields.get("tax_amount")
            _inv_amt = committed_fields.get("invoice_amount")
            _inv_total = committed_fields.get("invoice_total_incl_tax")
            _tax_pct = committed_fields.get("tax_percent")
            if (
                _tax is not None
                and _tax_pct is None
                and (
                    (_inv_amt is not None and _tax.value == _inv_amt.value)
                    or (_inv_total is not None and _tax.value == _inv_total.value)
                )
            ):
                log.debug(
                    "Spurious-tax suppression: tax_amount=%s equals invoice amount → removing",
                    _tax.value,
                )
                committed = [cf for cf in committed if cf.field_path != "tax_amount"]

        # 6d. Invoice-amount correction (no-tax invoices only):
        # When invoice_amount < invoice_total_incl_tax AND there is no tax evidence
        # (no tax_amount and no tax_percent committed), the two fields are likely
        # capturing the same "single total" line from different label anchors.
        # In that case, invoice_amount should equal invoice_total_incl_tax.
        #
        # When tax IS present (tax_amount or tax_percent committed), invoice_amount
        # is intentionally the net/pre-tax subtotal — do NOT override it.
        # The schema canonical_labels for invoice_amount ("Subtotal", "Net Amount",
        # "Amount Before Tax") are mutually exclusive with invoice_total_incl_tax
        # labels ("Grand Total", "Total", "Amount Due"), so the proximity extractor
        # should find the correct value for each field independently.
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            _ia = committed_fields.get("invoice_amount")
            _it = committed_fields.get("invoice_total_incl_tax")
            _tax_amt = committed_fields.get("tax_amount")
            _tax_pct = committed_fields.get("tax_percent")
            # Only correct when no tax is indicated at all
            tax_present = (_tax_amt is not None) or (_tax_pct is not None)
            if _ia is not None and _it is not None and not tax_present:
                try:
                    ia_val = float(_ia.value)
                    it_val = float(_it.value)
                    # No-tax invoice: invoice_amount should equal invoice_total_incl_tax.
                    # Only correct if they differ (within-rounding tolerance 0.5%).
                    if ia_val < it_val and abs(ia_val - it_val) > 0.005 * it_val:
                        log.debug(
                            "Invoice-amount no-tax correction: invoice_amount=%s != invoice_total_incl_tax=%s "
                            "and no tax found — setting invoice_amount = invoice_total_incl_tax",
                            ia_val, it_val,
                        )
                        committed = [cf for cf in committed if cf.field_path != "invoice_amount"]
                        committed.append(CommittedField(
                            field_path="invoice_amount",
                            value=_it.value,
                            page=_it.page,
                            bbox=_it.bbox,
                            evidence_text=_it.evidence_text,
                            model="pipeline_recovery",
                            model_confidence=0.70,
                            judge_actions=[],
                            final_confidence=0.70,
                        ))
                except (ValueError, TypeError):
                    pass

        # 7. Determine doc_pk from the committed fields (the field whose YAML
        # name matches the schema's doc_pk; for invoice that's invoice_id)
        doc_pk = next(
            (cf.value for cf in committed if cf.field_path == self._doc_pk_field(schema.doc_type)),
            None,
        )

        # 7b. Synthetic invoice_id fallback: if invoice_id could not be extracted,
        # generate a deterministic synthetic ID so the row can be persisted and
        # routed to the manual review queue (rather than silently dropped).
        # Format: SYN-{vendor_slug}-{date_iso}-{file_hash[:6]}
        # where vendor_slug = first 12 chars of supplier_name (if available)
        #       date_iso    = invoice_date if committed, else today (YYYY-MM-DD)
        #       file_hash   = sha256 of the source file bytes, first 6 hex chars
        if doc_pk is None and schema.doc_type == "invoice":
            try:
                import hashlib
                from datetime import date as _date

                # Compute file hash
                file_bytes = path.read_bytes()
                file_hash = hashlib.sha256(file_bytes).hexdigest()[:6]

                # Get vendor slug from committed supplier_name
                committed_map = {cf.field_path: cf.value for cf in committed}
                supplier_raw = committed_map.get("supplier_name", "")
                vendor_slug = re.sub(r'[^A-Za-z0-9]', '', supplier_raw.title())[:12] or "Unknown"

                # Get date: use invoice_date if committed, else today
                date_iso = committed_map.get("invoice_date", str(_date.today()))
                # Normalise date to YYYY-MM-DD (drop anything after the date)
                date_iso = date_iso[:10] if len(date_iso) >= 10 else str(_date.today())

                syn_id = f"SYN-{vendor_slug}-{date_iso}-{file_hash}"
                doc_pk = syn_id

                # Inject into committed so persistence writes invoice_id
                committed.append(CommittedField(
                    field_path="invoice_id",
                    value=syn_id,
                    page=1,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    evidence_text=f"synthetic_invoice_id:{syn_id}",
                    model="pipeline_recovery",
                    model_confidence=0.30,
                    judge_actions=[],
                    final_confidence=0.30,
                ))
                # Remove any residual for invoice_id
                residuals = [rf for rf in residuals if rf.field_path != "invoice_id"]
                log.warning(
                    "Synthetic invoice_id generated for %s: %s (no extractable invoice_id found)",
                    path.name, syn_id,
                )
            except Exception:
                log.exception("Failed to generate synthetic invoice_id for %s", path)

        return ExtractionResult(
            doc_type=schema.doc_type,
            doc_pk=doc_pk,
            committed=committed,
            residuals=residuals,
            judge_calls=orch.judge_calls,
            pipeline_version=PIPELINE_VERSION,
        )

    def _build_prelim_lines(self, line_cands: list[Candidate]) -> list[dict]:
        """Group line-item candidates by row index → list of dicts."""
        rows: dict[int, dict] = {}
        for c in line_cands:
            # field looks like "line_items[3].amount"
            try:
                idx = int(c.field.split("[", 1)[1].split("]", 1)[0])
                key = c.field.split("].", 1)[1]
            except (IndexError, ValueError):
                continue
            rows.setdefault(idx, {})[key] = c.value
        return [rows[i] for i in sorted(rows)]

    def _doc_pk_field(self, doc_type: str) -> str:
        return {
            "invoice": "invoice_id",
            "purchase_order": "po_id",
            "quote": "quote_id",
            "contract": "contract_id",
        }.get(doc_type, "id")
