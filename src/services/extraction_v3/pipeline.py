"""Pipeline V3 orchestrator. Sequence:
   1. Layer 1: parse the doc → ParsedDocument
   2. Layer 2: run all extractors named in the schema (parallel) → candidates
   3. Layer 3: type-bind, run invariants, judge-orchestrate, assemble result.
"""
from __future__ import annotations
import logging
import re
import concurrent.futures
from pathlib import Path
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.result import (
    ExtractionResult, CommittedField, ResidualField, JudgeAction, ResidualReason,
)
from src.services.extraction_v3.parsers.router import parse as parse_document
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema, DocSchema
from src.services.extraction_v3.yaml_schema.registry import (
    get_extractor, known_extractors,
)
from src.services.extraction_v3.binding.type_binder import bind_typed
from src.services.extraction_v3.binding.invariants_runner import run_invariants
from src.services.extraction_v3.judge.orchestrator import run_judge_orchestrator
from src.services.extraction_v3.judge.contracts import InvariantResultSummary
# Side-effect imports — registers each extractor at module load
import src.services.extraction_v3.extractors.layoutlmv3           # noqa
import src.services.extraction_v3.extractors.layoutlmv3_finetuned  # noqa
import src.services.extraction_v3.extractors.table_transformer      # noqa
import src.services.extraction_v3.extractors.sbert_anchor          # noqa
import src.services.extraction_v3.extractors.spacy_ner             # noqa
import src.services.extraction_v3.extractors.qa_roberta            # noqa
import src.services.extraction_v3.extractors.vendor_template       # noqa

log = logging.getLogger(__name__)

PIPELINE_VERSION = "v3.1.0"
HEADER_LINE_ITEM_PREFIX = "line_items["


def _all_extractor_names_for_schema(schema: DocSchema) -> set[str]:
    names: set[str] = set()
    for f in schema.fields:
        names.update(f.extractors)
    if schema.line_items:
        names.add(schema.line_items.primary_extractor)
        if schema.line_items.fallback_extractor:
            names.add(schema.line_items.fallback_extractor)
    return names


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

        # 2. Run all extractors in parallel (Layer 2)
        extractor_names = _all_extractor_names_for_schema(schema) & known_extractors()
        all_candidates: list[Candidate] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self._safe_run_extractor, name, parsed, schema): name
                for name in extractor_names
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    all_candidates.extend(fut.result())
                except Exception:
                    log.exception("extractor %s failed", name)

        log.info("Pipeline V3: %d candidates from %d extractors", len(all_candidates), len(extractor_names))

        # Split header candidates from line-item candidates
        header_cands = [c for c in all_candidates if not c.field.startswith(HEADER_LINE_ITEM_PREFIX)]
        line_cands = [c for c in all_candidates if c.field.startswith(HEADER_LINE_ITEM_PREFIX)]

        # 3a. Header candidates → judge orchestrator
        header_by_field = _group_candidates_by_field(header_cands)

        # Pre-filter: drop candidates that fail type-binding when valid
        # alternatives exist for the same field.  This prevents qa_roberta
        # multi-line / over-extracted values (e.g. "$27.00\n\nShipping…")
        # from winning the tiebreaker over a correctly-typed candidate.
        fields_by_name = {f.name: f for f in schema.fields}
        for fname, cands in list(header_by_field.items()):
            fspec = fields_by_name.get(fname)
            if not fspec or len(cands) < 2:
                continue
            valid = [c for c in cands if not bind_typed(c, fspec.type).bind_error]
            if valid and len(valid) < len(cands):
                log.debug(
                    "Pre-filter: dropping %d non-binding candidates for %s",
                    len(cands) - len(valid), fname,
                )
                header_by_field[fname] = valid

        # Pre-filter 2: for total/grand-total money fields with competing candidates,
        # keep only the LARGEST value. Financial totals (invoice_total_incl_tax,
        # invoice_amount) are always the maximum amount in the document — subtotals
        # like "Accessories Total" will never be larger than "Total Sale Price".
        # This deterministic selection avoids Ollama-dependent tiebreaking for this
        # common pattern.
        _MAX_VALUE_FIELDS = {"invoice_total_incl_tax", "invoice_amount"}
        for fname in _MAX_VALUE_FIELDS:
            cands = header_by_field.get(fname, [])
            if len(cands) >= 2:
                from src.services.extraction_v2.parsers.amounts import parse_amount as _pa
                def _to_float(c: Candidate) -> float:
                    try:
                        r = _pa(c.value or "")
                        return float(r) if r is not None else 0.0
                    except Exception:
                        return 0.0
                best = max(cands, key=_to_float)
                log.debug(
                    "Max-value pre-filter: %s → keeping %r (was %d candidates)",
                    fname, best.value, len(cands),
                )
                header_by_field[fname] = [best]

        # Pre-filter 3: Universal NER-type veto for ORG-typed fields.
        # For ALL candidates for fields with ner_type_check="ORG", run spaCy on
        # each non-spacy-ner candidate value to check its entity type. If the
        # value is classified as PERSON, DATE, CARDINAL, ORDINAL, or similar
        # non-organisation types, drop it — those are never valid buyer/supplier IDs.
        # This prevents layoutlmv3 (canonical-label proximity), layoutlmv3_finetuned
        # (jinhybr), sbert_anchor, or any other extractor from committing
        # "John Smith" or "June 5, 2025" as buyer_id when the BILL TO block
        # contains only a contact name.
        #
        # Veto logic (for buyer_id and other ORG-typed fields):
        #   - spacy_ner candidates: always kept (they already did ORG disambiguation)
        #   - PERSON-only → veto (person name is not a company)
        #   - DATE / TIME / CARDINAL / ORDINAL only → veto (temporal/numeric is not a company)
        #   - ORG present → keep (even if also other types)
        #   - No entities recognised → keep (could be unknown company name)
        #
        # Fires for ALL non-spacy candidates, regardless of whether spacy_ner
        # also produced a candidate for the same field. This is the universal fix:
        # it works even when layoutlmv3 at conf=0.90 would otherwise beat the
        # spacy_ner ORG candidate in the tiebreaker.
        _ORG_VETO_TYPES = frozenset({"PERSON", "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"})
        org_check_fields = {
            f.name for f in schema.fields
            if f.judge.ner_type_check == "ORG"
        }
        if org_check_fields:
            try:
                from src.services.extraction_v3.extractors.spacy_ner import _run_nlp as _spacy_run
                for fname in org_check_fields:
                    cands = header_by_field.get(fname, [])
                    if not cands:
                        continue
                    # Check each non-spacy_ner candidate's value against spaCy NER.
                    # spacy_ner candidates are always kept — they produced ORG-shaped
                    # values through their own disambiguation logic.
                    vetoed: list[Candidate] = []
                    surviving: list[Candidate] = []
                    for cand in cands:
                        # Always keep spacy_ner's own candidates unchanged
                        if cand.model == "spacy_ner":
                            surviving.append(cand)
                            continue
                        if not cand.value:
                            surviving.append(cand)
                            continue
                        try:
                            _doc = _spacy_run(cand.value.strip())
                            ent_labels = {e.label_ for e in _doc.ents}
                            # Keep if ORG is present in the recognised labels
                            if "ORG" in ent_labels:
                                surviving.append(cand)
                                continue
                            # Veto if any entity type is a clear non-ORG type
                            if ent_labels & _ORG_VETO_TYPES:
                                log.debug(
                                    "NER-type veto: dropping %s=%r from %s (spaCy=%r, expected ORG)",
                                    fname, cand.value, cand.model, ent_labels,
                                )
                                vetoed.append(cand)
                            else:
                                # No entities recognised OR non-vetoed types → keep
                                # (unknown company names often have no spaCy entity)
                                surviving.append(cand)
                        except Exception:
                            surviving.append(cand)  # veto failed — keep candidate
                    if surviving:
                        header_by_field[fname] = surviving
                    elif vetoed:
                        # All candidates vetoed — remove the field entirely
                        # (leaves buyer_id as residual/NULL rather than committing
                        # a person name or date string)
                        header_by_field.pop(fname, None)
                        log.debug(
                            "NER-type veto: all candidates for %s were non-ORG — field left as residual",
                            fname,
                        )
            except Exception as _ner_veto_exc:
                log.debug("NER-type veto skipped (import error): %s", _ner_veto_exc)

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
            header_by_field, schema, parsed.full_text, invariant_summaries,
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

        # 4b. Cross-field duplicate suppression: if a qa_roberta-only candidate's
        # committed value is identical to a higher-confidence candidate already
        # committed for a DIFFERENT required field, discard it (return to residual).
        # This prevents "PO id = 01-2024-001" from bleeding into payment_terms.
        required_field_values: dict[str, str] = {}
        for cf in list(committed):
            fspec = next((f for f in schema.fields if f.name == cf.field_path), None)
            if fspec and fspec.required:
                required_field_values[cf.field_path] = cf.value

        cleaned_committed: list[CommittedField] = []
        for cf in committed:
            fspec = next((f for f in schema.fields if f.name == cf.field_path), None)
            if (
                fspec and not fspec.required
                and cf.model == "qa_roberta"
                and cf.value in required_field_values.values()
                and cf.field_path not in required_field_values
            ):
                # This non-required field's qa_roberta value duplicates a required
                # field's committed value → suppress it (leave as residual / absent)
                log.debug(
                    "Suppressing qa_roberta duplicate: field=%s value=%r mirrors a required field",
                    cf.field_path, cf.value
                )
                continue
            cleaned_committed.append(cf)
        committed = cleaned_committed

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
                r"[\s:]+([A-Za-z0-9][A-Za-z0-9\-/\.]{1,30})",
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

        # 4c. Date-field duplicate suppression: when the document contains only one
        # date, qa_roberta will return the same value for every date field it answers.
        # Suppress qa_roberta date fields whose value matches a non-qa_roberta date
        # field already committed (the non-qa_roberta source is the authoritative one).
        # Also suppress when multiple qa_roberta date fields share the same value —
        # keep only the one for the field that most plausibly owns that date token
        # (the field with the highest layoutlmv3 confidence, else the required field).
        non_qa_date_values: dict[str, str] = {}  # field_path → value for non-qa dates
        for cf in committed:
            fspec = next((f for f in schema.fields if f.name == cf.field_path), None)
            if fspec and fspec.type == "iso_date" and cf.model != "qa_roberta":
                non_qa_date_values[cf.field_path] = cf.value

        cleaned_committed2: list[CommittedField] = []
        qa_date_by_value: dict[str, list[CommittedField]] = {}
        for cf in committed:
            fspec = next((f for f in schema.fields if f.name == cf.field_path), None)
            if (
                fspec and fspec.type == "iso_date"
                and cf.model == "qa_roberta"
                and cf.value in non_qa_date_values.values()
            ):
                # qa_roberta returned a date already committed by a better source → suppress
                log.debug(
                    "Suppressing qa_roberta date duplicate: field=%s value=%r already covered by non-qa source",
                    cf.field_path, cf.value,
                )
                continue
            cleaned_committed2.append(cf)
        committed = cleaned_committed2

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

        # 6b. Invoice-specific recovery: if invoice_amount is a residual but
        # invoice_total_incl_tax is committed, set invoice_amount = invoice_total_incl_tax.
        # This reflects the DB convention that invoice_amount stores the total billed
        # amount (Grand Total). Covers two sub-cases:
        #   (a) No-tax invoice — grand total = pre-tax amount.
        #   (b) Tax-inclusive invoice — grand total includes tax; invoice_amount = grand total.
        # Also handles spurious tax (tax_amount == invoice_total_incl_tax, i.e. the model
        # fired on the same GRAND TOTAL line for both fields).
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            residual_fields = {rf.field_path for rf in residuals}
            if (
                "invoice_amount" in residual_fields
                and "invoice_total_incl_tax" in committed_fields
            ):
                total_cf = committed_fields["invoice_total_incl_tax"]
                log.debug(
                    "invoice_amount recovery: invoice_amount ← invoice_total_incl_tax (%s)",
                    total_cf.value,
                )
                committed.append(CommittedField(
                    field_path="invoice_amount",
                    value=total_cf.value,
                    page=total_cf.page,
                    bbox=total_cf.bbox,
                    evidence_text=total_cf.evidence_text,
                    model="pipeline_recovery",
                    model_confidence=0.60,
                    judge_actions=[],
                    final_confidence=0.60,
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

        # 6d. Invoice-amount correction: if invoice_amount < invoice_total_incl_tax
        # and tax_amount is present and (invoice_amount + tax_amount) is not close
        # to invoice_total_incl_tax, then invoice_amount is likely a partial subtotal
        # (e.g. "Total Materials"). Replace with invoice_total_incl_tax so that DB
        # convention (invoice_amount = amount billed) holds.
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            _ia = committed_fields.get("invoice_amount")
            _it = committed_fields.get("invoice_total_incl_tax")
            if _ia is not None and _it is not None:
                try:
                    ia_val = float(_ia.value)
                    it_val = float(_it.value)
                    ta_val = float(committed_fields["tax_amount"].value) if "tax_amount" in committed_fields else 0.0
                    # If invoice_amount is smaller than invoice_total_incl_tax and
                    # the sum (invoice_amount + tax_amount) is not within 1% of
                    # invoice_total_incl_tax, invoice_amount is a subtotal → correct.
                    sum_check = ia_val + ta_val
                    within_1pct = abs(sum_check - it_val) <= 0.01 * it_val if it_val else False
                    if ia_val < it_val and not within_1pct:
                        log.debug(
                            "Invoice-amount correction: invoice_amount=%s < invoice_total_incl_tax=%s "
                            "and sum_check=%s not within 1%% — replacing with invoice_total_incl_tax",
                            ia_val, it_val, sum_check,
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
                except (ValueError, TypeError, KeyError):
                    pass

        # 6e. Tax-percent regex recovery: if tax_percent was not extracted by
        # any extractor but the document contains a pattern like "Tax (10%)" or
        # "Tax: 10%" or "GST (5%)", extract it from the raw text.
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
                        "Tax-percent regex recovery: found %r → %s%%",
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

        # 6f. Tax-amount derivation: if tax_amount is missing but BOTH tax_percent
        # and invoice_total_incl_tax are committed, derive tax_amount as:
        #   tax_amount = total - (total / (1 + pct/100))
        # This accurately recovers the tax dollar amount for invoices that show
        # "Tax (10%)" + grand total without printing a separate tax_amount line.
        # Only fire when invoice_total_incl_tax > 0 and pct in (0, 50].
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
                        # Evidence must be a substring of full_text.
                        # Use the tax_percent evidence (which IS in the text) as the
                        # provenance anchor for this derived field.
                        evidence_str = committed_fields["tax_percent"].evidence_text
                        log.debug(
                            "Tax-amount derivation: total=%s pct=%s%% → tax_amount=%s",
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
                        # Remove any residual for tax_amount
                        residuals = [rf for rf in residuals if rf.field_path != "tax_amount"]
                except (ValueError, TypeError, KeyError):
                    pass

        # 7. Determine doc_pk from the committed fields (the field whose YAML
        # name matches the schema's doc_pk; for invoice that's invoice_id)
        doc_pk = next(
            (cf.value for cf in committed if cf.field_path == self._doc_pk_field(schema.doc_type)),
            None,
        )

        return ExtractionResult(
            doc_type=schema.doc_type,
            doc_pk=doc_pk,
            committed=committed,
            residuals=residuals,
            judge_calls=orch.judge_calls,
            pipeline_version=PIPELINE_VERSION,
        )

    def _safe_run_extractor(self, name: str, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        cls = get_extractor(name)
        instance = cls()
        return instance.produce_candidates(parsed, schema)

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
