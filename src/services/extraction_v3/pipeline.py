"""Pipeline V3 orchestrator. Sequence:
   1. Layer 1: parse the doc → ParsedDocument
   2. Layer 2: run all extractors named in the schema (parallel) → candidates
   3. Layer 3: type-bind, run invariants, judge-orchestrate, assemble result.
"""
from __future__ import annotations
import logging
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
import src.services.extraction_v3.extractors.layoutlmv3       # noqa
import src.services.extraction_v3.extractors.table_transformer  # noqa
import src.services.extraction_v3.extractors.sbert_anchor      # noqa
import src.services.extraction_v3.extractors.spacy_ner         # noqa
import src.services.extraction_v3.extractors.qa_roberta        # noqa
import src.services.extraction_v3.extractors.vendor_template   # noqa

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

        # 5. Demote on critical invariant failures
        critical = [r for r in invariant_results if r.severity in ("CRITICAL", "critical", "fail")]
        if critical:
            # Already in residuals if it's about a specific field; otherwise just demote confidences
            for cf in committed:
                cf.final_confidence = max(0.0, cf.final_confidence - 0.15)

        # 6. Line items: emit committed CommittedFields per (row_idx, field_name)
        for cand in line_cands:
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
        # invoice_total_incl_tax is committed AND no tax was extracted, infer
        # invoice_amount = invoice_total_incl_tax (no-tax invoice pattern).
        # Also handles the case where tax_amount was spuriously extracted as
        # the same value as invoice_total_incl_tax (duplicate extraction from a
        # GRAND TOTAL line — e.g. PO5 split invoices with only one total figure).
        if schema.doc_type == "invoice":
            committed_fields = {cf.field_path: cf for cf in committed}
            residual_fields = {rf.field_path for rf in residuals}
            # Detect spurious tax: tax_amount == invoice_total_incl_tax → treat as no-tax
            _tax_amount_cf = committed_fields.get("tax_amount")
            _total_cf_check = committed_fields.get("invoice_total_incl_tax")
            _tax_equals_total = (
                _tax_amount_cf is not None
                and _total_cf_check is not None
                and _tax_amount_cf.value == _total_cf_check.value
            )
            if (
                "invoice_amount" in residual_fields
                and "invoice_total_incl_tax" in committed_fields
                and ("tax_amount" not in committed_fields or _tax_equals_total)
                and "tax_percent" not in committed_fields
            ):
                # No tax extracted — treat grand total as the pre-tax amount too
                total_cf = committed_fields["invoice_total_incl_tax"]
                log.debug(
                    "No-tax fallback: invoice_amount ← invoice_total_incl_tax (%s)",
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
