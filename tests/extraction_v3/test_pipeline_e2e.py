"""End-to-end integration tests for the extraction_v3 pipeline.

Parameterized over the 5 fixtures in tests/extraction_v3/fixtures/invoices/.
For each fixture: parse -> extract -> persist -> re-read DB -> compare against
ground-truth expected.json field by field.

Per the iteration mandate (project_extraction_redesign_iteration_mandate.md):
fixtures that fail must be iterated until they pass (in this plan or with a
written escalation to Plan 2 / Plan 3 / Plan 4). DO NOT mark with xfail/skip.
"""
from pathlib import Path
import json
import pytest

from src.services.extraction_v3.pipeline import PipelineV3, PIPELINE_VERSION
from src.services.extraction_v3.persistence import persist
from src.services.extraction_v3.parsers.router import parse as parse_doc
from src.services.extraction_v3.schemas.result import ExtractionResult
from src.services.db import get_conn


FIXTURES_DIR = Path(__file__).parent / "fixtures/invoices"

# Explicit column list for proc.bp_invoice header read-back.
# Must match _FakePostgresStore._schema_columns() "proc.bp_invoice" entry
# and the actual DB table (enforced by schema drift checks at schema load time).
_BP_INVOICE_COLS = [
    "invoice_id", "supplier_id", "po_id", "buyer_id", "requested_by",
    "requested_date", "invoice_date", "due_date", "invoice_paid_date",
    "payment_terms", "currency", "invoice_amount", "tax_percent",
    "tax_amount", "invoice_total_incl_tax", "exchange_rate_to_usd",
    "converted_amount_usd", "country", "region",
]


def _all_fixtures() -> list[tuple[str, Path, dict]]:
    """Discover (id, pdf_path, expected_dict) triples for every .expected.json."""
    out = []
    for ej in sorted(FIXTURES_DIR.glob("*.expected.json")):
        fixture_id = ej.stem.replace(".expected", "")
        # Match on either .pdf or .docx
        pdf_or_docx = next((p for p in [
            FIXTURES_DIR / f"{fixture_id}.pdf",
            FIXTURES_DIR / f"{fixture_id}.docx",
        ] if p.exists()), None)
        if pdf_or_docx is None:
            continue
        out.append((fixture_id, pdf_or_docx, json.loads(ej.read_text())))
    return out


def _cleanup_invoice(pk: str) -> None:
    """Remove DB rows for a test pk (idempotent fixture cleanup)."""
    if not pk:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_extraction_provenance_v3 WHERE doc_pk = %s", (pk,))
        cur.execute("DELETE FROM proc.bp_invoice_line_items WHERE invoice_id = %s", (pk,))
        cur.execute("DELETE FROM proc.bp_invoice WHERE invoice_id = %s", (pk,))


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_id,doc_path,expected",
    _all_fixtures(),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_e2e_extraction_zero_hallucinations(fixture_id, doc_path, expected):
    """Pipeline V3 ran on each fixture: every committed value's evidence_text
    must be a substring of parsed.full_text. Zero tolerance.

    This is the SAFETY test — accuracy is exercised by the next test.
    Iteration mandate: failures here are P0; iterate (don't skip)."""
    parsed = parse_doc(doc_path)
    result = PipelineV3().run(doc_path, "invoice")

    # Cleanup whatever pk this run produced (we don't actually need to persist
    # for the substring check; just check the ExtractionResult)
    if result.doc_pk:
        _cleanup_invoice(result.doc_pk)

    hallucinations = []
    for cf in result.committed:
        # Skip line item amounts where bind coercion changes format ("£100" -> "100.0")
        # — we check evidence_text not value
        if cf.evidence_text not in parsed.full_text:
            hallucinations.append((cf.field_path, cf.value, cf.evidence_text))

    assert not hallucinations, (
        f"\n[{fixture_id}] HALLUCINATIONS DETECTED — these committed values "
        f"have evidence_text NOT in parsed.full_text:\n"
        + "\n".join(f"  {fp}: value={v!r} evidence={e!r}" for fp, v, e in hallucinations)
    )


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_id,doc_path,expected",
    _all_fixtures(),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_e2e_extraction_against_ground_truth(fixture_id, doc_path, expected):
    """Pipeline V3 persisted output matches the ground-truth expected.json.

    Iteration mandate: failures here are the input to in-plan iteration —
    investigate and fix root cause (parser? extractor? bind? invariant?
    judge prompt?). Only escalate to Plan 2 (fine-tune) if it genuinely
    requires labeled-corpus work.

    Special case: INV-003 has invoice_id=null in the expected JSON. Since
    persist() requires a non-null doc_pk, a pipeline that correctly reports
    no invoice_id produces doc_pk=None, and we assert that rather than
    trying to persist. If the pipeline fabricates an invoice_id, the test
    fails.
    """
    expected_header = expected.get("header", {})
    expected_invoice_id = expected_header.get("invoice_id")

    result = PipelineV3().run(doc_path, "invoice")

    # --- Special case: fixture expects no invoice_id ---
    if expected_invoice_id is None:
        assert result.doc_pk is None, (
            f"[{fixture_id}] expected invoice_id=null but pipeline produced "
            f"doc_pk={result.doc_pk!r} — fabrication detected"
        )
        # Nothing to persist; test passes on the null-pk assertion alone.
        return

    # --- Normal path: invoice_id must have been extracted ---
    if not result.doc_pk:
        pytest.fail(
            f"[{fixture_id}] pipeline produced no doc_pk — invoice_id was not extracted"
        )

    persist(result)

    try:
        # Read back persisted header row using an explicit column list so the
        # in-memory fake DB (used when Postgres is unavailable) can resolve
        # column names correctly. The column set matches _BP_INVOICE_COLS above.
        col_list = ", ".join(_BP_INVOICE_COLS)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {col_list} FROM proc.bp_invoice WHERE invoice_id = %s",
                (result.doc_pk,),
            )
            row = cur.fetchone()
            assert row is not None, (
                f"[{fixture_id}] no row persisted for {result.doc_pk}"
            )
            colnames = [d.name for d in cur.description]
            actual_header = dict(zip(colnames, row))

        # Compare against expected
        diffs = []
        for field, expected_v in expected_header.items():
            # supplier_name resolves to supplier_id in the DB (not a direct column);
            # skip it for Plan 1 — it requires a lookup join outside the scope of
            # the extraction pipeline itself.
            if field == "supplier_name":
                continue
            actual_v = actual_header.get(field)
            # Numeric tolerance for amounts.
            # psycopg2 returns numeric(18,2) columns as Decimal objects; coerce
            # to float for comparison so we don't get Decimal('9085.00') vs 9085.0.
            from decimal import Decimal
            if isinstance(actual_v, Decimal):
                actual_v = float(actual_v)
            if isinstance(expected_v, (int, float)) and isinstance(actual_v, (int, float)):
                if abs(float(actual_v) - float(expected_v)) > 0.05:
                    diffs.append(f"  {field}: expected {expected_v}, got {actual_v}")
            elif expected_v is None:
                # null in expected = field should be NULL in DB
                if actual_v is not None:
                    diffs.append(f"  {field}: expected null, got {actual_v!r}")
            else:
                # string equality (case-insensitive strip)
                if str(actual_v).strip().lower() != str(expected_v).strip().lower():
                    diffs.append(f"  {field}: expected {expected_v!r}, got {actual_v!r}")

        if diffs:
            pytest.fail(
                f"\n[{fixture_id}] FIELD MISMATCHES vs ground truth — iteration required:\n"
                + "\n".join(diffs)
                + f"\n\n(committed: {len(result.committed)}, residuals: {len(result.residuals)}, "
                f"judge_calls: {result.judge_calls})"
            )

    finally:
        _cleanup_invoice(result.doc_pk)


@pytest.mark.gpu
@pytest.mark.integration
def test_e2e_pipeline_returns_valid_extraction_result():
    """Smoke: pipeline produces a structurally valid result on the easiest fixture."""
    result = PipelineV3().run(FIXTURES_DIR / "INV-001-clean.pdf", "invoice")
    assert isinstance(result, ExtractionResult)
    assert result.pipeline_version == PIPELINE_VERSION
    assert result.doc_type == "invoice"
    # Pipeline cost ceiling: judge_calls within bounds
    assert result.judge_calls < 50  # liberal upper bound; tighten in Plan 2
