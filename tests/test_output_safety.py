"""The output-safety contract.

Two duties, and they pull against each other:

  * Nothing internal escapes. Not a table, not a path, not an env var, not a route, not a
    stack trace — and not a mechanism-level *explanation* either, even when it names
    nothing at all. "Extraction is triggered before the upload finishes" leaks no
    identifier and is still a description of the backend, so it fails.

  * Real answers survive untouched. `invoice` and `supplier` are literally table names in
    proc, and they are also ordinary English words a procurement user says all day. A gate
    that cannot tell "Invoice Ltd" from `proc.invoice` is a gate nobody can ship.

The second duty is the hard one, so the regression half of this file matters as much as the
adversarial half.
"""

from __future__ import annotations

import pytest

from services import output_safety as osafe


# --------------------------------------------------------------------------------------
# Adversarial — the hostile questions from the live baseline. These are real leaks that the
# running server produced on 2026-07-14 before the gate existed.
# --------------------------------------------------------------------------------------

LEAKS = [
    pytest.param(
        "The system INSERTs proc.process_monitor with status='Running'.",
        "db_table",
        id="db-table-qualified",
    ),
    pytest.param(
        "Spend is read from the bp_invoice_trgt table.",
        "db_table",
        id="db-table-bp-prefix",
    ),
    pytest.param(
        "The extraction logic lives in src/services/extraction/dispatch.py.",
        "file_path",
        id="file-path",
    ),
    pytest.param(
        "The watcher waits for UPLOAD_FILE_WAIT_TIMEOUT_S before giving up.",
        "env_var",
        id="env-var",
    ),
    pytest.param(
        "Call POST /data-integration/presigned-url to get an upload url.",
        "route",
        id="internal-route",
    ),
    pytest.param(
        "There is no /spendiq/invoices endpoint.",
        "route",
        id="internal-route-negative-claim",
    ),
    pytest.param(
        "I ran SELECT total_amount FROM proc.bp_invoice_trgt WHERE deal_id = 7.",
        "sql",
        id="raw-sql",
    ),
    pytest.param(
        'Traceback (most recent call last):\n  File "src/api/main.py", line 42, in run',
        "stack_trace",
        id="stack-trace",
    ),
    pytest.param(
        "psycopg2.errors.UndefinedTable: relation \"proc.cat_product_mapping\" does not exist",
        "db_table",
        id="psycopg2-schema-disclosure",
    ),
    pytest.param(
        "The document is read by the DataExtractionAgent.",
        "code_identifier",
        id="class-name",
    ),
    pytest.param(
        "The model used is AgentNick:unified running on Ollama.",
        "model_name",
        id="internal-model-name",
    ),
    pytest.param(
        "Connect with postgresql://user:hunter2@10.100.10.180:5432/bp_sqldb",
        "credential",
        id="connection-string",
    ),
]


@pytest.mark.parametrize("text,expected_kind", LEAKS)
def test_identifier_leaks_are_caught(text, expected_kind):
    violations = osafe.inspect(text)
    assert violations, f"gate let an internal identifier through: {text!r}"
    assert expected_kind in {v.kind for v in violations}


# The register rule. None of these name a single internal identifier. They are still
# descriptions of how the backend works, and under the product-level rule they must fail.
MECHANISM_ONLY = [
    pytest.param(
        "Extraction is triggered before the file finishes uploading, so the watcher timed "
        "out waiting for the object to land.",
        id="trigger-watcher",
    ),
    pytest.param(
        "Your document goes through the extraction pipeline: first a regex pass, then an "
        "engineered pass, then an AI judge.",
        id="pipeline-stages",
    ),
    pytest.param(
        "The record failed the promotion gate, so it stayed in staging and was never "
        "promoted to the final table.",
        id="promotion-gate",
    ),
    pytest.param(
        "A cron job runs the scheduler overnight, which enqueues the document for a worker.",
        id="cron-worker-queue",
    ),
    pytest.param(
        "The orchestrator compiles the workflow into a DAG and executes each node in turn.",
        id="dag-orchestrator",
    ),
]


@pytest.mark.parametrize("text", MECHANISM_ONLY)
def test_mechanism_level_prose_is_caught_even_with_no_identifier(text):
    """The rule the user made hard: explain in product terms, never in backend flow.

    The fixture deliberately names no table, path, route or env var — so with the register
    check off it is *clean*. It is caught only because it describes the backend.
    """
    assert osafe.inspect(text, prose=False) == [], (
        "fixture is meant to contain no internal identifier — it should only fail the "
        "register check, so that this test proves the register check is what caught it"
    )

    violations = osafe.inspect(text, prose=True)
    assert violations, f"mechanism-level prose passed the register check: {text!r}"
    assert "mechanism" in {v.kind for v in violations}


def test_mechanism_check_does_not_apply_to_data_fields():
    """A data value is not prose. We must not mangle it hunting for register violations."""
    text = "Pipeline Supplies Ltd"
    assert osafe.inspect(text, prose=False) == []


# --------------------------------------------------------------------------------------
# Regression — the answers that MUST survive. If these break, the gate is unshippable.
# --------------------------------------------------------------------------------------

MUST_SURVIVE = [
    pytest.param(
        "Your file didn't finish uploading before we started reading it. Re-upload it from "
        "the Documents screen — you'll see the status change from Running to Extracted.",
        id="the-correct-product-level-answer",
    ),
    pytest.param(
        "Invoice Ltd quoted £12,400 — that's £1,900 below Dixon Reynolds on the same lines.",
        id="supplier-literally-called-invoice",
    ),
    pytest.param(
        "Supplier Solutions Ltd has 3 open invoices totalling £45,120.50, due 2026-08-01.",
        id="table-names-as-english-words",
    ),
    pytest.param(
        "This deal covers 12 purchase orders across 4 suppliers in the Facilities category.",
        id="domain-vocabulary",
    ),
    pytest.param(
        "I ranked Techworld first on price and delivery, but their compliance score is lower.",
        id="ranking-justification",
    ),
    pytest.param(
        "I couldn't retrieve that. I've raised it with the team.",
        id="the-safe-reply-itself-must-pass",
    ),
]


@pytest.mark.parametrize("text", MUST_SURVIVE)
def test_legitimate_answers_are_untouched(text):
    violations = osafe.inspect(text, prose=True)
    assert violations == [], f"false positive on a legitimate answer: {violations!r}"
    assert osafe.enforce(text) == text, "gate rewrote a clean answer"


# --------------------------------------------------------------------------------------
# Safe failure — a violation must NOT produce a redacted fragment that still shows the
# shape of the internals. The whole answer goes.
# --------------------------------------------------------------------------------------


def test_violation_replaces_the_whole_answer_not_a_fragment():
    leaky = (
        "Your spend comes from proc.bp_invoice_trgt. Everything else about your deal is fine "
        "and the totals reconcile."
    )
    out = osafe.enforce(leaky)

    assert out == osafe.SAFE_REPLY
    # No fragment of the original survives — not even the clean second sentence, and above
    # all not a "SELECT * FROM ███" style redaction.
    assert "proc." not in out
    assert "bp_invoice_trgt" not in out
    assert "reconcile" not in out
    assert "█" not in out and "[REDACTED]" not in out


def test_matched_text_is_never_returned_to_the_user():
    """The violation record carries the match for the LOG. It must not reach the reply."""
    v = osafe.inspect("stored in proc.process_monitor")[0]
    assert "process_monitor" in v.match  # available to the log
    assert "process_monitor" not in osafe.enforce("stored in proc.process_monitor")


# --------------------------------------------------------------------------------------
# Payload scrubbing — the boundary backstop walks a whole response body.
# --------------------------------------------------------------------------------------


def test_scrub_payload_replaces_prose_fields_and_leaves_data_alone():
    body = {
        "reply": "It failed because proc.process_monitor never got the row.",
        "supplier_name": "Invoice Ltd",
        "total_amount": 12400.5,
        "deal_id": 7,
        "nested": {"rationale": "The DataExtractionAgent could not parse it."},
        "lines": [{"description": "Pipeline fittings, 40mm"}],
    }
    out = osafe.scrub_payload(body)

    assert out["reply"] == osafe.SAFE_REPLY
    assert out["nested"]["rationale"] == osafe.SAFE_REPLY
    # Data survives byte-identical — including the supplier whose name is a table name and
    # the line item whose description contains a register word.
    assert out["supplier_name"] == "Invoice Ltd"
    assert out["total_amount"] == 12400.5
    assert out["deal_id"] == 7
    assert out["lines"][0]["description"] == "Pipeline fittings, 40mm"


def test_scrub_payload_catches_identifiers_in_non_prose_fields_too():
    """A table name in some random field is still a leak, even if it is not prose."""
    out = osafe.scrub_payload({"note_to_self": "see proc.bp_invoice_trgt"})
    assert "proc.bp_invoice_trgt" not in str(out)


def test_error_detail_degrades_to_plain_english():
    body = {"detail": 'relation "proc.cat_product_mapping" does not exist'}
    out = osafe.scrub_payload(body)
    assert out["detail"] == osafe.SAFE_REPLY
    assert "proc." not in out["detail"]
