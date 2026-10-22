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


def test_blocked_structured_field_is_withheld_not_replaced_with_prose():
    """A blocked identifier must not become a sentence.

    Live regression: POST /workflows/opportunities returned
    user_id="I couldn't retrieve that. I've raised it with the team." The run's user_id was
    the model name (settings.script_user), which correctly trips the model_name rule — but
    substituting the conversational reply turned an identifier into prose that a consumer
    could compare against or persist.
    """
    out = osafe.scrub_payload({"user_id": "AgentNick"})
    assert out["user_id"] == osafe.SAFE_FIELD
    assert out["user_id"] != osafe.SAFE_REPLY
    assert "AgentNick" not in str(out)


def test_prose_fields_still_get_the_sentence():
    """The structured-field change must not weaken the prose contract."""
    out = osafe.scrub_payload({"answer": "have a look in proc.bp_invoice_trgt"})
    assert out["answer"] == osafe.SAFE_REPLY


def test_clean_structured_fields_are_untouched():
    """Withholding applies only to values that actually violate."""
    body = {"user_id": "system", "status": "completed", "workflow_id": "abc-123"}
    assert osafe.scrub_payload(body) == body


def test_error_detail_degrades_to_plain_english():
    body = {"detail": 'relation "proc.cat_product_mapping" does not exist'}
    out = osafe.scrub_payload(body)
    assert out["detail"] == osafe.SAFE_REPLY
    assert "proc." not in out["detail"]


# --------------------------------------------------------------------------------------
# The email tracking marker. Every stored draft ends with a hidden comment the reply matcher
# reads, and its last part is "TOKEN:<12 hex>". The credential rule read that as a leaked
# secret and replaced the whole email with the apology sentence: on 2026-10-11, 42 of the 47
# drafts in the test corpus came back from the API unreadable. The token is the product's
# own marker and goes to the supplier inside the email; it is not a secret.
# --------------------------------------------------------------------------------------

def _marked(body: str, **kw) -> str:
    from utils.email_markers import attach_hidden_marker

    marked, _ = attach_hidden_marker(body, **kw)
    assert "TOKEN:" in marked  # the precondition: the marker really carries a token
    return marked


EMAIL = "<p>Dear Nadia,</p><p>We can offer 455.00 GBP per unit. Please reply by 17 October 2026.</p>"


def test_a_draft_with_its_tracking_marker_is_served_as_written():
    body = _marked(EMAIL, supplier_id="SUP-TestSuppliesLtd", unique_id="PROC-WF-4235C7CFA99F")

    assert osafe.scrub_payload({"body": body}, where="/agents/execute") == {"body": body}


def test_the_marker_with_a_run_id_is_served_as_written():
    body = _marked(EMAIL, supplier_id="SUP-A", unique_id="PROC-WF-4235C7CFA99F", run_id="run-7")

    assert osafe.is_safe(body)


@pytest.mark.parametrize("leak", [
    "token: sk_live_9f8a7b6c5d4e",
    "TOKEN:C2E75A83D69C",                                   # the bare shape, outside a marker
    "<!-- TOKEN:C2E75A83D69C -->",                          # a comment, but not the marker
    "<!-- PROCWISE_MARKER:TRACKING:PROC-WF-1|TOKEN:sk_live_9f8a7b6c -->",   # not 12 hex
    "<!-- PROCWISE_MARKER:TRACKING:PROC-WF-1|TOKEN:C2E75A83D69C|password=hunter2 -->",
    "<!-- PROCWISE_MARKER:TRACKING:PROC-WF-1|TOKEN:C2E75A83D69C --> token: C2E75A83D69C",
])
def test_a_token_anywhere_else_is_still_a_credential(leak):
    kinds = {v.kind for v in osafe.inspect(EMAIL + "\n\n" + leak)}

    assert "credential" in kinds


def test_the_marker_excuses_only_its_token_not_the_email_around_it():
    body = _marked(EMAIL + "<p>Rows are in proc.bp_supplier.</p>", unique_id="PROC-WF-4235C7CFA99F")

    assert osafe.enforce(body) == osafe.SAFE_REPLY


def test_a_leak_inside_the_marker_is_still_caught():
    body = EMAIL + "\n\n<!-- PROCWISE_MARKER:TRACKING:AWS_SECRET_ACCESS_KEY|TOKEN:C2E75A83D69C -->"

    assert not osafe.is_safe(body)


# --------------------------------------------------------------------------------------
# False positives that replaced harmless answers with the apology sentence (live QA, 2026-10-18).
# D5: "repository" in a sentence about the policy repository. F2: the tail of a date or a
# document reference read as an internal route. Both lists are pinned.
# --------------------------------------------------------------------------------------

HARMLESS = [
    pytest.param("Approval limits are listed in the policy repository under Delegation of Authority.", id="d5-policy-repository"),
    pytest.param("The contract repository shows this agreement expired on 03/14/2026.", id="f2-us-date"),
    pytest.param("It was signed on 06/15/2024 and ran to 14/03/2026, so it has already expired.", id="f2-two-dates"),
    pytest.param("Supplier reference SUP/2024/PS/0847 was onboarded in 2024.", id="document-reference"),
    pytest.param("Terms apply from 1/4/2025 and renew yearly; see schedule 2/3 and the and/or clause.", id="short-slashes"),
    pytest.param("Invoice INV/2026/0042 is due 30/09/2026.", id="invoice-reference"),
]


@pytest.mark.parametrize("text", HARMLESS)
def test_ordinary_procurement_text_is_not_withheld(text):
    assert osafe.inspect(text) == [], osafe.inspect(text)
    assert osafe.enforce(text) == text


STILL_LEAKS = [
    pytest.param("Clone the git repository and read the source code.", "mechanism", id="git-repository"),
    pytest.param("It is handled by InvoiceRepository.", "code_identifier", id="repository-class"),
    pytest.param("Use GET /spendiq/invoices to list them.", "route", id="verb-route"),
    pytest.param("There is no /spendiq/invoices endpoint.", "route", id="route-after-space"),
    pytest.param("See (/data-integration/presigned-url) for details.", "route", id="route-in-brackets"),
    pytest.param("Call 'POST /workflows/ask/stream' first.", "route", id="quoted-verb-route"),
]


@pytest.mark.parametrize("text,kind", STILL_LEAKS)
def test_the_narrowed_rules_still_catch_real_leaks(text, kind):
    assert kind in {v.kind for v in osafe.inspect(text)}


def test_every_block_names_the_rule_that_fired(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger=osafe.log.name):
        osafe.enforce("Call POST /data-integration/presigned-url now.", where="unit")
    line = next(r.getMessage() for r in caplog.records if "BLOCKED" in r.getMessage())
    assert "rules=" in line and "route:" in line and "reason=" in line and "unit" in line
    assert all(v.rule for v in osafe.inspect("Call POST /data-integration/presigned-url now. See src/api/main.py"))


def test_the_ask_guard_replies_pass_the_output_filter():
    """The fixed replies the input guardrail sends must not themselves be withheld."""
    from src.services import ask_input_guard, ask_output_check
    for reply in list(ask_input_guard.REPLIES.values()) + [ask_output_check.REPLY]:
        assert osafe.inspect(reply) == [], reply


def test_a_procurement_pipeline_is_product_english_but_a_processing_pipeline_is_mechanism():
    ok = ("The contract pipeline for the next six months: three contracts fall due, two auto-renew; "
          "the savings pipeline holds four open levers and the sales pipeline two bids.")
    assert not [v for v in osafe.inspect(ok) if v.kind == "mechanism"], osafe.inspect(ok)
    for bad in ("Your document goes through the extraction pipeline first.", "The pipeline stages run nightly.",
                "It sits in the data pipeline until promoted."):
        assert [v for v in osafe.inspect(bad) if v.kind == "mechanism"], bad



def test_the_retry_instruction_asks_for_plain_facts_not_a_tour_of_the_screen():
    from services import output_safety as OS
    low = OS.RETRY_INSTRUCTION.lower()
    assert "the screen they are on" not in low and "button they press" not in low
    assert "do not describe screens" in low and "same facts" in low
