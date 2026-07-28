#!/usr/bin/env python3
# scripts/demo_email_assistant.py
"""Live, end-to-end proof of the policy-gated email assistant, against ``bp_testdb``.

Task 11 of the 2026-07-28 email-assistant-policy-gated plan. Ten things are proven
here, in order, each with real output:

 1. both migrations apply cleanly to bp_testdb, and re-running them changes nothing;
 2. the governed policy resolves by slug, with auto_reply_intents empty;
 3. resolve_authority() returns governed=True, reading the limit from the existing
    approval_threshold policy -- asserted BEFORE any decision is reported;
 4. a realistic seeded supplier reply (both sides: our draft AND their response) is
    decided, with amounts chosen so the value-at-stake arithmetic can run;
 5. the decision, its rationale and its evidence (every fact sourced);
 6. the escalation on the Action Centre queue, GET /decisions;
 7. GET /decisions/email-reply/{id}/message, and a wrong-subject-type 404;
 8. POST .../action=reject records the human call and clears the queue card;
 9. an attachments round-trip (upload two, one duplicate filename, list, delete one);
10. the fail-closed proof: deactivate the policy, re-decide, restore, confirm one row.

Nothing here sends a real email (SMTP is out of scope by explicit instruction) and
nothing here writes to a real S3 bucket (the attachment upload's S3 call is stubbed;
the database write is real). ``bp_testdb`` is the target -- never ``bp_sqldb``.

Run with:  venv/bin/python scripts/demo_email_assistant.py

This script does not delete its own seed rows by default (see the final banner for
exactly what it leaves behind and why) -- it DOES restore the governed policy to
its shipped state (active, auto_reply_intents empty) no matter how it exits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
# Same order tests/conftest.py uses (repo root ends up ahead of src/). Getting this
# backwards is exactly the footgun that file's own comment warns about: importing
# "api.routers.decisions" and "src.api.routers.workflows" side by side binds two
# different module objects for the same file if the order flips mid-run.
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(REPO_ROOT / ".env"))

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

# ---------------------------------------------------------------------------
# 0. The one guard that matters most: this MUST be bp_testdb, never bp_sqldb.
# ---------------------------------------------------------------------------
DB_NAME = os.getenv("DB_NAME")
if DB_NAME != "bp_testdb":
    print(
        f"REFUSING TO RUN: DB_NAME is '{DB_NAME}', not 'bp_testdb'. This script is "
        "authorised for bp_testdb only -- it will not touch anything else."
    )
    raise SystemExit(2)

DB = dict(
    host=os.getenv("DB_HOST"),
    dbname=DB_NAME,
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT", "5432"),
)


def db_connect():
    return psycopg2.connect(**DB)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fetch_one(sql: str, params: tuple = ()) -> Optional[tuple]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def fetch_all(sql: str, params: tuple = ()) -> list:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def execute(sql: str, params: tuple = ()) -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


# ===========================================================================
# STEP 1 -- apply both migrations to bp_testdb, then again, and prove
# idempotence.
# ===========================================================================
MIGRATIONS = [
    REPO_ROOT / "deploy/sql/2026-07-28_bp_policy_email_reply_autonomy.sql",
    REPO_ROOT / "deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql",
]


def run_migration(path: Path) -> Tuple[bool, str]:
    env = dict(os.environ)
    env.update(
        PGHOST=DB["host"], PGDATABASE=DB["dbname"], PGUSER=DB["user"],
        PGPASSWORD=DB["password"], PGPORT=str(DB["port"]),
    )
    proc = subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-f", str(path)],
        env=env, capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    out = (proc.stdout or "") + (proc.stderr or "")
    return ok, out.strip()


def step1_migrations() -> None:
    banner("STEP 1: apply both migrations to bp_testdb, and prove idempotence")
    for path in MIGRATIONS:
        assert path.exists(), f"migration file missing: {path}"

    print("-- first application --")
    for path in MIGRATIONS:
        ok, out = run_migration(path)
        print(f"[{path.name}] rc_ok={ok}\n{out}\n")
        if not ok:
            print("MIGRATION FAILED. Stopping -- nothing further is trustworthy.")
            raise SystemExit(1)

    row = fetch_one(
        """SELECT policy_id, version, policy_status,
                  policy_details->'rules'->'auto_reply_intents'
             FROM proc.bp_policy
            WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
              AND policy_status=1"""
    )
    assert row, "EmailReplyAutonomyPolicy is not active after the first apply"
    print(f"EmailReplyAutonomyPolicy: policy_id={row[0]} version={row[1]} "
          f"status={row[2]} auto_reply_intents={row[3]}")
    version_after_first = row[1]

    col = fetch_one(
        """SELECT 1 FROM information_schema.columns
            WHERE table_schema='proc' AND table_name='draft_rfq_emails'
              AND column_name='attachments'"""
    )
    assert col, "draft_rfq_emails.attachments column is missing after the first apply"
    print("draft_rfq_emails.attachments column: present")

    print("\n-- second application (idempotence check) --")
    for path in MIGRATIONS:
        ok, out = run_migration(path)
        print(f"[{path.name}] rc_ok={ok}\n{out}\n")
        if not ok:
            print("RE-APPLYING FAILED -- the migration is not safely re-runnable.")
            raise SystemExit(1)

    active_rows = fetch_all(
        """SELECT policy_id, version, policy_details->'rules'->'auto_reply_intents'
             FROM proc.bp_policy
            WHERE policy_type='email_autonomy' AND policy_status=1"""
    )
    assert len(active_rows) == 1, (
        f"expected exactly ONE active email_autonomy row after re-applying, "
        f"found {len(active_rows)}"
    )
    print(f"Active email_autonomy rows after re-applying: {len(active_rows)} "
          f"(policy_id={active_rows[0][0]}, version={active_rows[0][1]}, "
          f"auto_reply_intents={active_rows[0][2]})")

    col2 = fetch_one(
        """SELECT count(*) FROM information_schema.columns
            WHERE table_schema='proc' AND table_name='draft_rfq_emails'
              AND column_name='attachments'"""
    )
    print(f"draft_rfq_emails.attachments column count after re-applying: {col2[0]} "
          "(ADD COLUMN IF NOT EXISTS is a true no-op the second time)")

    print(
        "\nHONEST NOTE ON IDEMPOTENCE: the attachments migration is a genuine no-op "
        "on re-run (IF NOT EXISTS). The policy migration is idempotent in EFFECT -- "
        "there is still exactly one active row, and its rules are unchanged -- but "
        "NOT byte-for-byte idempotent: its own ON CONFLICT ... DO UPDATE bumps "
        f"`version` and `last_modified_date` on every re-run (version was "
        f"{version_after_first} after the first apply, {active_rows[0][1]} after the "
        "second). That is a property of how the migration is authored (an upsert, "
        "not a guarded no-op), not a bug this script papers over."
    )


# ===========================================================================
# STEP 2 & 3 -- the governed policy resolves by slug; resolve_authority()
# returns governed=True with the limit read from approval_threshold. Asserted
# BEFORE any decision is reported.
# ===========================================================================
def step2_3_authority() -> Dict[str, Any]:
    banner("STEP 2: the governed policy resolves by slug")
    from engines.policy_engine import PolicyEngine

    # connection_factory is a REAL callable reading bp_testdb -- NOT PolicyEngine()
    # with no arguments, which resolves to no connection factory at all and silently
    # returns zero policies. That is the exact trap the brief warned about.
    engine = PolicyEngine(connection_factory=db_connect)
    print(f"PolicyEngine loaded {len(engine.list_policies())} policies from bp_testdb")
    assert len(engine.list_policies()) >= 10, (
        "PolicyEngine loaded suspiciously few policies -- it may not really be "
        "reading bp_testdb. Stopping rather than presenting a hollow result."
    )

    policy = engine.get_policy("email_reply_autonomy")
    assert policy, "the governed policy did not resolve by its slug"
    rules = policy.get("details", {}).get("rules", {})
    print(f"resolved policy: {policy['raw_row'].get('policy_name')} "
          f"(policy_id={policy['raw_row'].get('policy_id')})")
    print(f"auto_reply_intents = {rules.get('auto_reply_intents')}")
    assert rules.get("auto_reply_intents") == [], (
        "auto_reply_intents is not empty -- the shipped default has changed"
    )

    banner("STEP 3: resolve_authority() -- governed=True, limit from approval_threshold")
    from src.services.governance_tools.authority import resolve_authority

    authority = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    print(json.dumps(authority, indent=2, default=str))

    # THE assertion the brief calls out by name: if this is False, the demonstration
    # is broken, not the system. Say so loudly and stop, rather than continuing as
    # though an ungoverned fail-closed result were the real state of bp_testdb.
    if not authority.get("governed"):
        print(
            "\n*** DEMONSTRATION HARNESS FAILURE, NOT A SYSTEM FINDING ***\n"
            f"resolve_authority() returned governed=False ({authority.get('reason')}) "
            "immediately after the migration confirmed the policy is active. That "
            "means THIS SCRIPT is not really reading bp_testdb the way it should be, "
            "not that the system is failing closed for a real reason. Stopping."
        )
        raise SystemExit(1)

    assert authority["limit_gbp"] == "10000" and authority["limit_currency"] == "GBP", (
        f"expected the limit to come from ApprovalThresholdPolicy (10000 GBP), got "
        f"{authority['limit_gbp']} {authority['limit_currency']}"
    )
    print("\nASSERTED: governed=True, limit=10000 GBP (read from approval_threshold), "
          "auto_intents=[] -- proceeding to a real decision on this basis.")
    return {"engine": engine, "authority": authority}


# ===========================================================================
# STEP 4 -- seed BOTH sides of a realistic supplier reply, with amounts that
# put real money at stake, and decide it.
# ===========================================================================
DEMO_WORKFLOW_ID = "demo-email-assistant-2026-07-28"
DEMO_SUPPLIER_ID = "Northfield Laboratory Supplies Ltd"
DEMO_UNIQUE_ID = f"{DEMO_WORKFLOW_ID}-northfield-labs"
# Our last counter to the supplier. The reply below quotes back a HIGHER number,
# 36,500 GBP away from it -- comfortably over the 10,000 GBP governed limit, so if
# the classified intent ever reaches the money gate, the arithmetic has something
# real to test and a real reason to escalate on amount, not merely on intent.
DEMO_COUNTER_PRICE = Decimal("96000.00")
DEMO_REPLY_PRICE = Decimal("132500.00")
# An unambiguous price counter-offer -- the realistic case this policy exists for.
# A live classification of this body would be expected to read as 'price_change',
# which is on escalate_intents by default, so the ESCALATE-LIST gate (not the
# default-deny gate, and not the money gate) would be the one to fire if the
# classifier returns at all. The quoted sentence is what the classifier must
# ground its answer in.
DEMO_REPLY_BODY = (
    "Thank you for your offer. Following a review of current material costs, we "
    "would like to counter at GBP 132,500.00 for the full framework, an increase "
    "over your proposed GBP 96,000.00. Please let us know if this is acceptable."
)


def step4_seed_and_decide(engine, authority: Dict[str, Any]) -> Dict[str, Any]:
    banner("STEP 4: seed both sides of a realistic supplier reply")

    # Idempotent re-seed: a prior run of this script leaves rows behind on purpose
    # (see the final banner), but re-running must not accumulate a second, third,
    # fourth copy of the same demo thread every time. Clear anything from an
    # earlier run on this exact unique_id first.
    execute("DELETE FROM proc.bp_decision WHERE subject_id=%s", (DEMO_UNIQUE_ID,))
    execute("DELETE FROM proc.supplier_response WHERE unique_id=%s", (DEMO_UNIQUE_ID,))
    execute("DELETE FROM proc.draft_rfq_emails WHERE unique_id=%s", (DEMO_UNIQUE_ID,))

    payload = json.dumps({
        "round": 1,
        "metadata": {"round": 1, "counter_price": float(DEMO_COUNTER_PRICE)},
        "body": "<li>Our target positioning: &#163;96,000.00</li>",
    })
    draft_row = fetch_one(
        """INSERT INTO proc.draft_rfq_emails
               (rfq_id, supplier_id, supplier_name, subject, body, created_on, sent,
                sender, payload, workflow_id, run_id, unique_id, mailbox)
           VALUES (%s,%s,%s,%s,%s, now(), true, %s, %s::jsonb, %s, %s, %s, %s)
           RETURNING id""",
        ("RFQ-DEMO-01", DEMO_SUPPLIER_ID, DEMO_SUPPLIER_ID,
         "RFQ Round 2 -- Analytical Instruments Framework",
         "Our target positioning: GBP 96,000.00 for the full framework.",
         "procurement@beyondprocwise.example", payload,
         DEMO_WORKFLOW_ID, "demo-run-01", DEMO_UNIQUE_ID,
         "procurement@beyondprocwise.example"),
    )
    draft_id = draft_row[0]
    print(f"seeded proc.draft_rfq_emails.id={draft_id} unique_id={DEMO_UNIQUE_ID} "
          f"payload.metadata.counter_price={DEMO_COUNTER_PRICE}")

    reply_row = fetch_one(
        """INSERT INTO proc.supplier_response
               (workflow_id, unique_id, supplier_id, supplier_email,
                response_subject, response_text, response_from, round_number,
                match_confidence, price, currency, payment_terms, lead_time,
                received_time, processed, created_at)
           VALUES (%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s,%s, now(), false, now())
           RETURNING id""",
        (DEMO_WORKFLOW_ID, DEMO_UNIQUE_ID, DEMO_SUPPLIER_ID,
         "sales@northfield-labs.example",
         "RE: RFQ Round 2 -- Analytical Instruments Framework",
         DEMO_REPLY_BODY, "sales@northfield-labs.example", 1,
         Decimal("0.92"), DEMO_REPLY_PRICE, "GBP", "45 Days", 28),
    )
    reply_id = reply_row[0]
    print(f"seeded proc.supplier_response.id={reply_id} price={DEMO_REPLY_PRICE} "
          f"GBP, currency=GBP, round_number=1")
    print(f"value at stake if the money gate runs: "
          f"|{DEMO_REPLY_PRICE} - {DEMO_COUNTER_PRICE}| = "
          f"{abs(DEMO_REPLY_PRICE - DEMO_COUNTER_PRICE)} GBP, against a governed "
          f"limit of {authority['limit_gbp']} {authority['limit_currency']}")

    banner("STEP 4b: temporarily widen auto_reply_intents to reach the money gate")
    print(
        "The shipped default ships auto_reply_intents EMPTY, by design: on day one "
        "every reply escalates, and every possible intent is denied at the "
        "'not one of the kinds of reply...' gate before the value-at-stake "
        "arithmetic ever runs. To prove that arithmetic actually executes (not "
        "merely that some other gate fired first), the policy is widened here to "
        "allow the ROUTINE intents (never the consequential ones already on "
        "escalate_intents) for this ONE decision, then restored immediately "
        "afterwards -- a runtime policy edit, not a new migration, mirroring what "
        "a person would do on the Policies screen."
    )
    routine_intents = json.dumps([
        "acknowledge", "confirm_receipt", "request_missing_document",
        "chase_no_response", "clarify_lead_time", "out_of_office",
    ])
    execute(
        """UPDATE proc.bp_policy
              SET policy_details = jsonb_set(policy_details, '{rules,auto_reply_intents}', %s::jsonb)
            WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
              AND policy_status=1""",
        (routine_intents,),
    )
    engine.reload_policies()
    from src.services.governance_tools.authority import resolve_authority
    widened_authority = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    print(f"widened auto_intents = {widened_authority['auto_intents']}")

    decision = None
    timed_out = False
    gate_note = ""
    try:
        from engines.decision_engine import DecisionEngine

        nick = _DemoNick(engine)
        eng = DecisionEngine(nick)

        decision, timed_out = _decide_with_hard_timeout(
            eng, reply_id, widened_authority, timeout_s=90,
        )
    finally:
        # RESTORED NO MATTER WHAT HAPPENED ABOVE -- including a timeout, an
        # exception, or an assertion failure. Leaving the policy widened would be
        # exactly the kind of ambient state change this task explicitly warns
        # against ("do not leave a widened policy behind").
        execute(
            """UPDATE proc.bp_policy
                  SET policy_details = jsonb_set(policy_details, '{rules,auto_reply_intents}', '[]'::jsonb)
                WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
                  AND policy_status=1""",
        )
        engine.reload_policies()
        restored = engine.get_policy("email_reply_autonomy")
        restored_intents = restored.get("details", {}).get("rules", {}).get("auto_reply_intents")
        print(f"RESTORED auto_reply_intents = {restored_intents} (must be [])")
        assert restored_intents == [], "the widened policy was NOT restored -- fix before anything else"

    if timed_out:
        print(
            "\n*** THE CLASSIFIER DID NOT RETURN WITHIN 90s ***\n"
            "The one live LLM call this demonstration needed did not complete in "
            "time. Per instruction, generation parameters were NOT tuned to force "
            "it to finish faster -- that would demonstrate a configuration that is "
            "not the one production runs. The call was abandoned (its thread is a "
            "daemon thread and will not block this script's exit); no decision was "
            "recorded from it. The intent gate and the value-at-stake gate were "
            "NOT exercised in this run. See the report for what this means and "
            "what was checked instead."
        )
        return {"decision": None, "reply_id": reply_id, "draft_id": draft_id,
                "timed_out": True}

    print(f"\nresolution={decision.resolution} decision={decision.decision}")
    print(f"rationale: {decision.rationale}")
    print("facts:")
    for k, v in decision.facts.items():
        print(f"  {k} = {v}")
    print("evidence (fact = value  <- source):")
    for item in decision.evidence:
        print(f"  {item.fact} = {str(item.value)[:70]!r}  <- {item.source}")

    rationale = decision.rationale
    if "could not be grounded" in rationale:
        gate_note = "GROUNDING gate (classification unusable/ungrounded) -- NOT the intent gate."
    elif "is not one of the kinds of reply" in rationale:
        gate_note = "INTENT-LIST gate (default-deny: intent not on auto_reply_intents)."
    elif "always puts in front of a person" in rationale:
        gate_note = "ESCALATE-LIST gate (intent is on escalate_intents)."
    elif "above the limit of" in rationale:
        gate_note = "VALUE-AT-STAKE gate (money gate) -- the arithmetic ran and exceeded the limit."
    elif "no authority" in rationale.lower() or "email_reply_autonomy" in rationale and "governed" not in widened_authority:
        gate_note = "AUTHORITY gate."
    else:
        gate_note = "some other escalation exit -- see rationale above."
    print(f"\nGATE THAT ACTUALLY FIRED: {gate_note}")

    return {"decision": decision, "reply_id": reply_id, "draft_id": draft_id,
            "timed_out": False, "gate_note": gate_note}


class _DemoNick:
    """The harness object DecisionEngine/BaseAgent need, built to avoid the exact
    traps the brief calls out by name:

    * ``settings`` is the REAL ``config.settings`` object (not absent), so
      ``BaseAgent.__init__`` does not raise and ``call_ollama`` gets real values
      instead of silently defaulting every GPU/runtime option.
    * ``agents = {}`` (no pre-registered agents), which is the genuine production
      shape whenever no agent happens to be registered yet -- ``_reply_caller``
      falls back to constructing a bare ``BaseAgent(self)``, exactly as it would
      in production, rather than being handed a mock that hides that path.
    * ``ollama_options()`` mirrors ``AgentNick.ollama_options()`` (``num_gpu=999``
      on a CUDA device). Confirmed live during this task's investigation: a stub
      that omits this makes ``call_ollama``'s own ``options.setdefault("num_gpu",
      int(os.getenv("OLLAMA_NUM_GPU", "1")))`` silently force the request onto
      (almost) the CPU, which is a harness defect, not a governance outcome --
      exactly the kind of thing that must not be mistaken for the system "failing
      closed for a real reason."
    """

    def __init__(self, policy_engine):
        from config.settings import settings as real_settings

        self.policy_engine = policy_engine
        self.settings = real_settings
        self.agents: Dict[str, Any] = {}

    def get_db_connection(self):
        return db_connect()

    def ollama_options(self) -> Dict[str, Any]:
        return {"num_gpu": 999, "keep_alive": "10m"}


def _decide_with_hard_timeout(engine, reply_id, authority, timeout_s: int = 90):
    """Run decide_email_reply on a daemon thread with a hard wall-clock bound.

    ``decide_email_reply`` never raises and never times out on its own -- if the
    classifier's HTTP call blocks, the whole call blocks. This script must not
    hang on that, and must not lie about it either, so the call runs on a daemon
    thread: if it does not finish in time, this function returns
    ``(None, True)`` and walks away. The daemon thread cannot block process exit
    (unlike a ThreadPoolExecutor, which python's own atexit hook joins).
    """
    box: Dict[str, Any] = {}
    done = threading.Event()

    def _run():
        try:
            box["decision"] = engine.decide_email_reply(str(reply_id), authority=authority)
        except Exception as exc:  # pragma: no cover - decide_email_reply itself never raises
            box["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_run, name="decide-email-reply", daemon=True)
    t.start()
    finished = done.wait(timeout_s)
    if not finished:
        return None, True
    if "error" in box:
        raise box["error"]
    return box["decision"], False


# ===========================================================================
# STEP 5 handled inline above (rationale + evidence are already printed as
# part of step 4's output -- they are the direct product of the same decision).
# ===========================================================================


def step4c_llm_free_escalation(engine, reply_id: int) -> Dict[str, Any]:
    """An LLM-free escalation on the SAME seeded reply, to drive steps 6-8.

    Step 4's classification-dependent decision could not complete live in this
    environment (see the timeout banner above and the report). Steps 6, 7 and 8
    are about the QUEUE and its endpoints, not about which gate produced the row
    on it -- so rather than leave them unproven, this uses the SAME real
    mechanism step 10 uses on its own: the governed policy row is genuinely
    deactivated in the database, `resolve_authority` is asked again and
    genuinely returns governed=False (not fed `authority=None` by hand -- an
    earlier version of this script did that, and it produced a decision row
    indistinguishable from a broken policy lookup; this version does not use
    that shortcut, deliberately), the decision escalates on that real authority
    gate, and the policy is reactivated immediately afterwards. One fail-closed
    mechanism in this script, not two, and anyone inspecting the resulting
    bp_decision row later sees the same story this banner tells.

    This is still NOT the intent-list gate or the money gate. Those require a
    successful classification, which did not happen -- see step 4's own
    banner. This function exists only so steps 6-8 have a genuine, live,
    DB-backed row to operate on instead of an empty queue.
    """
    banner("STEP 4c: an LLM-free escalation, to give steps 6-8 a real decision")
    print(
        "The classifier could not complete live (see above). Rather than invent "
        "a shortcut authority value, this deactivates the REAL governed policy "
        "row (the same action step 10 performs), lets resolve_authority "
        "genuinely observe that, decides on the SAME seeded reply, then "
        "reactivates the policy immediately -- so the escalation recorded below "
        "is honestly what it says it is: a real authority-gate failure caused by "
        "a real (temporary) deactivation, not a hand-fed None."
    )
    execute(
        """UPDATE proc.bp_policy SET policy_status=0
            WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
              AND policy_status=1"""
    )
    engine.reload_policies()
    from src.services.governance_tools.authority import resolve_authority
    from engines.decision_engine import DecisionEngine

    authority = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    assert authority["governed"] is False, "the policy deactivation did not take effect"
    print(f"authority after deactivating (real, observed): governed={authority['governed']}, "
          f"reason={authority['reason']!r}")

    try:
        nick = _DemoNick(engine)
        eng = DecisionEngine(nick)
        decision, timed_out = _decide_with_hard_timeout(eng, reply_id, authority, timeout_s=20)
        assert not timed_out, "the authority-gate-only path did not return in 20s"
    finally:
        execute(
            """UPDATE proc.bp_policy SET policy_status=1
                WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
                  AND policy_status=0
                  AND policy_id = (SELECT max(policy_id) FROM proc.bp_policy
                                     WHERE policy_type='email_autonomy'
                                       AND policy_name='EmailReplyAutonomyPolicy')"""
        )
        engine.reload_policies()
        active = fetch_all(
            """SELECT policy_id FROM proc.bp_policy
                WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
                  AND policy_status=1"""
        )
        print(f"reactivated immediately: active rows = {active} (must be exactly one)")
        assert len(active) == 1

    print(f"\nresolution={decision.resolution} decision={decision.decision}")
    print(f"rationale: {decision.rationale}")
    for item in decision.evidence:
        print(f"  {item.fact} = {str(item.value)[:70]!r}  <- {item.source}")
    assert decision.resolution == "escalated"

    decision_id = eng.record(
        decision, workflow_id=DEMO_WORKFLOW_ID, agent="email_drafting_agent",
        created_by="demo_email_assistant.py",
    )
    decision.decision_id = decision_id
    print(f"\nrecorded to proc.bp_decision as decision_id={decision_id}")
    return {"decision": decision, "decision_id": decision_id}


# ===========================================================================
# Minimal FastAPI app wrapping the real decisions router, for steps 6-8.
# ===========================================================================
def _build_decisions_client(nick):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.routers.decisions as decisions_router

    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = nick
    return TestClient(app)


def step6_queue(client, decision_id: Optional[int]) -> None:
    banner("STEP 6: GET /decisions?subject_type=email_reply&status=open")
    resp = client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    print(f"HTTP {resp.status_code}")
    body = resp.json()
    print(f"total={body['total']} rows_returned={len(body['data'])}")
    ids = [row["decision_id"] for row in body["data"]]
    print(f"decision_ids on the queue: {ids}")
    if decision_id is not None:
        present = decision_id in ids
        print(f"our seeded decision_id={decision_id} present on the open queue: {present}")
        assert present, "the seeded escalation is not on the open queue -- something is wrong"
    # `total` must be the true server-side count, not merely len(data).
    count_row = fetch_one(
        "SELECT count(*) FROM proc.bp_decision WHERE resolution='escalated' "
        "AND subject_type='email_reply' AND status='open'"
    )
    print(f"independently counted in the DB: {count_row[0]} "
          f"(must equal the endpoint's total={body['total']})")
    assert count_row[0] == body["total"], "endpoint total does not match the real DB count"


def step7_message(client, decision_id: Optional[int]) -> Optional[int]:
    banner("STEP 7: GET /decisions/email-reply/{id}/message, and a wrong-subject 404")
    probe_id = None
    if decision_id is not None:
        resp = client.get(f"/decisions/email-reply/{decision_id}/message")
        print(f"HTTP {resp.status_code}")
        print(json.dumps(resp.json(), indent=2, default=str))
        assert resp.status_code == 200
        assert resp.json()["available"] is True, "the supplier's message was not served"

    print("\n-- wrong subject_type must 404 without reading anything --")
    probe = fetch_one(
        """INSERT INTO proc.bp_decision
               (subject_type, subject_id, decision, resolution, rationale,
                facts, evidence, status, created_by)
           VALUES ('finding','demo-probe-not-email','escalate','escalated',
                   'demo probe row: proves subject_type scoping, not a real finding',
                   '{}'::jsonb, '[]'::jsonb, 'open', 'demo_email_assistant.py')
           RETURNING decision_id"""
    )
    probe_id = probe[0]
    resp = client.get(f"/decisions/email-reply/{probe_id}/message")
    print(f"probe decision_id={probe_id} (subject_type='finding') -> HTTP {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, default=str))
    assert resp.status_code == 404, "a finding's decision_id must 404 on the email-reply route"
    return probe_id


def step8_action_and_requeue(client, decision_id: Optional[int]) -> None:
    banner("STEP 8: POST /decisions/email-reply/{id}/action (reject) clears the queue card")
    if decision_id is None:
        print("SKIPPED -- no decision_id was produced in step 4 (classifier did not "
              "return in time). Nothing to act on.")
        return
    resp = client.post(
        f"/decisions/email-reply/{decision_id}/action",
        json={"action": "reject", "user_id": "demo_email_assistant.py"},
    )
    print(f"HTTP {resp.status_code}")
    body = resp.json()
    print(json.dumps(body, indent=2, default=str))
    assert resp.status_code == 200 and body.get("applied") is True
    assert body.get("queue_closed") is True, "the action was recorded but the queue was not closed"

    print("\n-- re-querying the open queue: the card must be gone --")
    resp2 = client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    ids = [row["decision_id"] for row in resp2.json()["data"]]
    print(f"open queue decision_ids now: {ids}")
    assert decision_id not in ids, "the rejected decision is still on the open queue"
    print(f"confirmed: decision_id={decision_id} is no longer on the open queue "
          f"(new audit row recorded at audit_decision_id={body.get('audit_decision_id')})")


# ===========================================================================
# STEP 9 -- attachments round-trip. S3 is stubbed; the DB write is real.
# ===========================================================================
class _FakeUpload:
    """Duck-types fastapi.UploadFile well enough for the real handler function."""

    def __init__(self, filename: str, content_type: str, data: bytes):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _StubDispatchServiceForAttachments:
    """Stands in for EmailDispatchService inside the attachments endpoint, so this
    demonstration never constructs a real SES-backed EmailService and never calls
    boto3.client('s3').put_object against the real 'procwisemvp' bucket. The
    _write_s3_bytes call is the ONLY thing stubbed; add_email_attachments's own
    logic (dedup keys, size limits, DB persistence) all runs for real.
    """

    _ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
    _ATTACHMENT_MAX_TOTAL_BYTES = 25 * 1024 * 1024

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick

    def _write_s3_bytes(self, s3_key: str, data: bytes, content_type) -> None:
        FAKE_S3_BUCKET[s3_key] = data


FAKE_S3_BUCKET: Dict[str, bytes] = {}


def step9_attachments(nick, unique_id: str) -> None:
    import asyncio
    import importlib

    banner("STEP 9: attachments round-trip (S3 stubbed, DB write real)")
    workflows_mod = importlib.import_module("src.api.routers.workflows")
    original_service = workflows_mod.EmailDispatchService
    workflows_mod.EmailDispatchService = _StubDispatchServiceForAttachments
    try:
        files = [
            _FakeUpload("terms.pdf", "application/pdf", b"%PDF-1.4 demo terms document"),
            _FakeUpload("terms.pdf", "application/pdf", b"%PDF-1.4 a DIFFERENT terms document"),
        ]
        result = asyncio.run(
            workflows_mod.add_email_attachments(
                unique_id, files=files, user_id="demo_email_assistant.py", agent_nick=nick,
            )
        )
        print("upload response:")
        print(json.dumps(result, indent=2, default=str))
        assert not result["rejected"], f"unexpected rejections: {result['rejected']}"
        records = result["attachments"]
        assert len(records) == 2, f"expected 2 stored attachments, got {len(records)}"
        keys = [r["s3_key"] for r in records]
        assert len(set(keys)) == 2, "the two same-named files collapsed onto one storage key"
        print(f"\ntwo files named 'terms.pdf' stored under DISTINCT keys: {keys}")
        assert set(keys) <= set(FAKE_S3_BUCKET), "a record's key was never actually written"
        assert FAKE_S3_BUCKET[keys[0]] != FAKE_S3_BUCKET[keys[1]], (
            "both keys hold the SAME bytes -- one file's content was overwritten"
        )
        print("confirmed the two keys hold DIFFERENT bytes (no overwrite).")

        print("\n-- listing: read the draft row back from the DB --")
        row = fetch_one(
            "SELECT attachments FROM proc.draft_rfq_emails WHERE unique_id=%s "
            "ORDER BY id DESC LIMIT 1",
            (unique_id,),
        )
        stored = row[0] if row else None
        print(json.dumps(stored, indent=2, default=str))
        assert stored and len(stored) == 2

        print("\n-- deleting attachment at index 0 --")
        del_result = workflows_mod.remove_email_attachment(unique_id, 0, agent_nick=nick)
        print(json.dumps(del_result, indent=2, default=str))
        assert len(del_result["attachments"]) == 1

        row2 = fetch_one(
            "SELECT attachments FROM proc.draft_rfq_emails WHERE unique_id=%s "
            "ORDER BY id DESC LIMIT 1",
            (unique_id,),
        )
        print("\nremaining state in the DB:")
        print(json.dumps(row2[0], indent=2, default=str))
        assert len(row2[0]) == 1
        assert row2[0][0]["s3_key"] == keys[1], "the WRONG attachment was removed"
        print(f"confirmed: the remaining attachment is the second upload ({keys[1]}), "
              "and its bytes were never touched by the delete.")
    finally:
        workflows_mod.EmailDispatchService = original_service


# ===========================================================================
# STEP 10 -- the fail-closed proof.
# ===========================================================================
def step10_fail_closed(engine, reply_id: Optional[int]) -> None:
    banner("STEP 10: fail-closed proof -- deactivate the policy, re-decide, restore")
    from src.services.governance_tools.authority import resolve_authority
    from engines.decision_engine import DecisionEngine

    execute(
        """UPDATE proc.bp_policy SET policy_status=0
            WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
              AND policy_status=1"""
    )
    engine.reload_policies()
    authority = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    print("authority with the policy deactivated:")
    print(json.dumps(authority, indent=2, default=str))
    assert authority["governed"] is False, "deactivating the policy did not un-govern it"

    if reply_id is not None:
        nick = _DemoNick(engine)
        eng = DecisionEngine(nick)
        # No classifier call is reached here -- decide_email_reply's authority gate
        # (gate 1) short-circuits BEFORE _classify() is ever called, so this does
        # not carry the earlier hang risk and needs no generous timeout.
        decision, timed_out = _decide_with_hard_timeout(eng, reply_id, authority, timeout_s=20)
        assert not timed_out, "even the authority-gate-only path did not return in 20s"
        print(f"\nre-decided with the policy off: resolution={decision.resolution}")
        print(f"rationale: {decision.rationale}")
        assert decision.resolution == "escalated"
        assert "email_reply_autonomy" in decision.rationale
        print("CONFIRMED: still escalates, and names the missing governed policy "
              "('email_reply_autonomy') -- correct fail-closed behaviour.")
    else:
        print("No reply_id available to re-decide (step 4 did not produce one); "
              "the authority-level fail-closed result above still stands on its own.")

    print("\n-- restoring the policy --")
    execute(
        """UPDATE proc.bp_policy SET policy_status=1
            WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
              AND policy_status=0
            AND policy_id = (SELECT max(policy_id) FROM proc.bp_policy
                               WHERE policy_type='email_autonomy'
                                 AND policy_name='EmailReplyAutonomyPolicy')"""
    )
    engine.reload_policies()
    active = fetch_all(
        """SELECT policy_id FROM proc.bp_policy
            WHERE policy_type='email_autonomy' AND policy_name='EmailReplyAutonomyPolicy'
              AND policy_status=1"""
    )
    print(f"active EmailReplyAutonomyPolicy rows after restoring: {len(active)} -> {active}")
    assert len(active) == 1, "restoring the policy did not leave exactly one active row"


# ===========================================================================
# main
# ===========================================================================
def main() -> int:
    print(f"Target database: {DB['dbname']} @ {DB['host']}")
    print(f"Run marker: {uuid.uuid4().hex[:8]}")

    step1_migrations()
    ctx = step2_3_authority()
    engine, authority = ctx["engine"], ctx["authority"]

    result = step4_seed_and_decide(engine, authority)
    decision = result.get("decision")
    decision_id: Optional[int] = None
    if decision is not None:
        # The classifier-dependent call actually completed live -- this IS the
        # real intent/money-gate decision, and it drives steps 6-8 directly.
        from engines.decision_engine import DecisionEngine

        nick = _DemoNick(engine)
        eng = DecisionEngine(nick)
        decision_id = eng.record(
            decision, workflow_id=DEMO_WORKFLOW_ID, agent="email_drafting_agent",
            created_by="demo_email_assistant.py",
        )
        decision.decision_id = decision_id
        print(f"\nrecorded to proc.bp_decision as decision_id={decision_id}")
    else:
        # The classifier did not return. Steps 6-8 still get a real, live,
        # DB-backed escalation to operate on -- see step4c's own banner for
        # exactly which gate it is (authority, not intent/money).
        c4 = step4c_llm_free_escalation(engine, result["reply_id"])
        decision_id = c4["decision_id"]

    nick = _DemoNick(engine)
    client = _build_decisions_client(nick)

    step6_queue(client, decision_id)
    probe_id = step7_message(client, decision_id)
    step8_action_and_requeue(client, decision_id)
    step9_attachments(nick, DEMO_UNIQUE_ID)
    step10_fail_closed(engine, result.get("reply_id"))

    # Clean up the throwaway probe row from step 7 (never part of the meaningful
    # seed -- it exists only to prove the subject_type scoping).
    if probe_id is not None:
        execute("DELETE FROM proc.bp_decision WHERE decision_id=%s", (probe_id,))
        print(f"\ncleaned up the step-7 probe decision row (decision_id={probe_id})")

    banner("WHAT WAS LEFT IN bp_testdb")
    print(f"proc.draft_rfq_emails: unique_id={DEMO_UNIQUE_ID} (workflow_id={DEMO_WORKFLOW_ID})")
    print(f"proc.supplier_response: id={result.get('reply_id')} on the same unique_id")
    if decision_id is not None:
        print(f"proc.bp_decision: decision_id={decision_id} (the seeded escalation) "
              "plus its 'reject' audit row from step 8")
    print("proc.bp_policy: EmailReplyAutonomyPolicy left ACTIVE with auto_reply_intents=[] "
          "(the shipped default) -- verified above, both after step 4b's widen/restore "
          "and after step 10's deactivate/restore.")
    print("These seed rows are left in place deliberately (bp_testdb, not production "
          "data) rather than deleted, so the decision above remains inspectable via "
          "GET /decisions/{decision_id} after this script exits.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
