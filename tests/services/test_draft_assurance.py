"""Draft assurance: facts from Postgres with provenance, and a draft that cannot smuggle a figure.

Every guard here has a test that makes it fail on purpose -- a check that has never
been seen red proves nothing. The family under test is the real policy row from the
migration, parsed from the SQL file, so the migration cannot drift from the code.
"""

import json
import re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services import draft_assurance as da
from src.services.draft_assurance import validator as V
from src.services.draft_assurance.facts import FactResolver, ResolvedFact, Unresolved

MIGRATION = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-07_email_family_negotiation_counter.sql"


def _family_rules():
    return json.loads(MIGRATION.read_text().split("$json$")[1])["rules"]


@pytest.fixture
def family():
    return da.parse_family(_family_rules(), version=1)


class FakeCursor:
    def __init__(self, conn):
        self.conn, self.rows = conn, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append(sql)
        params = list(params or [])
        if "to_regclass" in sql:                     # the inbound-flag lookup asks whether its table exists: here it does not,
            self.rows = [(None,)]                    # so nothing on any thread is flagged
            return
        m = re.match(r"SELECT (\w+), (\w+) FROM proc\.(\w+) WHERE (.+?)(?: ORDER BY (.+?))? LIMIT 2$", sql)
        if m:
            rid, col, table, where, order = m.groups()
            cols = re.findall(r"(\w+) = %s", where)
            rows = [r for r in self.conn.tables.get(table, [])
                    if all(r.get(c) == v for c, v in zip(cols, params))]
            if order:
                first = order.split(",")[0].split()[0]
                rows.sort(key=lambda r: r.get(first) or 0, reverse="DESC" in order.split(",")[0])
            self.rows = [(r[rid], r.get(col)) for r in rows][:2]
            return
        m = re.match(r"SELECT (\w+) FROM proc\.(\w+) WHERE (\w+) = %s$", sql)
        if m:
            col, table, rid = m.groups()
            self.rows = [(r.get(col),) for r in self.conn.tables.get(table, []) if str(r.get(rid)) == str(params[0])]
            return
        self.rows = []

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, tables, autocommit=False):
        self.tables, self.autocommit, self.log = tables, autocommit, []

    def cursor(self):
        return FakeCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


TABLES = {
    "supplier_response": [
        {"id": 1, "workflow_id": "wf-1", "supplier_id": "S-1", "round_number": 1,
         "price": Decimal("50.00"), "currency": "GBP", "lead_time": "14 days", "rfq_id": "RFQ-20260901-AAAA", "extraction_status": "confirmed"},
        {"id": 2, "workflow_id": "wf-1", "supplier_id": "S-1", "round_number": 2,
         "price": Decimal("47.50"), "currency": "GBP", "lead_time": "14 days", "rfq_id": "RFQ-20260901-AAAA", "extraction_status": "confirmed"},
    ],
    "bp_supplier": [{"supplier_id": "S-1", "contact_name_1": "Alex Morgan"}],
}
KEYS = {"workflow_id": "wf-1", "supplier_id": "S-1"}


# --- config ----------------------------------------------------------------

def test_the_migration_row_parses_into_a_family(family):
    assert family.family_id == "negotiation_counter"
    assert "supplier_current_offer" in family.required_facts
    assert family.mode == "shadow"


@pytest.mark.parametrize("field,bad", [("table", "x; DROP TABLE proc.bp_po"), ("column", "a b"), ("row_id", "id--")])
def test_a_non_identifier_in_a_fact_source_is_refused(field, bad):
    rules = _family_rules()
    rules["fact_sources"]["currency"][field] = bad
    with pytest.raises(da.FamilyConfigUnavailable):
        da.parse_family(rules)


def test_order_by_must_be_a_plain_list():
    rules = _family_rules()
    rules["fact_sources"]["currency"]["order_by"] = "id; DELETE FROM x"
    with pytest.raises(da.FamilyConfigUnavailable):
        da.parse_family(rules)


def test_the_version_is_read_from_the_engines_raw_row():
    row = {"details": {"rules": _family_rules()}, "raw_row": {"version": 7}}
    assert da.load_family(da.SLUG, SimpleNamespace(get_policy=lambda s: row)).version == 7


def test_a_missing_family_raises_rather_than_defaulting():
    engine = SimpleNamespace(get_policy=lambda slug: None)
    with pytest.raises(da.FamilyConfigUnavailable):
        da.load_family(da.SLUG, engine)


def test_an_unreadable_policy_store_raises_not_empty():
    def boom(slug):
        raise RuntimeError("store down")
    with pytest.raises(da.FamilyConfigUnavailable):
        da.load_family(da.SLUG, SimpleNamespace(get_policy=boom))


# --- resolver ----------------------------------------------------------------

def test_a_fact_carries_table_column_row_and_time(family):
    got = FactResolver(FakeConn(TABLES)).resolve(family.facts["supplier_current_offer"], KEYS)
    assert isinstance(got, ResolvedFact)
    assert got.value == Decimal("47.50")          # latest round, not round 1
    assert (got.table, got.column, got.row_id) == ("supplier_response", "price", "2")
    assert got.retrieved_at


def test_zero_matches_is_unresolved_not_guessed(family):
    got = FactResolver(FakeConn({"supplier_response": []})).resolve(family.facts["currency"], KEYS)
    assert got == Unresolved("currency", "none")


def test_several_matches_without_an_ordering_is_ambiguous(family):
    src = family.facts["supplier_contact_name"]
    tables = {"bp_supplier": [{"supplier_id": "S-1", "contact_name_1": "A"},
                              {"supplier_id": "S-1", "contact_name_1": "B"}]}
    got = FactResolver(FakeConn(tables)).resolve(src, KEYS)
    assert got.reason == "multiple"


def test_a_missing_lookup_key_is_reported(family):
    got = FactResolver(FakeConn(TABLES)).resolve(family.facts["currency"], {"supplier_id": "S-1"})
    assert got.reason == "no_lookup_key"


def test_lookups_run_in_a_read_only_block_on_autocommit_connections(family):
    conn = FakeConn(TABLES, autocommit=True)
    FactResolver(conn).resolve(family.facts["currency"], KEYS)
    assert conn.log[0] == "BEGIN READ ONLY" and conn.log[-1] == "ROLLBACK"


def test_values_are_bound_not_interpolated(family):
    conn = FakeConn(TABLES)
    FactResolver(conn).resolve(family.facts["currency"], {"workflow_id": "x' OR '1'='1", "supplier_id": "S-1"})
    assert "x'" not in conn.log[0] and "1'='1" not in conn.log[0] and "%s" in conn.log[0]


# --- conflicts: Postgres wins -------------------------------------------------

def test_postgres_overwrites_a_disagreeing_payload_and_records_it(family):
    data = {"current_offer": 49.0, "current_offer_numeric": 49.0, "currency": "GBP",
            "contact_name": "Someone Else"}
    inp = da.prepare_inputs(FakeConn(TABLES), family, data, lookup_keys=KEYS)
    assert data["current_offer"] == 47.5 and data["current_offer_numeric"] == 47.5
    assert data["contact_name"] == "Alex Morgan"
    conflict = next(c for c in inp.conflicts if c["fact"] == "supplier_current_offer")
    assert conflict["postgres"] == "47.50" and conflict["supplied"] == 49.0
    assert conflict["resolution"] == "postgres_wins"


def test_a_value_with_no_row_is_carried_and_marked_unverified(family):
    data = {"current_offer": 49.0, "currency": "GBP"}
    inp = da.prepare_inputs(FakeConn({}), family, data, lookup_keys=KEYS)
    assert inp.carried["supplier_current_offer"] == 49.0
    out = inp.finalize("Please confirm.", [], [])
    assert out["status"] == "needs_review"
    assert out["carried_unverified"]["supplier_current_offer"] == 49.0


# --- validator -----------------------------------------------------------------

def _inputs(family, **extra):
    data = {"current_offer": 47.5, "currency": "GBP", "counter_price": 44.8,
            "response_deadline": "30 October 2026", "asks": ["confirm revised pricing"],
            "reasoned_basis": {"response_deadline": ["email_thread_summary"]}, **extra}
    return da.prepare_inputs(FakeConn(TABLES), family, data, lookup_keys=KEYS)


GOOD = ("Thank you for your offer of 47.50 GBP. We propose 44.80 GBP. "
        "Please confirm by 30 October 2026?")


def test_a_clean_draft_is_verified(family):
    out = _inputs(family).finalize(GOOD, ["a@x.test"], ["a@x.test"])
    assert out["violations"] == [] and out["status"] == "verified", out


def test_an_invented_price_fails(family):
    text = GOOD.replace("44.80", "43.10")
    kinds = [v["kind"] for v in _inputs(family).check_text(text)]
    assert "ungrounded_figure" in kinds


def test_a_changed_supplier_offer_fails(family):
    text = GOOD.replace("47.50", "49.00")
    assert any(v["kind"] == "ungrounded_figure" and v["detail"] == "49.00"
               for v in _inputs(family).check_text(text))


def test_an_invented_date_and_reference_fail(family):
    kinds = {v["kind"] for v in _inputs(family).check_text(GOOD + " Reply by 12 November 2026 re PO-99887.")}
    assert {"ungrounded_date", "ungrounded_reference"} <= kinds


def test_the_walkaway_price_never_appears(family):
    inp = _inputs(family, walkaway_price=46.0)
    kinds = [v["kind"] for v in inp.check_text(GOOD + " Our limit is 46.00 GBP.")]
    assert "internal_figure_leaked" in kinds


def test_walkaway_leak_is_caught_even_when_it_equals_an_allowed_figure(family):
    # walkaway == the counter price: still must not be stated as a limit.
    inp = _inputs(family, walkaway_price=44.8)
    assert any(v["kind"] == "internal_figure_leaked" for v in inp.check_text(GOOD))


@pytest.mark.parametrize("sentence,rule", [
    ("Send funds to IBAN GB29NWBK60161331926819.", "bank_details"),
    ("Please confirm your bank details.", "bank_details"),
    ("We accept full liability for the delay.", "liability_admission"),
    ("We hereby waive our right to claim.", "rights_waiver"),
    ("We will award the contract to you.", "award_commitment"),
])
def test_forbidden_content_is_caught(family, sentence, rule):
    hits = [v for v in _inputs(family).check_text(GOOD + " " + sentence) if v["kind"] == "forbidden_content"]
    assert any(h["detail"].startswith(rule) for h in hits)


def test_an_unresolved_placeholder_fails(family):
    assert any(v["kind"] == "unresolved_placeholder"
               for v in _inputs(family).check_text(GOOD + " Reply by [date 5-7 days out]."))


def test_missing_ask_and_deadline_are_reported(family):
    kinds = [v["detail"] for v in _inputs(family).check_text("Thank you for your note.")
             if v["kind"] == "missing_required_element"]
    assert set(kinds) == {"explicit_ask", "deadline"}


# A deadline is a date or time STATED AS A DEADLINE. Any date anywhere used to count, so a draft that
# mentioned when the contract started, or a reference like "PO 12/34", passed with no deadline at all
# (found 2026-10-09 checking the judge's missed-deadline control by hand).
@pytest.mark.parametrize("text", [
    "Please reply by 6 November 2026.",
    "Could you confirm by 6 Nov?",
    "We need your answer no later than 2026-11-06.",
    "Please confirm on or before 06/11/2026.",
    "Please reply within 3 working days.",
    "Please reply by Friday.",
    "Please let us know by next Tuesday.",
    "Please confirm by close of business tomorrow.",
    "Please reply by end of the week.",
    "Please confirm by COB today.",
    "The deadline for your reply is 30 October 2026.",
    "Please reply before 30 October.",
    "Could you respond by noon on Thursday?",
    "Please respond by the 30th.",
    "We would appreciate your response by Friday, 30 October.",
])
def test_a_stated_deadline_counts(text):
    assert V.check_required(text, ["deadline"], []) == []


@pytest.mark.parametrize("text", [
    "Thank you for your offer of 9,200.00 GBP. We can agree 8,600.00 GBP.",
    "Our contract started 1 March 2025; can you accept 44.80?",
    "Please see PO 12/34 attached and confirm.",
    "Your quote dated 6 November 2026 is noted; can you improve it?",
    "We would be glad to hear from you as soon as possible.",
    "Please reply when you can.",
    "We have worked together since March 2020.",
])
def test_a_date_that_is_not_a_deadline_does_not_count(text):
    assert [v["detail"] for v in V.check_required(text, ["deadline"], [])] == ["deadline"]


def test_a_recipient_not_on_the_supplier_master_fails(family):
    out = _inputs(family).finalize(GOOD, ["attacker@evil.test"], ["a@x.test"])
    assert any(v["kind"] == "recipient_not_on_master" for v in out["violations"])
    assert out["status"] == "needs_review"


def test_a_reasoned_value_without_basis_is_an_assumption(family):
    inp = da.prepare_inputs(FakeConn(TABLES), family,
                            {"response_deadline": "30 October 2026", "counter_price": 44.8},
                            lookup_keys=KEYS)
    assert any("response_deadline" in a for a in inp.assumptions)
    assert inp.reasoned["counter_price"]["basis"] == ["supplier_current_offer"]


def test_a_made_up_basis_is_ignored(family):
    inp = da.prepare_inputs(FakeConn(TABLES), family,
                            {"response_deadline": "30 October 2026",
                             "reasoned_basis": {"response_deadline": ["a_fact_that_does_not_exist"]}},
                            lookup_keys=KEYS)
    assert inp.reasoned["response_deadline"]["basis"] == []
    assert inp.assumptions


# --- send-time recheck ---------------------------------------------------------

def test_recheck_reports_a_fact_that_moved(family):
    inp = _inputs(family)
    record = inp.finalize(GOOD, [], [])
    assert da.recheck_facts(FakeConn(TABLES), family, record) == []
    moved = json.loads(json.dumps(TABLES, default=str))
    moved["supplier_response"][1]["price"] = Decimal("45.00")
    changes = da.recheck_facts(FakeConn({**TABLES, "supplier_response": [
        TABLES["supplier_response"][0], {**TABLES["supplier_response"][1], "price": Decimal("45.00")}]}),
        family, record)
    assert [c["fact"] for c in changes] == ["supplier_current_offer"]
    assert changes[0]["was"] == "47.50" and changes[0]["now"] == "45.00"


def test_recheck_reports_a_row_that_vanished(family):
    record = _inputs(family).finalize(GOOD, [], [])
    changes = da.recheck_facts(FakeConn({}), family, record)
    assert {c["fact"] for c in changes} >= {"supplier_current_offer"}


# --- the wrapper inside the agent ------------------------------------------------

def _agent(monkeypatch, canned, tables=TABLES, policy=True):
    from agents import email_drafting_agent as module
    from src.services import supplier_contact
    agent = module.EmailDraftingAgent()
    stored = []
    monkeypatch.setattr(agent, "_draft_intelligent_negotiation_email", lambda c, d: canned)
    monkeypatch.setattr(agent, "_store_draft", lambda d: stored.append(d))
    monkeypatch.setattr(agent, "_record_learning_events", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_master_contact",
                        lambda sid: supplier_contact.SupplierContact(emails=["a@x.test"], name="Alex Morgan"))
    agent.agent_nick.get_db_connection = lambda: FakeConn(tables)
    row = {"details": {"rules": _family_rules()}, "version": 1}
    agent.agent_nick.policy_engine = SimpleNamespace(get_policy=lambda s: row if policy else None)
    return agent, stored


def _ctx(**data):
    from agents.base_agent import AgentContext
    payload = {"supplier_id": "S-1", "workflow_id": "wf-1", "recipients": ["a@x.test"],
               "supplier_name": "Acme", "current_offer": 49.0, "currency": "GBP",
               "counter_price": 44.8, "response_deadline": "30 October 2026",
               "asks": ["confirm revised pricing"],
               "reasoned_basis": {"response_deadline": ["email_thread_summary"]}, **data}
    return AgentContext(workflow_id="wf-1", agent_id="email_drafting", user_id="t", input_data=payload), payload


CANNED = "Subject: Re: Pricing\n" + GOOD


def test_agent_stores_an_assurance_record_with_provenance_and_the_conflict(monkeypatch):
    agent, stored = _agent(monkeypatch, CANNED)
    ctx, payload = _ctx()
    agent._handle_negotiation_counter(ctx, payload)
    a = stored[0]["assurance"]
    assert a["status"] == "needs_review"                       # the 49.0 vs 47.50 conflict
    assert a["facts"]["supplier_current_offer"]["row_id"] == "2"
    assert a["facts"]["supplier_current_offer"]["table"] == "supplier_response"
    assert a["conflicts"][0]["postgres"] == "47.50"
    assert stored[0]["metadata"]["assurance_status"] == "needs_review"
    assert stored[0]["metadata"]["current_offer"] == 47.5       # the draft carries Postgres' figure


def test_agent_repairs_once_then_shows_what_is_left(monkeypatch):
    agent, stored = _agent(monkeypatch, CANNED.replace("44.80", "43.10"))
    calls = []
    monkeypatch.setattr(agent, "_repair_assured_body", lambda body, failed: calls.append(failed) or None)
    ctx, payload = _ctx()
    agent._handle_negotiation_counter(ctx, payload)
    assert len(calls) == 1                                       # exactly one repair pass
    assert any(v["kind"] == "ungrounded_figure" for v in stored[0]["assurance"]["violations"])


def test_agent_uses_the_repaired_text_when_repair_fixes_it(monkeypatch):
    agent, stored = _agent(monkeypatch, CANNED.replace("44.80", "43.10"))
    monkeypatch.setattr(agent, "_repair_assured_body", lambda body, failed: body.replace("43.10", "44.80"))
    ctx, payload = _ctx()
    agent._handle_negotiation_counter(ctx, payload)
    a = stored[0]["assurance"]
    assert a["repaired"] is True
    assert not [v for v in a["violations"] if v["severity"] == "fail"]
    assert "44.80" in stored[0]["text"]


def test_agent_marks_the_draft_unassured_when_the_family_is_missing(monkeypatch):
    agent, stored = _agent(monkeypatch, CANNED, policy=False)
    ctx, payload = _ctx()
    out = agent._handle_negotiation_counter(ctx, payload)
    assert out.status.name == "SUCCESS"                          # the draft still goes out
    assert stored[0]["assurance"]["status"] == "unassured"
    assert "no policy named" in stored[0]["assurance"]["reason"]


# --- from_decision and from_prompt -------------------------------------------------

FREE_PROMPT = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-07_email_family_free_prompt.sql"


def _free_prompt_rules():
    return json.loads(FREE_PROMPT.read_text().split("$json$")[1])["rules"]


def _wrapped(monkeypatch, chat_reply, tables=TABLES, families=("negotiation_counter", "free_prompt")):
    """An agent whose model returns ``chat_reply`` and whose policy store serves the real rows."""
    from agents import email_drafting_agent as module
    from src.services import supplier_contact
    agent = module.EmailDraftingAgent()
    seen = []
    monkeypatch.setattr(module, "_chat", lambda m, sys_, user, **k: seen.append(user) or chat_reply)
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20260101")
    monkeypatch.setattr(agent, "_master_contact",
                        lambda sid: supplier_contact.SupplierContact(emails=["a@x.test"], name="Alex Morgan"))
    agent.agent_nick.get_db_connection = lambda: FakeConn(tables)
    rows = {"email_family_negotiation_counter": {"details": {"rules": _family_rules()}, "version": 1},
            "email_family_free_prompt": {"details": {"rules": _free_prompt_rules()}, "version": 1}}
    agent.agent_nick.policy_engine = SimpleNamespace(
        get_policy=lambda s: rows.get(s) if s.replace("email_family_", "") in families else None)
    return agent, seen


DECISION = {"supplier_id": "S-1", "supplier_name": "Acme", "workflow_id": "wf-1", "to": "a@x.test",
            "current_offer": 49.0, "currency": "GBP", "counter_price": 44.8,
            "asks": ["Confirm revised pricing"]}


def test_both_new_family_rows_parse():
    assert da.parse_family(_free_prompt_rules()).family_id == "free_prompt"


def test_from_decision_shows_the_model_postgres_figure_not_the_payload_one(monkeypatch):
    agent, seen = _wrapped(monkeypatch, "Subject: Re\n" + GOOD)
    draft = agent.from_decision(dict(DECISION))
    assert '"current_offer": 47.5' in seen[0] and "49.0" not in seen[0]
    assert draft["assurance"]["conflicts"][0]["postgres"] == "47.50"
    assert draft["metadata"]["current_offer"] == 47.5


def test_from_decision_flags_an_invented_price_and_repairs_once(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Re\n" + GOOD.replace("44.80", "43.10"))
    calls = []
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: calls.append(f) or None)
    draft = agent.from_decision(dict(DECISION))
    assert len(calls) == 1
    assert any(v["kind"] == "ungrounded_figure" for v in draft["assurance"]["violations"])


def test_from_decision_checks_a_premade_negotiation_message_too(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "unused")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: None)
    draft = agent.from_decision({**DECISION, "negotiation_message": GOOD.replace("47.50", "52.00")})
    assert any(v["detail"] == "52.00" for v in draft["assurance"]["violations"])


def test_from_decision_without_a_family_still_returns_a_draft_marked_unassured(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Re\n" + GOOD, families=())
    draft = agent.from_decision(dict(DECISION))
    assert draft["assurance"]["status"] == "unassured"


def test_from_prompt_reports_figures_taken_from_the_persons_words_as_unverified(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Chair order\nPlease quote for 25 chairs.")
    draft = agent.from_prompt("Ask Acme to quote for 25 chairs", context={"supplier_id": "S-1", "workflow_id": "wf-1",
                                                                         "recipients": ["a@x.test"]})
    a = draft["assurance"]
    assert a["unverified_figures"] == ["25"] and a["status"] == "needs_review"
    assert not [v for v in a["violations"] if v["severity"] == "fail"]


def test_from_prompt_flags_a_figure_the_person_never_gave(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Chairs\nPlease quote for 40 chairs by 12 March 2027.")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: None)
    draft = agent.from_prompt("Ask Acme to quote for 25 chairs", context={"supplier_id": "S-1", "workflow_id": "wf-1",
                                                                         "recipients": ["a@x.test"]})
    kinds = {v["kind"] for v in draft["assurance"]["violations"]}
    assert {"ungrounded_figure", "ungrounded_date"} <= kinds


def test_from_prompt_refuses_bank_details_and_a_stranger_recipient(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Pay\nPlease confirm your bank details for 25 chairs.")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: None)
    draft = agent.from_prompt("Ask about 25 chairs", context={"supplier_id": "S-1", "workflow_id": "wf-1",
                                                              "recipients": ["stranger@evil.test"]})
    kinds = {v["kind"] for v in draft["assurance"]["violations"]}
    assert {"forbidden_content", "recipient_not_on_master"} <= kinds


def test_from_prompt_never_states_a_walkaway_price_even_if_the_person_typed_it(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Price\nOur limit is 46.00 GBP.")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: None)
    draft = agent.from_prompt("Counter at 44.80", context={"supplier_id": "S-1", "workflow_id": "wf-1",
                                                           "recipients": ["a@x.test"], "walkaway_price": 46.0})
    assert any(v["kind"] == "internal_figure_leaked" for v in draft["assurance"]["violations"])


def test_from_prompt_without_a_family_is_unassured_not_blocked(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Hi\nPlease reply.", families=())
    draft = agent.from_prompt("say hi", context={"supplier_id": "S-1", "recipients": ["a@x.test"]})
    assert draft["assurance"]["status"] == "unassured"
    assert "Please reply" in draft["text"]


def test_the_round_number_is_not_reported_as_an_unverified_figure(family):
    inp = _inputs(family, round=2)
    out = inp.finalize(GOOD + " This is round 2.", ["a@x.test"], ["a@x.test"])
    assert out["unverified_figures"] == []


def test_from_prompt_uses_the_repaired_text_and_clears_the_violation(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Chairs\nPlease quote for 40 chairs.")
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: b.replace("40", "25"))
    draft = agent.from_prompt("Ask Acme to quote for 25 chairs", context={"supplier_id": "S-1", "workflow_id": "wf-1",
                                                                         "recipients": ["a@x.test"]})
    assert draft["assurance"]["repaired"] is True
    assert not [v for v in draft["assurance"]["violations"] if v["severity"] == "fail"]
    assert "25 chairs" in draft["text"] and "40" not in draft["text"]
    assert "25 chairs" in draft["html"] and "40" not in draft["html"]


def test_from_decision_uses_the_repaired_text_and_clears_the_violation(monkeypatch):
    agent, _ = _wrapped(monkeypatch, "Subject: Re\n" + GOOD.replace("44.80", "43.10"))
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f: b.replace("43.10", "44.80"))
    draft = agent.from_decision({**DECISION, "response_deadline": "30 October 2026"})
    assert draft["assurance"]["repaired"] is True
    assert not [v for v in draft["assurance"]["violations"] if v["severity"] == "fail"]
    assert "44.80" in draft["text"] and "43.10" not in draft["text"]
