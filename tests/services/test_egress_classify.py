"""1.20 step (c): what is in an outbound payload is classified before it leaves, and a rule may cap it.

Four cheap detectors tag the text an external call carries (URL query string and body): email
addresses, money amounts, document references (invoice / PO / quote numbers) and long verbatim text.
Tags map to a level: public (nothing found) < commercial (amounts, references) < confidential
(email addresses, long passages). An allow rule may say ``max_classification``; a payload above it
WOULD be denied (still shadow). The tags are recorded on the event row's ``classification`` column.
Local destinations are never classified: the model host must not pay for it.
"""
from __future__ import annotations

import pytest

from src.services import egress, egress_log
from src.services import egress_classify as ec
from src.services import egress_policy as ep
from src.services.egress import Purpose


# ---- detectors ------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,tag", [
    ("contact jane.doe@borcelle.co.uk for the order", "email"),
    ("total £1,234.56 due", "amount"),
    ("GBP 15877.50 at risk", "amount"),
    ("price 1234.00 EUR per unit", "amount"),
    ("$99", "amount"),
    ("see INV-2024-001 attached", "document_ref"),
    ("against PO 4500123", "document_ref"),
    ("invoice 01-2024-002 appears twice", "document_ref"),
    ("quote QT-77A for the laptops", "document_ref"),
])
def test_each_detector_finds_its_kind(text, tag):
    assert tag in ec.tags_for(text)


@pytest.mark.parametrize("text", [
    "Borcelle Ltd supplier profile",
    "widgets 2024 catalogue",
    "version 3.1.4 released",
    "call 0121 496 0000",
    "Softcat Ltd", "South London Power Tools", "potential suppliers", "inventory", "purchase ledger",
    "Pound Shop", "Portsmouth", "Quotation", "so good", "Point of sale", "RFQ for widgets",
    "https://duckduckgo.com/html/?q=Softcat+plc+head+office",
    "5 cad", "inr 5",
    "2024.10", "id 1234567890",
])
def test_ordinary_words_numbers_and_versions_are_not_tagged(text):
    assert ec.tags_for(text) == []


@pytest.mark.parametrize("text", [
    "1," * 32_768, "a" * 65_536, "9" * 65_536, "[" + ",".join(str(i) for i in range(12_000)) + "]",
    "ab@" * 20_000, "£" * 30_000 + "1", "PO " * 20_000, "invoice " * 8_000,
    "a" * 65_000 + "@x.io",          # one address at the end of a long run: an unanchored pattern is quadratic here
    "a" * 65_000 + " x@y.io",
])
def test_the_scan_is_linear_on_hostile_input(text):
    import time
    t = time.perf_counter()
    ec.tags_for(text)
    assert time.perf_counter() - t < 0.05, "a pattern backtracked across the input"


def test_a_date_after_the_word_quote_is_an_accepted_false_positive():
    assert "document_ref" in ec.tags_for("quote 2024-10-01"), "documented: a dated token after 'quote' reads as a reference"


def test_long_verbatim_text_is_tagged_and_the_scan_is_capped():
    assert "long_text" in ec.tags_for("x" * (ec.LONG_TEXT_CHARS + 1))
    assert "long_text" not in ec.tags_for("x" * (ec.LONG_TEXT_CHARS - 1))
    huge = "a " * ec.SCAN_CAP_CHARS + " jane@x.io"     # the address sits beyond the cap
    tags = ec.tags_for(huge)
    assert "long_text" in tags and "email" not in tags, "only the first SCAN_CAP_CHARS are scanned"


# ---- levels ---------------------------------------------------------------------------------------

def test_tags_map_to_an_ordered_level():
    assert ec.level_of([]) == "public"
    assert ec.level_of(["amount"]) == ec.level_of(["document_ref"]) == "commercial"
    assert ec.level_of(["email"]) == ec.level_of(["long_text"]) == "confidential"
    assert ec.level_of(["amount", "email"]) == "confidential"
    assert ec.LEVELS == ("public", "commercial", "confidential")
    assert ec.exceeds("commercial", "public") and not ec.exceeds("public", "commercial")
    assert not ec.exceeds("commercial", "commercial")


# ---- what a call carries ---------------------------------------------------------------------------

def test_classify_reads_the_query_string_and_the_body():
    assert ec.classify("https://s.example/?q=Borcelle+jane%40borcelle.co.uk", None) == ["email"]
    assert ec.classify("https://m.example/api", {"prompt": "pay INV-9001 now", "n": 1}) == ["document_ref"]
    assert ec.classify("https://m.example/api", "total £5") == ["amount"]
    assert ec.classify("https://m.example/api", b"ref PO 123456") == ["document_ref"]
    assert ec.classify("https://api.example/v6/latest/GBP", None) == []


def test_an_unserialisable_body_classifies_as_nothing_rather_than_raising():
    class Odd:
        def __repr__(self):
            raise RuntimeError("no")
    assert ec.classify("https://m.example/", {"x": Odd()}) == []


# ---- the rule cap -----------------------------------------------------------------------------------

def _policy(rules):
    return ep.Policy(version=1, rules=rules)


def test_an_allow_rule_caps_what_may_leave_and_defaults_to_public():
    capped = _policy({"allow": [{"purpose": "supplier_research", "hosts": ["*"], "agents": ["*"]}]})
    assert capped.decide(Purpose.SUPPLIER_RESEARCH, "a.example", "x", level="public").effect == "allow"
    d = capped.decide(Purpose.SUPPLIER_RESEARCH, "a.example", "x", level="commercial")
    assert d.effect == "deny" and "commercial" in d.reason and "public" in d.reason
    wide = _policy({"allow": [{"purpose": "mailbox", "hosts": ["graph.microsoft.com"], "agents": ["*"],
                               "max_classification": "confidential"}]})
    assert wide.decide(Purpose.MAILBOX, "graph.microsoft.com", "x", level="confidential").effect == "allow"


def test_a_null_cap_in_an_edited_row_reads_as_public_not_as_a_dropped_rule():
    p = _policy({"allow": [{"purpose": "fx_rates", "hosts": ["a"], "agents": ["*"], "max_classification": None}]})
    assert p.decide(Purpose.FX_RATES, "a", "x", level="public").effect == "allow"
    assert p.decide(Purpose.FX_RATES, "a", "x", level="commercial").effect == "deny"
    assert "max_classification" not in ep.validate({"allow": [{"purpose": "fx_rates", "hosts": ["a"], "agents": ["*"],
                                                                "max_classification": None}]})["allow"][0]


def test_validate_accepts_a_level_and_refuses_an_unknown_one():
    ok = ep.validate({"allow": [{"purpose": "fx_rates", "hosts": ["a"], "agents": ["*"],
                                 "max_classification": "commercial"}]})
    assert ok["allow"][0]["max_classification"] == "commercial"
    with pytest.raises(ep.InvalidRules):
        ep.validate({"allow": [{"purpose": "fx_rates", "hosts": ["a"], "agents": ["*"],
                                "max_classification": "secret"}]})


def test_the_seed_lets_the_mailbox_carry_addresses_but_research_only_public_words():
    p = _policy(ep.SEED_RULES)
    assert p.decide(Purpose.MAILBOX, "graph.microsoft.com", "m", level="confidential").effect == "allow"
    assert p.decide(Purpose.SUPPLIER_RESEARCH, "x.example", "r", level="public").effect == "allow"
    assert p.decide(Purpose.SUPPLIER_RESEARCH, "x.example", "r", level="commercial").effect == "deny"


# ---- wired into the gate and the audit row --------------------------------------------------------

class _Resp:
    status_code = 200
    headers = {}
    text = ""


@pytest.fixture
def world(monkeypatch):
    calls, events = [], []
    monkeypatch.setattr(egress.requests, "request", lambda m, u, **kw: calls.append((m, u)) or _Resp())
    monkeypatch.setattr(egress, "_resolve",
                        lambda host: ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"])
    monkeypatch.setattr(egress_log, "record", lambda **kw: events.append(kw))
    monkeypatch.setattr(ep, "_load_row", lambda: ep.Policy(version=1, rules=ep.SEED_RULES))
    ep.invalidate()
    yield calls, events
    ep.invalidate()


def test_an_external_call_records_its_tags_and_the_cap_decides_the_shadow(world):
    calls, events = world
    egress.get("https://x.example/?q=Borcelle", purpose=Purpose.SUPPLIER_RESEARCH, agent="r")
    assert events[-1]["classification"] == [] and events[-1]["shadow_decision"] == "allow"
    egress.post("https://x.example/", purpose=Purpose.SUPPLIER_RESEARCH, agent="r",
                json={"q": "Borcelle invoice INV-2024-001 for £17,065.50"})
    assert sorted(events[-1]["classification"]) == ["amount", "document_ref"]
    assert events[-1]["shadow_decision"] == "deny" and "commercial" in events[-1]["shadow_reason"]
    assert len(calls) == 2, "still shadow: both calls went out"


def test_a_local_call_is_never_classified(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(ec, "classify", lambda url, body: (_ for _ in ()).throw(AssertionError("classified a local call")))
    egress.post("http://localhost:11434/api/generate", purpose=Purpose.MODEL_INFERENCE,
                require_global=False, json={"prompt": "jane@x.io owes £5 on INV-1"})
    assert events[-1]["classification"] == [] and events[-1]["shadow_decision"] == "allow"


def test_the_event_row_carries_the_tags():
    row = egress_log._row({
        "occurred_at": None, "purpose": "supplier_research", "destination": "x", "method": "POST",
        "outcome": "http_200", "detail": None, "payload_sha256": "abc", "payload_bytes": 3,
        "classification": ["amount", "email"],
    })
    assert row[10] == ["amount", "email"]
    assert egress_log._row({"occurred_at": None, "purpose": "audit", "destination": "-", "method": "-",
                            "outcome": "events_dropped", "detail": None, "payload_sha256": None,
                            "payload_bytes": None})[10] == []
