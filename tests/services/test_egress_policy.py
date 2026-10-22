"""1.20 step (b): what may leave the system is an administrator's POLICY row, not code.

``EgressPolicy`` (proc.bp_policy, identifier ``egress``) holds allow and deny rules keyed on purpose,
destination host pattern and agent. The egress gate's SHADOW decision now comes from that row when it
exists, and from the built-in default rule set when it does not. Still shadow: nothing is blocked
(enforcement per purpose is step d). Fail closed: an unreadable row means every external destination
WOULD be denied; local destinations are always allowed and never consult the row.

Starting rules (seed, from the last week's evidence and D-62): currency rates to open.er-api.com,
mailbox to the Microsoft hosts, supplier research to the public web (the SSRF guard still applies),
and an explicit DENY of the model host api.ollama.com: company text never goes to Ollama cloud.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services import egress, egress_log
from src.services import egress_policy as ep
from src.services.egress import Purpose
from tests.services.test_permission_settings import FakePolicyTable


@pytest.fixture(autouse=True)
def fresh_cache():
    ep.invalidate()
    yield
    ep.invalidate()


@pytest.fixture
def table(monkeypatch):
    t = FakePolicyTable()
    monkeypatch.setattr(ep, "_record", lambda conn, **kw: t.audit.append(kw))
    monkeypatch.setattr(ep, "get_conn", lambda: _ctx(t))
    return t


class _ctx:
    def __init__(self, t):
        self.t = t

    def __enter__(self):
        return self.t

    def __exit__(self, *a):
        return False


def _shadow(purpose, host, agent="unknown"):
    d = ep.shadow_for(purpose, host, agent, local=False)
    return d.effect, d.reason, d.version


# ---- no row: the built-in default rule set --------------------------------------------------------

def test_without_a_row_external_is_denied_and_local_allowed_under_the_default_rules(table):
    effect, reason, version = _shadow(Purpose.FX_RATES, "open.er-api.com")
    assert (effect, version) == ("deny", ep.DEFAULT_VERSION)
    d = ep.shadow_for(Purpose.MODEL_INFERENCE, "localhost", "rag", local=True)
    assert (d.effect, d.version) == ("allow", ep.DEFAULT_VERSION)


# ---- the seeded row -----------------------------------------------------------------------------

def test_seed_writes_the_starting_rules_once(table):
    assert ep.seed("admin-1")["changed"] is True
    assert ep.seed("admin-1")["changed"] is False
    row = table.rows[ep.POLICY_NAME]
    assert row["details"]["policy_identifier"] == ep.IDENTIFIER
    assert row["version"] == 1


def test_seeded_rules_allow_the_known_callers_and_deny_everything_else(table):
    ep.seed("admin-1")
    assert _shadow(Purpose.FX_RATES, "open.er-api.com")[0] == "allow"
    assert _shadow(Purpose.FX_RATES, "evil.example")[0] == "deny"
    assert _shadow(Purpose.MAILBOX, "graph.microsoft.com", "mail_intake")[0] == "allow"
    assert _shadow(Purpose.MAILBOX, "login.microsoftonline.com")[0] == "allow"
    assert _shadow(Purpose.SUPPLIER_RESEARCH, "www.some-supplier.co.uk", "outside_research")[0] == "allow"
    assert _shadow(Purpose.NOTIFICATION, "smtp.example.com")[0] == "deny", "nothing seeded for it: shadow will show the need"
    assert all(_shadow(Purpose.FX_RATES, "open.er-api.com")[2] == "egress-policy-v1" for _ in range(1))


def test_the_seed_denies_ollama_cloud_by_name_as_d62_requires(table):
    ep.seed("admin-1")
    effect, reason, _ = _shadow(Purpose.MODEL_INFERENCE, "api.ollama.com", "negotiation")
    assert effect == "deny" and "Ollama cloud" in reason


def test_local_destinations_never_consult_the_row(table):
    ep.seed("admin-1")
    table.rows.clear()                         # even with the row gone...
    d = ep.shadow_for(Purpose.MODEL_INFERENCE, "localhost", "rag", local=True)
    assert d.effect == "allow"


# ---- matching ------------------------------------------------------------------------------------

def _policy(rules):
    return ep.Policy(version=7, rules=rules)


def test_host_patterns_exact_suffix_wildcard_and_any():
    p = _policy({"allow": [{"purpose": "object_storage", "hosts": ["*.amazonaws.com"], "agents": ["*"]},
                          {"purpose": "fx_rates", "hosts": ["Open.ER-API.com"], "agents": ["*"]},
                          {"purpose": "supplier_research", "hosts": ["*"], "agents": ["*"]}]})
    assert p.decide(Purpose.OBJECT_STORAGE, "bucket.s3.eu-west-1.amazonaws.com", "x").effect == "allow"
    assert p.decide(Purpose.OBJECT_STORAGE, "amazonaws.com.evil.example", "x").effect == "deny"
    assert p.decide(Purpose.FX_RATES, "open.er-api.com", "x").effect == "allow", "case does not matter"
    assert p.decide(Purpose.SUPPLIER_RESEARCH, "anything.example", "x").effect == "allow"
    assert p.decide(Purpose.FX_RATES, "open.er-api.com.evil.example", "x").effect == "deny"


def test_a_rule_can_be_limited_to_named_agents():
    p = _policy({"allow": [{"purpose": "supplier_research", "hosts": ["*"], "agents": ["outside_research"]}]})
    assert p.decide(Purpose.SUPPLIER_RESEARCH, "a.example", "outside_research").effect == "allow"
    d = p.decide(Purpose.SUPPLIER_RESEARCH, "a.example", "unknown")
    assert d.effect == "deny" and "unknown" in d.reason


def test_a_deny_rule_beats_an_allow_rule():
    p = _policy({"allow": [{"purpose": "model_inference", "hosts": ["*"], "agents": ["*"]}],
                 "deny": [{"purpose": "model_inference", "hosts": ["api.ollama.com"], "agents": ["*"],
                           "note": "no"}]})
    assert p.decide(Purpose.MODEL_INFERENCE, "api.ollama.com", "x").effect == "deny"
    assert p.decide(Purpose.MODEL_INFERENCE, "other.example", "x").effect == "allow"


def test_a_malformed_rule_in_the_row_is_ignored_not_widened():
    p = _policy({"allow": [{"purpose": "fx_rates"}, "junk", {"purpose": "nope", "hosts": ["*"], "agents": ["*"]}]})
    assert p.decide(Purpose.FX_RATES, "open.er-api.com", "x").effect == "deny"


# ---- fail closed ---------------------------------------------------------------------------------

def test_an_unreadable_row_denies_external_and_allows_local(monkeypatch):
    class Boom:
        def __enter__(self):
            raise RuntimeError("db down")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ep, "get_conn", lambda: Boom())
    d = ep.shadow_for(Purpose.FX_RATES, "open.er-api.com", "fx", local=False)
    assert d.effect == "deny" and "could not be read" in d.reason and d.version == ep.UNREADABLE_VERSION
    assert ep.shadow_for(Purpose.MODEL_INFERENCE, "localhost", "rag", local=True).effect == "allow"


# ---- changing the rules --------------------------------------------------------------------------

GOOD = {"allow": [{"purpose": "fx_rates", "hosts": ["open.er-api.com"], "agents": ["*"]}], "deny": []}


def test_replace_rules_validates_bumps_version_and_records_history_and_audit(table):
    ep.seed("admin-1")
    out = ep.replace_rules(GOOD, by="admin-2", expected_version=1)
    assert out == {"version": 2, "changed": True}
    row = table.rows[ep.POLICY_NAME]
    assert row["version"] == 2 and row["details"]["rules"] == GOOD
    assert row["details"]["history"][-1]["changed_by"] == "admin-2"
    assert table.audit[-1]["action_type"] == "egress_policy_changed"
    assert _shadow(Purpose.MAILBOX, "graph.microsoft.com")[0] == "deny", "the cache saw the change"


@pytest.mark.parametrize("bad", [
    {"allow": [{"purpose": "teleport", "hosts": ["*"], "agents": ["*"]}]},
    {"allow": [{"purpose": "fx_rates", "hosts": [], "agents": ["*"]}]},
    {"allow": [{"purpose": "fx_rates", "hosts": ["a"], "agents": []}]},
    {"allow": [{"purpose": "fx_rates", "hosts": "a", "agents": ["*"]}]},
    {"allow": "x"},
    {"allow": [], "deny": [], "enforce": ["fx_rates"]},
])
def test_bad_rules_are_refused_and_nothing_is_written(table, bad):
    ep.seed("admin-1")
    with pytest.raises(ep.InvalidRules):
        ep.replace_rules(bad, by="admin-2", expected_version=1)
    assert table.rows[ep.POLICY_NAME]["version"] == 1


def test_a_stale_save_is_refused(table):
    ep.seed("admin-1")
    with pytest.raises(ep.RulesConflict):
        ep.replace_rules(GOOD, by="admin-2", expected_version=0)


def test_current_reports_the_rules_version_and_whether_seeded(table):
    before = ep.current()
    assert before["configured"] is False and before["rules"] == {"allow": [], "deny": []}
    assert before["seed"] == ep.SEED_RULES, "the seed is offered, not presented as in force"
    ep.seed("admin-1")
    cur = ep.current()
    assert cur["configured"] is True and cur["version"] == 1 and cur["rules"]["deny"]


# ---- the cache ------------------------------------------------------------------------------------

def test_the_row_is_read_once_per_ttl_not_once_per_call(table, monkeypatch):
    ep.seed("admin-1")
    reads = []
    orig = ep._load_row
    monkeypatch.setattr(ep, "_load_row", lambda: reads.append(1) or orig())
    for _ in range(5):
        _shadow(Purpose.FX_RATES, "open.er-api.com")
    assert len(reads) == 1
    ep.invalidate()
    _shadow(Purpose.FX_RATES, "open.er-api.com")
    assert len(reads) == 2


# ---- wired into the gate ------------------------------------------------------------------------

def test_egress_records_the_policy_rows_decision_and_version(table, monkeypatch):
    ep.seed("admin-1")
    events = []
    monkeypatch.setattr(egress.requests, "request", lambda m, u, **kw: type("R", (), {"status_code": 200, "headers": {}, "text": ""})())
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(egress_log, "record", lambda **kw: events.append(kw))
    egress.get("https://open.er-api.com/v6/latest/GBP", purpose=Purpose.FX_RATES, agent="fx")
    assert (events[-1]["shadow_decision"], events[-1]["policy_version"]) == ("allow", "egress-policy-v1")
    egress.post("https://api.ollama.com/v1/chat", purpose=Purpose.MODEL_INFERENCE, json={"q": 1}, agent="neg")
    assert events[-1]["shadow_decision"] == "deny" and events[-1]["outcome"] == "http_200", "still shadow"


def test_the_seed_sql_draft_matches_the_code_seed():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "deploy" / "sql" / "2026-10-22_egress_policy.sql").read_text()
    assert ep.POLICY_NAME in text and "api.ollama.com" in text and "WHERE NOT EXISTS" in text
    start, end = text.index("$json$") + len("$json$"), text.rindex("$json$")
    assert json.loads(text[start:end])["rules"] == ep.SEED_RULES


def test_rules_are_normalised_on_save_and_dead_patterns_refused(table):
    ep.seed("admin-1")
    ep.replace_rules({"allow": [{"purpose": "fx_rates", "hosts": [" Open.ER-API.com "], "agents": [" fx "]}]},
                     by="a", expected_version=1)
    rule = table.rows[ep.POLICY_NAME]["details"]["rules"]["allow"][0]
    assert rule == {"purpose": "fx_rates", "hosts": ["open.er-api.com"], "agents": ["fx"]}
    assert _shadow(Purpose.FX_RATES, "open.er-api.com", "fx")[0] == "allow"
    for bad in ("https://open.er-api.com", "open.er-api.com/v6", "open.er-api.com:443", "*.", "a*b", "*x"):
        with pytest.raises(ep.InvalidRules):
            ep.replace_rules({"allow": [{"purpose": "fx_rates", "hosts": [bad], "agents": ["*"]}]},
                             by="a", expected_version=2)


def test_a_failed_read_is_remembered_so_an_outage_costs_one_attempt_per_window(monkeypatch):
    attempts = []

    def boom():
        attempts.append(1)
        raise RuntimeError("db down")
    monkeypatch.setattr(ep, "_load_row", boom)
    for _ in range(5):
        assert ep.shadow_for(Purpose.FX_RATES, "open.er-api.com", "fx", local=False).version == ep.UNREADABLE_VERSION
    assert len(attempts) == 1
    monkeypatch.setattr(ep, "_cached_at", 0.0)          # the failure window has passed
    ep.shadow_for(Purpose.FX_RATES, "open.er-api.com", "fx", local=False)
    assert len(attempts) == 2


def test_two_seeds_at_once_are_a_conflict_not_a_crash(table, monkeypatch):
    orig = table.cursor

    class RacingCur:
        def __init__(self, inner):
            self.inner = inner

        def execute(self, sql, params=None):
            if sql.lstrip().lower().startswith("insert into proc.bp_policy"):
                raise RuntimeError("duplicate key value violates unique constraint ux_bp_policy_active_type_name")
            return self.inner.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self.inner, name)
    monkeypatch.setattr(table, "cursor", lambda: RacingCur(orig()))
    with pytest.raises(ep.RulesConflict):
        ep.seed("admin-2")


def test_the_default_version_is_the_one_literal_in_egress():
    assert ep.DEFAULT_VERSION == egress.SHADOW_RULES_VERSION
