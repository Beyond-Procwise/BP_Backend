"""The rule book: detection rules loaded from proc.bp_rule.

A rule says "this is the case" about facts and produces a finding. It never
grants or refuses permission -- that is a policy's job, and the two are kept in
separate tables precisely so neither can quietly become the other.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engines.rule_book import Rule, RuleBook, RuleBookUnavailable


def _rows():
    return [
        {
            "rule_id": 1,
            "rule_name": "Price Benchmark Variance",
            "detector_slug": "price_variance_check",
            "finding_type": "opportunity",
            "scope": "po_lines",
            "required_fields": json.dumps(["supplier_id", "item_id"]),
            "conditions": json.dumps({}),
            "severity": "medium",
            "rule_status": 1,
            "version": 1,
        },
        {
            "rule_id": 2,
            "rule_name": "Contract Expiry Opportunity",
            "detector_slug": "contract_expiry_check",
            "finding_type": "opportunity",
            "scope": "contracts",
            "required_fields": json.dumps(["negotiation_window_days"]),
            "conditions": json.dumps({"negotiation_window_days": 90}),
            "severity": "medium",
            "rule_status": 1,
            "version": 1,
        },
    ]


def test_loads_active_rules_from_rows():
    book = RuleBook(rule_rows=_rows())

    slugs = [rule.detector_slug for rule in book.active_rules()]

    assert slugs == ["price_variance_check", "contract_expiry_check"]
    assert all(isinstance(rule, Rule) for rule in book.active_rules())


def test_carries_conditions_and_required_fields_from_the_row():
    book = RuleBook(rule_rows=_rows())

    expiry = book.rule_for("contract_expiry_check")

    assert expiry.conditions == {"negotiation_window_days": 90}
    assert expiry.required_fields == ["negotiation_window_days"]
    assert expiry.rule_name == "Contract Expiry Opportunity"
    assert expiry.scope == "contracts"


def test_absent_threshold_stays_absent():
    """An empty conditions map means "no default" -- never a zero.

    Ten of the twelve detectors have no default threshold anywhere in the
    codebase. Seeding those as 0.0 would not be preserving behaviour, it would
    be inventing a value, and on a minimum-value filter 0.0 means "fire on
    everything".
    """
    book = RuleBook(rule_rows=_rows())

    assert book.rule_for("price_variance_check").conditions == {}


def test_inactive_rules_are_excluded():
    rows = _rows()
    rows[0]["rule_status"] = 0

    book = RuleBook(rule_rows=rows)

    assert [r.detector_slug for r in book.active_rules()] == ["contract_expiry_check"]


def test_rule_for_unknown_slug_is_none():
    assert RuleBook(rule_rows=_rows()).rule_for("no_such_check") is None


def test_an_empty_rule_book_raises_rather_than_detecting_nothing():
    """Zero rules is an outage, not a configuration.

    Detection that silently finds nothing is indistinguishable from a clean
    scan, which is the fail-open trap this codebase already carries in its
    governance read path. The rule book refuses to be the next instance.
    """
    with pytest.raises(RuleBookUnavailable):
        RuleBook(rule_rows=[])


def test_an_unreadable_store_raises_rather_than_returning_empty():
    def exploding_factory():
        raise RuntimeError("connection reset")

    with pytest.raises(RuleBookUnavailable):
        RuleBook(connection_factory=exploding_factory)


def test_conditions_may_be_a_dict_already():
    """psycopg2 hands back JSONB as a dict; a driver that hands back text works too."""
    rows = _rows()
    rows[0]["conditions"] = {"variance_threshold_pct": 5.0}
    rows[0]["required_fields"] = ["supplier_id"]

    book = RuleBook(rule_rows=rows)

    assert book.rule_for("price_variance_check").conditions == {
        "variance_threshold_pct": 5.0
    }
    assert book.rule_for("price_variance_check").required_fields == ["supplier_id"]


# -- startup wiring ------------------------------------------------------


def test_load_rule_book_returns_none_when_the_table_is_missing(caplog):
    """A missing rule table must not stop the API from booting.

    The blast radius of "no detection rules" is detection. Raising here would
    take down every unrelated endpoint with it, so the failure is carried as a
    None and re-raised later by whoever actually tries to detect something.
    """
    from engines.rule_book import load_rule_book

    def exploding_factory():
        raise RuntimeError('relation "proc.bp_rule" does not exist')

    nick = type("Nick", (), {"get_db_connection": staticmethod(exploding_factory)})()

    with caplog.at_level("ERROR"):
        book = load_rule_book(nick)

    assert book is None
    assert "bp_rule" in caplog.text


def test_load_rule_book_returns_the_book_when_the_table_is_readable():
    from engines.rule_book import load_rule_book

    book = load_rule_book(None, rule_rows=_rows())

    assert book is not None
    assert book.slugs() == ["price_variance_check", "contract_expiry_check"]


def test_camel_case_variance_threshold_resolves():
    """``varianceThresholdPct`` must reach ``variance_threshold_pct``.

    It did not: the alias set held ``variance_thresholdpct`` and
    ``thresholdpct`` but not the plain camelCase form, which normalises to
    ``variancethresholdpct``. The gap was invisible while a bp_policy row
    supplied the same value through a different path.
    """
    from agents.opportunity_miner_agent import _NORMALISED_CONDITION_ALIASES
    from utils.instructions import normalize_instruction_key

    aliases = _NORMALISED_CONDITION_ALIASES["variance_threshold_pct"]

    assert normalize_instruction_key("varianceThresholdPct") in aliases
