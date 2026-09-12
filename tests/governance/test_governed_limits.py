"""A governance limit comes from policy, and a missing one refuses.

Thirty-five limits — the thirty-three P9 moved, plus the two the reseller
catalog added — deciding what reaches the financial record, what counts as a
match on money, who a supplier is, what an agent may offer and how far it may
reach. The original thirty-three were read from `os.getenv` with a hardcoded
default: changeable with no code change AND no policy edit, versioned by
nothing, on no governance screen.

This is the one place that reads them now. The env var keeps working for one
release so a tuned deployment does not silently revert during rollout, but it
warns whenever it disagrees with policy -- an override nobody can see is how the
environment came to be the real source of truth in the first place.

The part that has to be right is the refusal. Fifteen of the thirty-three had
their default INSIDE the getenv call, so "the policy is missing" and "the policy
says the old number" were the same value; a fail-closed guard is untestable while
that is true, which is the same hole P6 found one layer down.
"""

import logging

import pytest

from src.services import governed_limits as GL


class _Engine:
    """Stands in for PolicyEngine. `rules=None` means the row does not exist."""

    def __init__(self, rules, slug="promotion_thresholds"):
        self._rules = rules
        self._slug = slug

    def get_policy(self, slug):
        if slug != self._slug or self._rules is None:
            return None
        return {"policyName": "PromotionThresholdPolicy",
                "details": {"policy_identifier": slug, "rules": dict(self._rules)}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    GL.reset_cache()
    monkeypatch.delenv("PROMOTE_MIN_CONFIDENCE", raising=False)
    yield
    GL.reset_cache()


def _with(rules, monkeypatch, slug="promotion_thresholds"):
    monkeypatch.setattr(GL, "_engine", lambda: _Engine(rules, slug))


# ---------------------------------------------------------------------------
# the policy value is the value
# ---------------------------------------------------------------------------
def test_the_policy_value_is_used(monkeypatch):
    _with({"promote_min_confidence": 50}, monkeypatch)

    assert GL.limit("promotion_thresholds", "promote_min_confidence") == 50.0


def test_an_integer_limit_comes_back_as_an_integer(monkeypatch):
    _with({"propose_max_candidates": 5}, monkeypatch)

    value = GL.limit("promotion_thresholds", "propose_max_candidates", cast=int)
    assert value == 5 and isinstance(value, int)


def test_a_stated_no_limit_is_not_a_missing_limit(monkeypatch):
    """`neg_thread_transcript_limit` is present and null on purpose: null is a
    decision ("no limit"), absent is an unanswered question."""
    _with({"neg_thread_transcript_limit": None}, monkeypatch, slug="agent_reach")

    assert GL.limit("agent_reach", "neg_thread_transcript_limit", cast=int) is None


def test_a_boolean_switch_reads_as_a_boolean(monkeypatch):
    _with({"duplicate_invoice_detector_enabled": False}, monkeypatch,
          slug="autonomous_operation")

    assert GL.limit("autonomous_operation", "duplicate_invoice_detector_enabled",
                    cast=bool) is False


# ---------------------------------------------------------------------------
# a missing value refuses -- it does not fall back
# ---------------------------------------------------------------------------
def test_a_missing_rule_refuses(monkeypatch):
    _with({"something_else": 1}, monkeypatch)

    with pytest.raises(GL.LimitUnavailable):
        GL.limit("promotion_thresholds", "promote_min_confidence")


def test_a_missing_policy_row_refuses(monkeypatch):
    _with(None, monkeypatch)

    with pytest.raises(GL.LimitUnavailable):
        GL.limit("promotion_thresholds", "promote_min_confidence")


def test_an_unreadable_policy_store_refuses(monkeypatch):
    class _Broken:
        def get_policy(self, slug):
            raise RuntimeError("governance database is unreachable")

    monkeypatch.setattr(GL, "_engine", lambda: _Broken())

    with pytest.raises(GL.LimitUnavailable):
        GL.limit("promotion_thresholds", "promote_min_confidence")


def test_no_environment_variable_can_substitute_for_a_missing_policy(monkeypatch):
    """The override is an override OF a policy, not a replacement FOR one.
    Otherwise 'unset the policy, set the env var' is a way to govern nothing."""
    _with({}, monkeypatch)
    monkeypatch.setenv("PROMOTE_MIN_CONFIDENCE", "10")

    with pytest.raises(GL.LimitUnavailable):
        GL.limit("promotion_thresholds", "promote_min_confidence",
                 env="PROMOTE_MIN_CONFIDENCE")


# ---------------------------------------------------------------------------
# the environment override, for one release, and never silently
# ---------------------------------------------------------------------------
def test_an_environment_override_wins_and_says_so(monkeypatch, caplog):
    _with({"promote_min_confidence": 50}, monkeypatch)
    monkeypatch.setenv("PROMOTE_MIN_CONFIDENCE", "70")

    with caplog.at_level(logging.WARNING):
        value = GL.limit("promotion_thresholds", "promote_min_confidence",
                         env="PROMOTE_MIN_CONFIDENCE")

    assert value == 70.0
    warning = " ".join(r.getMessage() for r in caplog.records)
    assert "PROMOTE_MIN_CONFIDENCE" in warning, warning
    assert "70" in warning and "50" in warning, (
        f"the warning must name both values, or nobody can tell what was "
        f"overridden: {warning}")


def test_an_override_that_agrees_with_policy_is_silent(monkeypatch, caplog):
    """Warn on a difference, not on the mere presence of the variable — a
    warning on every read is a warning nobody reads."""
    _with({"promote_min_confidence": 50}, monkeypatch)
    monkeypatch.setenv("PROMOTE_MIN_CONFIDENCE", "50")

    with caplog.at_level(logging.WARNING):
        value = GL.limit("promotion_thresholds", "promote_min_confidence",
                         env="PROMOTE_MIN_CONFIDENCE")

    assert value == 50.0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        [r.getMessage() for r in caplog.records])


def test_an_unusable_override_is_ignored_in_favour_of_policy(monkeypatch, caplog):
    """A typo in an environment variable must not decide what reaches the
    financial record, and must not pass silently either."""
    _with({"promote_min_confidence": 50}, monkeypatch)
    monkeypatch.setenv("PROMOTE_MIN_CONFIDENCE", "about fifty")

    with caplog.at_level(logging.WARNING):
        value = GL.limit("promotion_thresholds", "promote_min_confidence",
                         env="PROMOTE_MIN_CONFIDENCE")

    assert value == 50.0
    assert caplog.records, "an unusable override passed without a word"


def test_an_empty_override_is_not_an_override(monkeypatch):
    _with({"promote_min_confidence": 50}, monkeypatch)
    monkeypatch.setenv("PROMOTE_MIN_CONFIDENCE", "   ")

    assert GL.limit("promotion_thresholds", "promote_min_confidence",
                    env="PROMOTE_MIN_CONFIDENCE") == 50.0


# ---------------------------------------------------------------------------
# the live rows this all depends on
# ---------------------------------------------------------------------------
def test_every_governed_limit_is_present_in_the_live_policy_set():
    """Reads the LIVE rows, because a fixture richer than the database is how
    this project has repeatedly shipped a guard that was already dead."""
    import os

    import psycopg2

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), connect_timeout=10)
    except Exception:
        pytest.skip("no DB")

    with conn, conn.cursor() as cur:
        cur.execute("""SELECT policy_details->>'policy_identifier',
                              policy_details->'rules'
                         FROM proc.bp_policy
                        WHERE policy_type = 'limit' AND policy_status = 1""")
        live = {slug: rules for slug, rules in cur.fetchall()}
    conn.close()

    assert len(live) == 8, f"expected eight limit rows, found {sorted(live)}"
    assert sum(len(r) for r in live.values()) == 35, (
        f"expected 35 governed values, found "
        f"{ {k: len(v) for k, v in live.items()} }")
    for slug, rules in live.items():
        assert rules, f"{slug} states no limits at all"


def test_the_in_memory_seed_matches_the_live_rows():
    """The test suite runs against a COPY of these rows (services/db.py), because
    the in-memory store has no governance database. A copy can drift, and a
    drifted copy is worse than none: the suite would be proving a value the
    product does not use. This is the only thing that makes the copy safe."""
    import os

    import psycopg2

    from tests.conftest import GOVERNED_LIMIT_SEED

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), connect_timeout=10)
    except Exception:
        pytest.skip("no DB")

    with conn, conn.cursor() as cur:
        cur.execute("""SELECT policy_details->>'policy_identifier',
                              policy_details->'rules'
                         FROM proc.bp_policy
                        WHERE policy_type = 'limit' AND policy_status = 1""")
        live = {slug: rules for slug, rules in cur.fetchall()}
    conn.close()

    seed = GOVERNED_LIMIT_SEED
    assert set(seed) == set(live), (
        f"seed and live disagree on which policies exist: "
        f"{set(seed) ^ set(live)}")
    for slug in sorted(seed):
        assert seed[slug] == live[slug], (
            f"{slug} has drifted:\n  seed = {seed[slug]}\n  live = {live[slug]}")
