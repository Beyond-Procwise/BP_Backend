"""P9's tail: the limits that still got round policy.

59fd971 moved thirty-three limits into policy. Four did not fully make it:

  * Two read the environment themselves, ahead of policy, and fell back to a
    number carried in code when the variable was unusable -- the exact shape P9
    removed everywhere else: an override nobody can see, and a default that
    makes "policy missing" and "policy says 30" the same value. One of them was
    also read once at import, pinning it until the process restarted.
  * Two were converted in a working tree and never committed, because the file
    they live in carried another session's unfinished work.
"""

from __future__ import annotations

import logging

import pytest

from src.services import governed_limits as GL


class _Engine:
    def __init__(self, policies):
        self._policies = policies

    def get_policy(self, slug):
        rules = self._policies.get(slug)
        if rules is None:
            return None
        return {"policyName": slug, "details": {"policy_identifier": slug,
                                                "rules": dict(rules)}}


def _policies(monkeypatch, **policies):
    GL.reset_cache()
    monkeypatch.setattr(GL, "_engine", lambda: _Engine(policies))


# ---------------------------------------------------------------------------
# deal assignment: the two limits that never got committed
# ---------------------------------------------------------------------------
def test_the_quote_anchor_bar_is_its_own_governed_value(monkeypatch):
    """It used to default to the link bar INSIDE the getenv call. Equal today is
    a decision, and it is written down where it can change on its own."""
    from src.services import deal_assignment_service as das

    _policies(monkeypatch, promotion_thresholds={
        "promote_min_link_score": 80, "quote_anchor_min_score": 70})

    assert das.MIN_LINK_SCORE() == 80.0
    assert das.QUOTE_ANCHOR_MIN_SCORE() == 70.0


def test_a_missing_quote_anchor_bar_refuses(monkeypatch):
    from src.services import deal_assignment_service as das

    _policies(monkeypatch, promotion_thresholds={"promote_min_link_score": 80})

    with pytest.raises(GL.LimitUnavailable):
        das.QUOTE_ANCHOR_MIN_SCORE()


# ---------------------------------------------------------------------------
# capture retention: policy first, and no number carried in code
# ---------------------------------------------------------------------------
def test_retention_is_the_policy_value(monkeypatch):
    from src.services import capture_retention as R

    monkeypatch.delenv("CAPTURE_RETENTION_DAYS", raising=False)
    _policies(monkeypatch, autonomous_operation={"capture_retention_days": 45})

    assert R.retention_days() == 45


def test_an_unusable_retention_override_gives_way_to_policy(monkeypatch, caplog):
    """It used to give way to a constant in this module -- 30 -- which is exactly
    the value P9 exists to stop code from carrying."""
    from src.services import capture_retention as R

    _policies(monkeypatch, autonomous_operation={"capture_retention_days": 45})
    for bad in ("not-a-number", "0", "-3"):
        GL.reset_cache()
        monkeypatch.setenv("CAPTURE_RETENTION_DAYS", bad)
        with caplog.at_level(logging.WARNING):
            assert R.retention_days() == 45, bad
    assert caplog.records, "an unusable override passed without a word"


def test_a_retention_override_that_differs_from_policy_says_so(monkeypatch, caplog):
    from src.services import capture_retention as R

    _policies(monkeypatch, autonomous_operation={"capture_retention_days": 45})
    monkeypatch.setenv("CAPTURE_RETENTION_DAYS", "14")

    with caplog.at_level(logging.WARNING):
        assert R.retention_days() == 14
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "14" in said and "45" in said, said


@pytest.mark.parametrize("stated", [0, -1, None])
def test_a_retention_policy_that_would_delete_everything_or_nothing_refuses(
        monkeypatch, stated):
    """Zero or less means "delete every capture"; null would mean "keep them
    forever", which is the state this module was written to end. Neither is a
    number anyone set on purpose, so neither is acted on."""
    from src.services import capture_retention as R

    monkeypatch.delenv("CAPTURE_RETENTION_DAYS", raising=False)
    _policies(monkeypatch, autonomous_operation={"capture_retention_days": stated})

    with pytest.raises(GL.LimitUnavailable):
        R.retention_days()


def test_a_missing_retention_policy_refuses(monkeypatch):
    from src.services import capture_retention as R

    monkeypatch.delenv("CAPTURE_RETENTION_DAYS", raising=False)
    _policies(monkeypatch, autonomous_operation={})

    with pytest.raises(GL.LimitUnavailable):
        R.retention_days()


# ---------------------------------------------------------------------------
# negotiation transcript: policy first, read when used, not at import
# ---------------------------------------------------------------------------
def test_an_unusable_transcript_override_gives_way_to_policy(monkeypatch, caplog):
    """It used to mean "full history" -- a decision about how much an agent sees
    before it makes an offer, taken by a typo."""
    from src.agents import negotiation_agent as NA

    _policies(monkeypatch, agent_reach={"neg_thread_transcript_limit": 25})
    monkeypatch.setenv("NEG_THREAD_TRANSCRIPT_LIMIT", "twenty")

    with caplog.at_level(logging.WARNING):
        assert NA._resolve_thread_transcript_limit() == 25
    assert caplog.records


def test_a_transcript_override_that_differs_from_policy_says_so(monkeypatch, caplog):
    from src.agents import negotiation_agent as NA

    _policies(monkeypatch, agent_reach={"neg_thread_transcript_limit": 25})
    monkeypatch.setenv("NEG_THREAD_TRANSCRIPT_LIMIT", "10")

    with caplog.at_level(logging.WARNING):
        assert NA._resolve_thread_transcript_limit() == 10
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "10" in said and "25" in said, said


def test_a_stated_no_limit_still_means_the_full_history(monkeypatch):
    from src.agents import negotiation_agent as NA

    monkeypatch.delenv("NEG_THREAD_TRANSCRIPT_LIMIT", raising=False)
    _policies(monkeypatch, agent_reach={"neg_thread_transcript_limit": None})

    assert NA._resolve_thread_transcript_limit() is None


def test_the_transcript_limit_is_read_when_used_not_when_imported(monkeypatch):
    """A module-level read happens before anything knows whether the governance
    store answered, and pins the value until the process restarts."""
    from src.agents import negotiation_agent as NA

    monkeypatch.delenv("NEG_THREAD_TRANSCRIPT_LIMIT", raising=False)
    _policies(monkeypatch, agent_reach={"neg_thread_transcript_limit": 2})
    entries = [{"n": i} for i in range(5)]

    kept = NA.NegotiationAgent._select_thread_history_entries(None, entries)

    assert kept == entries[-2:]
