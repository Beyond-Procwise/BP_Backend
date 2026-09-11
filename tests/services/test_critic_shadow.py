"""Shadow mode, in the shape services/guardrail.py established.

Per detector, never globally; every enrolment carries an expiry; a finding a
human has already advanced can never be suppressed whatever the config says.
"""
from datetime import datetime, timedelta, timezone

from src.services.opportunity_critic.governed import Thresholds
from src.services.opportunity_critic.shadow import (
    NEVER_SUPPRESS_STAGES, is_shadowed, may_suppress, shadow_status,
)

_FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
_PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _thresholds(detectors):
    return Thresholds(shadow_detectors=tuple(detectors))


def test_nothing_is_shadowed_by_default():
    assert is_shadowed("Price Benchmark Variance", _thresholds([])) is False


def test_an_enrolled_detector_is_shadowed():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    assert is_shadowed("Price Benchmark Variance", t) is True


def test_an_expired_enrolment_is_not_honoured():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _PAST}])
    assert is_shadowed("Price Benchmark Variance", t) is False


def test_an_enrolment_without_an_expiry_is_not_honoured():
    # Shadow mode must not become permanent by nobody getting round to it.
    t = _thresholds([{"detector": "Price Benchmark Variance"}])
    assert is_shadowed("Price Benchmark Variance", t) is False


def test_enrolment_does_not_leak_to_other_detectors():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    assert is_shadowed("Duplicate Invoice Recovery", t) is False


def test_a_finding_a_human_has_advanced_can_never_be_suppressed():
    # Someone is already acting on it; a model changing its mind must not pull
    # it out from under them. This is code, not config.
    for stage in NEVER_SUPPRESS_STAGES:
        allowed, reason = may_suppress({"stage": stage}, _thresholds([]))
        assert allowed is False
        assert stage in reason


def test_the_never_suppress_stages_are_real_opportunity_stages():
    # A stage spelled differently from the store's vocabulary would make the
    # guard above pass in tests and never fire in production.
    from src.services.opportunity_store import _STAGES
    assert set(NEVER_SUPPRESS_STAGES) <= set(_STAGES)


def test_an_identified_finding_may_be_suppressed():
    allowed, _ = may_suppress({"stage": "identified"}, _thresholds([]))
    assert allowed is True


def test_a_shadowed_detector_may_not_suppress():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    allowed, reason = may_suppress(
        {"stage": "identified", "detector_type": "Price Benchmark Variance"}, t)
    assert allowed is False
    assert "shadow" in reason.lower()


def test_health_reports_unavailable_rather_than_an_empty_enrolment():
    # Every governance read here is fail-open: an unreadable policy looks exactly
    # like "nothing governs this". For shadow mode that would say "nothing is
    # enrolled" during an outage, which is the one answer that must not appear.
    from src.services.opportunity_critic.shadow import health_status
    assert health_status(None) == {"error": "unavailable"}


def test_health_reports_the_enrolment_when_the_policy_resolves():
    from src.engines.policy_engine import PolicyEngine
    from src.services.opportunity_critic.shadow import health_status
    row = {"policy_id": 901, "policy_name": "opportunity_critic_thresholds",
           "policy_type": "critique", "policy_status": 1, "version": 1,
           "policy_linked_agents": "opportunity_critic",
           "policy_details": {"rules": {"shadow_detectors": [
               {"detector": "Price Benchmark Variance", "until": _FUTURE}]}}}
    status = health_status(PolicyEngine(policy_rows=[row]))
    assert status["enrolled"][0]["detector"] == "Price Benchmark Variance"
    assert status["enrolled"][0]["active"] is True


def test_status_reports_the_enrolment_and_its_expiry():
    t = _thresholds([{"detector": "Price Benchmark Variance", "until": _FUTURE}])
    status = shadow_status(t)
    assert status["enrolled"][0]["detector"] == "Price Benchmark Variance"
    assert status["enrolled"][0]["until"] == _FUTURE
    assert "never_suppressed_stages" in status
