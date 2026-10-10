"""The four statistics and the guardrails (brief §6.2, §6.3). No database."""
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import learning_stats as LS
from services.agent_policy.settings import DEFAULTS

CFG = DEFAULTS["learning"]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def group(n=30, approved=None, approvers=3, days=30, seconds=45, reversed_at=(), edited_at=()):
    """n decisions spread evenly over `days`, by `approvers` people, each taking `seconds`."""
    approved = n if approved is None else approved
    out = []
    for i in range(n):
        asked = T0 + timedelta(days=days * i / (n - 1) if n > 1 else 0)
        out.append(LS.Decided(firing_id=i, values={"args.amount": 600 + i}, approved=i < approved,
                              decided_by=f"person-{i % approvers}", asked_at=asked,
                              decided_at=asked + timedelta(seconds=seconds),
                              reversed=i in reversed_at, edited=i in edited_at))
    return out


@pytest.mark.parametrize("k,n,expect", [(30, 30, 0.8865), (29, 30, 0.8333), (0, 30, 0.0), (58, 60, 0.8864)])
def test_wilson_lower_bound(k, n, expect):
    assert round(LS.wilson_lower(k, n), 4) == expect


def test_wilson_of_nothing_is_zero():
    assert LS.wilson_lower(0, 0) == 0.0


def test_stats():
    s = LS.stats(group(n=30, approved=29, approvers=4, days=45, seconds=90))
    assert s == {"decisions": 30, "approved": 29, "wilsonLower": 0.8333, "approvers": 4, "days": 45,
                 "edited": 0, "reversed": 0, "medianSeconds": 90.0}


def test_stats_of_an_empty_group():
    assert LS.stats([])["decisions"] == 0 and LS.stats([])["medianSeconds"] is None


def test_the_same_person_in_different_letter_case_is_one_approver():
    g = group(n=4, approvers=1)
    g[1] = LS.Decided(**{**g[1].__dict__, "decided_by": "PERSON-0 "})
    assert LS.stats(g)["approvers"] == 1


def test_every_guardrail_holds_on_the_ruled_defaults():
    assert LS.failed_guardrails(LS.stats(group()), CFG) == []


@pytest.mark.parametrize("kwargs,failed", [
    (dict(n=29), ["min_decisions"]),
    (dict(days=29), ["min_days"]),
    (dict(approvers=2), ["min_approvers"]),
    (dict(approved=29), ["wilson_lower"]),
    (dict(reversed_at=(3,)), ["edited_or_reversed"]),
    (dict(edited_at=(3,)), ["edited_or_reversed"]),
    (dict(seconds=29), ["median_seconds_floor"]),
])
def test_each_guardrail_alone(kwargs, failed):
    assert LS.failed_guardrails(LS.stats(group(**kwargs)), CFG) == failed


def test_exactly_on_each_threshold_passes():
    # 30 decisions, 30 days, 3 approvers, median exactly 30 s: "at least" (design §2, Defaults)
    assert LS.failed_guardrails(LS.stats(group(n=30, days=30, approvers=3, seconds=30)), CFG) == []


def test_a_setting_that_is_not_a_number_raises_and_never_guesses():
    with pytest.raises(TypeError):
        LS.failed_guardrails(LS.stats(group()), {**CFG, "min_decisions": "thirty"})
