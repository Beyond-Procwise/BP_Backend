"""enforcement.check (pure) and live_policies.load (cached, fail-closed)."""
import copy

import pytest

from services.agent_policy import enforcement, live_policies
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS


def _doc(key="FIN-0012", outcome="approve", mutate=None, status="live"):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["outcome"] = outcome
    if outcome in ("block", "notify"):
        form["notify"] = ["Finance Manager"] if outcome == "notify" else []
    if mutate:
        mutate(form)
    return compile_policy(form, policy_key=key, version=1, status=status,
                          settings=SETTINGS, never_suggest=False)


def _ctx(amount=900, **extra):
    ctx = {"checkpoint": "tool.call.before", "tool.name": "refund.issue", "agent.name": "agent_nick",
           "agent.reason": "customer asked", "args": {"amount": amount}}
    ctx.update(extra)
    return ctx


def test_block_wins_over_approve():
    v = enforcement.check(_ctx(), [_doc("FIN-0001", "approve"), _doc("FIN-0002", "block")])
    assert v.result == "blocked"
    assert [h["id"] for h in v.blocks] == ["FIN-0002"]
    assert [h["id"] for h in v.approvals] == ["FIN-0001"]
    assert v.to_agent["result"] == "blocked"
    assert v.to_agent["reasonCode"] == "FIN-0002.over_limit"
    assert v.to_agent["reason"] == FORM_EXAMPLE["messageForAgent"]
    assert v.to_agent["messageForPerson"] == FORM_EXAMPLE["messageForPerson"]
    assert v.to_agent["policies"] == ["FIN-0002"]


def test_notify_still_listed_when_blocked():
    block_with_notify = _doc("FIN-0003", "block", lambda f: f.update(notify=["CFO"]))
    v = enforcement.check(_ctx(), [_doc("FIN-0002", "notify"), block_with_notify])
    assert v.result == "blocked"
    ids = {h["id"]: h["notify"] for h in v.notifies}
    assert ids == {"FIN-0002": ["Finance Manager"], "FIN-0003": ["CFO"]}


def test_two_approve_policies_give_two_approvals():
    v = enforcement.check(_ctx(), [_doc("FIN-0001"), _doc("FIN-0004")])
    assert v.result == "paused_for_approval"
    assert [h["id"] for h in v.approvals] == ["FIN-0001", "FIN-0004"]
    ta = v.to_agent
    assert ta["result"] == "paused_for_approval"
    assert ta["requestIds"] == [] and ta["whilePaused"] == "no_retry"
    assert ta["respondWithin"] == SETTINGS["response_time"]
    assert ta["reasonCode"] == "FIN-0001.over_limit"


def test_missing_field_with_fail_closed_matches_and_is_recorded():
    ctx = _ctx()
    ctx["args"] = {}
    doc = _doc("FIN-0005", "block", lambda f: f["hidden"].update(onMissingData="fail_closed"))
    v = enforcement.check(ctx, [doc])
    assert v.result == "blocked"
    assert v.blocks[0]["missing"] == ["args.amount"]


def test_missing_field_with_fail_open_does_not_match_but_is_recorded():
    ctx = _ctx()
    ctx["args"] = {}
    doc = _doc("FIN-0005", "block", lambda f: f["hidden"].update(onMissingData="fail_open"))
    v = enforcement.check(ctx, [doc])
    assert v.result == "allowed" and v.to_agent is None
    assert v.evaluated[0]["matched"] is False and v.evaluated[0]["missing"] == ["args.amount"]


def test_unreadable_condition_blocks():
    doc = _doc("FIN-0006", "notify")
    doc["trigger"]["condition"] = {"field": "args.amount", "op": "bogus", "value": 1}
    v = enforcement.check(_ctx(), [doc])
    assert v.result == "blocked"
    assert v.to_agent["reasonCode"] == "FIN-0006.condition_unreadable"
    assert v.blocks[0]["unreadable"] is True


def test_comparison_error_at_run_time_blocks():
    v = enforcement.check(_ctx(amount="lots"), [_doc("FIN-0007", "notify")])
    assert v.result == "blocked"
    assert v.to_agent["reasonCode"] == "FIN-0007.condition_unreadable"


def test_different_checkpoint_is_ignored():
    v = enforcement.check(_ctx(checkpoint="message.send.before"), [_doc("FIN-0001", "block")])
    assert v.result == "allowed" and v.to_agent is None and v.evaluated == []


def test_condition_not_met_is_allowed():
    v = enforcement.check(_ctx(amount=100), [_doc("FIN-0001", "block")])
    assert v.result == "allowed" and v.to_agent is None and v.blocks == []


def test_sensitive_values_are_masked_in_matched_values():
    def sensitive(form):
        form["hidden"]["inputs"][0]["sensitive"] = True
    v = enforcement.check(_ctx(), [_doc("FIN-0008", "approve", sensitive)])
    mv = v.approvals[0]["matched_values"]
    assert mv["args.amount"] == "•••"
    assert mv["tool.name"] == "refund.issue"


def test_mask_helper():
    doc = _doc(mutate=lambda f: f["hidden"]["inputs"][2].update(sensitive=True))
    assert enforcement.mask({"agent.reason": "x", "args.amount": 5}, doc) == {"agent.reason": "•••", "args.amount": 5}


def test_limit_is_ignored_and_says_so():
    doc = _doc("FIN-0009", "notify", lambda f: f.update(limit={"on": True, "text": "only the refunds agent"}))
    v = enforcement.check(_ctx(), [doc])
    assert v.notifies[0]["limitIgnored"] is True
    plain = enforcement.check(_ctx(), [_doc("FIN-0010", "notify")])
    assert plain.notifies[0]["limitIgnored"] is False


def test_no_live_policies_is_allowed_with_no_message():
    v = enforcement.check(_ctx(), [])
    assert v.result == "allowed" and v.to_agent is None
    assert v.blocks == v.approvals == v.notifies == []


# ---- live_policies ---------------------------------------------------------------------------

class _Calls:
    def __init__(self, docs):
        self.docs, self.n = docs, 0

    def __call__(self, conn):
        self.n += 1
        return self.docs


@pytest.fixture(autouse=True)
def _fresh_cache():
    live_policies.invalidate()
    yield
    live_policies.invalidate()


def test_load_filters_invalid_and_caches(monkeypatch):
    good, bad = _doc("FIN-0001"), _doc("FIN-0002", mutate=lambda f: f.update(checked=None))
    calls = _Calls([good, bad, "not a dict"])
    monkeypatch.setattr(live_policies.repo, "live_documents", calls)
    monkeypatch.setattr(live_policies, "load_registry", lambda conn=None: REGISTRY)
    assert [d["id"] for d in live_policies.load(conn=object())] == ["FIN-0001"]
    live_policies.load(conn=object())
    assert calls.n == 1
    live_policies.invalidate()
    live_policies.load(conn=object())
    assert calls.n == 2


def test_load_failure_raises_policy_store_unavailable(monkeypatch):
    def boom(conn):
        raise RuntimeError("db down")
    monkeypatch.setattr(live_policies.repo, "live_documents", boom)
    monkeypatch.setattr(live_policies, "load_registry", lambda conn=None: REGISTRY)
    with pytest.raises(live_policies.PolicyStoreUnavailable):
        live_policies.load(conn=object())
