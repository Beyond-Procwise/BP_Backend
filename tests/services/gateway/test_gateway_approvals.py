"""get_approval_chain reads approval rules from the policy engine only; it states its gaps and never invents a chain."""
import datetime as dt

import pytest

from src.engines.policy_engine import PolicyStoreUnavailable
from src.services.gateway import Backends, CallerContext, build_default_gateway

THRESHOLD = {"policyId": "approval_threshold", "policyName": "ApprovalThresholdPolicy", "raw_row": {"version": 1},
             "details": {"rules": {"effect": "allow", "currency": "GBP", "on_above": "escalate", "on_at_or_below": "approve",
                                   "default_threshold_gbp": 10000},
                         "applies_to": ["spend.approve"], "required_role": "Approver", "policy_identifier": "approval_threshold"}}
CHAIN = {"policyId": "approval_chain", "policyName": "ApprovalChainPolicy", "raw_row": {"version": 2},
         "details": {"policy_identifier": "approval_chain", "rules": {"chains": [
             {"doc_type": "po", "business_unit": "*", "currency": "GBP",
              "steps": [{"min_amount": 0, "max_amount": 10000, "approvers": ["Buyer"]},
                        {"min_amount": 10000, "max_amount": None, "approvers": ["Buyer", "Approver"]}]},
             {"doc_type": "po", "business_unit": "IT", "currency": "GBP",
              "steps": [{"min_amount": 0, "max_amount": None, "approvers": ["Buyer", "Approver", "CFO"]}]}]}}}


class Engine:
    def __init__(self, threshold=(THRESHOLD,), chain=None, down=False):
        self.threshold, self.chain, self.down = list(threshold), chain, down

    def policies_for_action(self, action):
        if self.down:
            raise PolicyStoreUnavailable("down")
        return list(self.threshold) if action == "spend.approve" else []

    def get_policy(self, slug):
        if self.down:
            raise PolicyStoreUnavailable("down")
        return self.chain if slug == "approval_chain" else None


def call(engine, **p):
    params = {"doc_type": "po", "amount": 80000, "currency": "GBP", "business_unit": "IT"}
    params.update(p)
    g = build_default_gateway(Backends(extra={"policy_engine": engine}))
    return g, g.call(CallerContext("s", "Buyer", ("B1",), agent="ask_eligibility"), "get_approval_chain", params)


def test_above_the_threshold_escalates_to_the_required_role_and_states_there_is_no_chain():
    _, r = call(Engine())
    d = r.data
    assert r.ok and d["threshold"]["outcome"] == "escalate" and d["threshold"]["required_role"] == "Approver"
    assert d["threshold"]["threshold"] == 10000 and d["chain"] is None
    assert any("no approval-chain policy" in g for g in d["gaps"])
    assert r.provenance[0]["table"] == "bp_policy" and "approval_threshold" in r.provenance[0]["record_ids"]


@pytest.mark.parametrize("amount,outcome", [(5000, "approve"), (10000, "approve"), (10000.01, "escalate")])
def test_the_boundary_is_at_or_below_approves(amount, outcome):
    assert call(Engine(), amount=amount)[1].data["threshold"]["outcome"] == outcome


def test_a_currency_the_threshold_is_not_written_in_is_not_converted_or_tested():
    d = call(Engine(), currency="USD")[1].data
    assert d["threshold"]["outcome"] is None and any("not converted" in g for g in d["gaps"])


def test_a_chain_policy_gives_the_steps_for_the_amount_and_the_most_specific_chain_wins():
    d = call(Engine(chain=CHAIN), business_unit="IT", amount=500)[1].data
    assert [s["approver"] for s in d["chain"]] == ["Buyer", "Approver", "CFO"]            # the IT chain, not the wildcard one
    assert [s["step"] for s in d["chain"]] == [1, 2, 3]
    d = call(Engine(chain=CHAIN), business_unit="Ops", amount=500)[1].data
    assert [s["approver"] for s in d["chain"]] == ["Buyer"]
    d = call(Engine(chain=CHAIN), business_unit="Ops", amount=50000)[1].data
    assert [s["approver"] for s in d["chain"]] == ["Buyer", "Approver"]


def test_a_chain_policy_with_no_matching_entry_is_a_stated_gap_not_a_guess():
    d = call(Engine(chain=CHAIN), doc_type="invoice")[1].data
    assert d["chain"] is None and any("invoice" in g for g in d["gaps"])


def test_no_approval_policy_at_all_is_an_honest_empty_answer_with_the_search_recorded():
    _, r = call(Engine(threshold=()))
    assert r.ok and r.data["threshold"] is None and r.data["chain"] is None and r.data["gaps"] and r.provenance


def test_two_different_thresholds_are_a_gap_not_a_pick():
    other = {**THRESHOLD, "policyId": "other", "details": {**THRESHOLD["details"], "rules": {**THRESHOLD["details"]["rules"], "default_threshold_gbp": 500}}}
    d = call(Engine(threshold=(THRESHOLD, other)))[1].data
    assert d["threshold"] is None and any("more than one" in g for g in d["gaps"])


def test_a_policy_store_outage_is_unavailable_never_no_rule():
    _, r = call(Engine(down=True))
    assert not r.ok and r.error.code == "UNAVAILABLE"


def test_a_past_date_is_answered_with_the_policy_in_force_now_and_says_so():
    d = call(Engine(), as_of=(dt.date.today() - dt.timedelta(days=90)).isoformat())[1].data
    assert any("in force now" in g for g in d["gaps"])


def test_a_negative_amount_and_a_missing_currency_are_rejected():
    assert call(Engine(), amount=-1)[1].error.code == "SCHEMA_INVALID"
    assert call(Engine(), currency="")[1].error.code == "SCHEMA_INVALID"


def test_evaluate_status_stays_unsupported_and_names_what_it_waits_on():
    g = build_default_gateway(Backends(extra={"policy_engine": Engine()}))
    r = g.call(CallerContext("s", "Buyer", None), "evaluate_approval_status", {"doc_type": "po", "doc_id": "PO1"})
    assert not r.ok and r.error.code == "UNSUPPORTED" and "approver" in r.error.message.lower() and "chain" in r.error.message.lower()


def test_the_hard_coded_screen_copy_is_not_a_second_source():
    import src.services.gateway.approvals as mod
    src = open(mod.__file__).read()
    assert "import approval_route" not in src and "approval_route import" not in src and "services.approval_route" not in src


class HalfDownEngine(Engine):
    """Only the threshold lookup fails; the chain lookup works. An outage in either must stay an outage."""

    def policies_for_action(self, action):
        raise PolicyStoreUnavailable("threshold store down")


class HalfDownChainEngine(Engine):
    def get_policy(self, slug):
        raise PolicyStoreUnavailable("chain store down")


@pytest.mark.parametrize("engine", [HalfDownEngine(), HalfDownChainEngine()])
def test_an_outage_in_either_policy_lookup_alone_is_still_unavailable(engine):
    _, r = call(engine)
    assert not r.ok and r.error.code == "UNAVAILABLE"
