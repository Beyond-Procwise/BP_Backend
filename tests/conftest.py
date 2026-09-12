import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# The governed limits (P9).
#
# Thirty-three limits are read from proc.bp_policy now, and a missing one
# REFUSES rather than falling back to a number carried in the code -- which is
# the whole point of the change. The in-memory database the suite runs against
# cannot serve policies at all (PolicyEngine does `with conn.cursor()`, and the
# fake cursor is not a context manager), so without this every test that touches
# a threshold would fail on a governance store that is simply not there.
#
# This seeds the same values the migration writes. It is a COPY, and a copy can
# drift into proving a value the product does not use, so
# tests/governance/test_governed_limits.py compares it against the LIVE rows and
# fails if they disagree. That comparison is the only thing that makes a second
# copy safe to have.
#
# Tests about the limit machinery itself patch `governed_limits._engine`
# directly; being function-scoped they are applied after this and win.
# ---------------------------------------------------------------------------
GOVERNED_LIMIT_SEED = {
    "promotion_thresholds": {
        "promote_min_confidence": 50, "promote_min_link_score": 80,
        "promote_review_min": 65, "propose_min_link_score": 40,
        "propose_max_candidates": 5, "quote_anchor_min_score": 80},
    "reconciliation_tolerances": {
        "amount_tolerance_pct": 0.01, "amount_tolerance_abs": 1.00,
        "tax_tolerance_pct": 0.1},
    "supplier_identity": {
        "review_low": 82, "review_high": 96, "sweep_min_score": 88,
        "research_name_match": 85, "research_propose_conf": 0.75},
    "negotiation_bounds": {
        "max_volume_limit": 1000, "max_term_days": 120,
        "max_supplier_replies": 3, "first_counter_aggr_pct": 0.12,
        "market_review_pct": 0.2, "market_escalation_pct": 0.4,
        "lt_value_pct_per_week": 0.01, "cost_of_capital_apr": 0.12},
    "agent_reach": {
        "max_dynamic_agents": 3, "tool_runtime_max_rounds": 6,
        "governed_reasoning_max_rounds": 5, "supplier_research_max_rounds": 4,
        "neg_thread_transcript_limit": None},
    "extraction_effort": {"judge_max_calls": 12, "judge_budget_s": 25},
    "autonomous_operation": {
        "opportunity_mining_min_impact": 100, "capture_retention_days": 30,
        "duplicate_invoice_detector_enabled": False,
        "supplier_research_enabled": True},
    "reseller_catalog": {"fuzzy_propose_min": 88, "calibration_min_closed": 30},
}


class _SeededPolicyEngine:
    def get_policy(self, slug):
        rules = GOVERNED_LIMIT_SEED.get(slug)
        if rules is None:
            return None
        return {"policyName": slug, "details": {"policy_identifier": slug,
                                                "rules": dict(rules)}}


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _governed_limits_available(monkeypatch):
    from src.services import governed_limits

    governed_limits.reset_cache()
    monkeypatch.setattr(governed_limits, "_engine", lambda: _SeededPolicyEngine())
    yield
    governed_limits.reset_cache()
