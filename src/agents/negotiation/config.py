"""What this agent is allowed to do, and the knobs that shape how it says it.

Split out of negotiation_agent.py so the helpers that read these values can live
beside the work they support instead of importing the whole agent back.

The bounds are governed rather than configured: an agent allowed to concede 40%
instead of 20% is a different agent, so a missing limit refuses rather than
assuming one. Each is a function, not a constant, because it is read when used --
read at import, it would be pinned before the governance store had answered.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.services.governed_limits import limit as _governed_limit

logger = logging.getLogger(__name__)


# -----------------------------
# Tunables & feature toggles
# -----------------------------
def MAX_SUPPLIER_REPLIES() -> int:
    """NegotiationBoundsPolicy (P9)."""
    return _governed_limit("negotiation_bounds", "max_supplier_replies",
                           env="NEG_MAX_SUPPLIER_REPLIES", cast=int)
LLM_ENABLED = os.getenv("NEG_ENABLE_LLM", "1").strip() not in {"0", "false", "False"}
# AgentNick, like every other non-extraction caller. The old default was
# "llama3.2:latest", which is not a model this host has ever had installed, so the one
# call that uses it (_llm_* in the counter composer) answered 404 on every attempt,
# retried three times, and fell through to the non-LLM path -- silently, because the
# fallback looks like an ordinary result. NEG_LLM_MODEL still overrides, and
# AGENTNICK_MODEL is the same knob src/services/tool_runtime.py reads.
LLM_MODEL = os.getenv("NEG_LLM_MODEL") or os.getenv(
    "AGENTNICK_MODEL", "BeyondProcwise/AgentNick:unified"
)
def COST_OF_CAPITAL_APR() -> float:
    """NegotiationBoundsPolicy (P9)."""
    return _governed_limit("negotiation_bounds", "cost_of_capital_apr",
                           env="NEG_COST_OF_CAPITAL_APR")
def LEAD_TIME_VALUE_PCT_PER_WEEK() -> float:
    """NegotiationBoundsPolicy (P9)."""
    return _governed_limit("negotiation_bounds", "lt_value_pct_per_week",
                           env="NEG_LT_VALUE_PCT_PER_WEEK")
def _resolve_thread_transcript_limit() -> Optional[int]:
    """How many thread entries the agent reads before it counters, or ``None``
    for the full history. AgentReachPolicy (P9).

    Policy states null for "no limit", which is a decision somebody made -- not
    the same as the key being absent, which refuses. Zero or less also means the
    full history, as it always has.

    NEG_THREAD_TRANSCRIPT_LIMIT overrides for one release through
    governed_limits, which warns when it disagrees with policy and ignores it in
    favour of policy when it is not a number. It used to turn an unreadable
    value into "full history": how much the agent sees before making an offer,
    decided by a typo.

    Read on every use rather than once at import, where it ran before anything
    knew whether the governance store had answered, and pinned the value until
    the process restarted.
    """
    limit = _governed_limit("agent_reach", "neg_thread_transcript_limit",
                            env="NEG_THREAD_TRANSCRIPT_LIMIT", cast=int)
    if limit is None or limit <= 0:
        return None
    return limit


def AGGRESSIVE_FIRST_COUNTER_PCT() -> float:
    """NegotiationBoundsPolicy (P9)."""
    return _governed_limit("negotiation_bounds", "first_counter_aggr_pct",
                           env="NEG_FIRST_COUNTER_AGGR_PCT")
FINAL_OFFER_PATTERNS = (
    "best and final",
    "best final",
    "best offer we can",
    "best price we can",
    "cannot go lower",
    "final offer",
    "final price",
    "final quotation",
    "last price",
    "lowest we can do",
    "our best price",
    "rock bottom",
    "take it or leave it",
    "ultimatum",
)

#: The decisions that end a negotiation. `plan_counter` returns one of these when
#: it has resolved the outcome -- an offer inside our threshold is accepted, one
#: above it is declined -- and `_execute_negotiation_round` closes the supplier on
#: them. Both readings must agree, so they share this definition rather than each
#: spelling out the pair: they did not agree before, and a decision to accept was
#: being reopened downstream (see `_adaptive_strategy`).
TERMINAL_STRATEGIES = frozenset({"accept", "decline"})

#: The key `resolve_authority` files this agent's mandate under, and the name in
#: EmailReplyAutonomyPolicy.policy_linked_agents. Must match both.
AUTHORITY_AGENT_KEY = "negotiation_agent"

# LEVER_CATEGORIES and TRADE_OFF_HINTS now live in
# services.negotiation_advice.ranking, next to the scoring that uses them.


# NegotiationBoundsPolicy (P9): what this agent may put to a supplier. An agent
# allowed to concede 40% instead of 20% is a different agent, so these are rules
# rather than settings, and a missing one refuses rather than assuming.
def MARKET_REVIEW_THRESHOLD() -> float:
    return _governed_limit("negotiation_bounds", "market_review_pct",
                           env="NEG_MARKET_REVIEW_PCT")


def MARKET_ESCALATION_THRESHOLD() -> float:
    return _governed_limit("negotiation_bounds", "market_escalation_pct",
                           env="NEG_MARKET_ESCALATION_PCT")


def MAX_VOLUME_LIMIT() -> float:
    return _governed_limit("negotiation_bounds", "max_volume_limit",
                           env="NEG_MAX_VOLUME_LIMIT")


def MAX_TERM_DAYS() -> int:
    return _governed_limit("negotiation_bounds", "max_term_days",
                           env="NEG_MAX_TERM_DAYS", cast=int)

DEFAULT_NEGOTIATION_MESSAGE_TEMPLATE = "{header}\n{details}{context_sections}"

BATCH_INPUT_KEYS = ("negotiation_batch", "supplier_responses_batch", "batch_responses")
BATCH_SHARED_KEYS = (
    "shared_context",
    "shared_payload",
    "batch_defaults",
    "shared_fields",
    "defaults",
)
BATCH_EXCLUDE_KEYS = {
    "negotiation_batch",
    "supplier_responses_batch",
    "batch_responses",
    "shared_context",
    "shared_payload",
    "batch_defaults",
    "shared_fields",
    "defaults",
    "batch_metadata",
    "batch_results",
    "batch_summary",
    "agentic_plan",
    "pass_fields",
    "results",
    "drafts",
    "supplier_responses",
}
