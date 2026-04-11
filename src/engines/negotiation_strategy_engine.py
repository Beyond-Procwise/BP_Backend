"""Negotiation strategy selection and adaptation engine.

Provides deterministic strategy selection, adaptation across rounds,
position generation with BATNA analysis, and policy-rail enforcement
(auto-approve, escalation thresholds) for procurement negotiations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NegotiationContext:
    """Context for a single negotiation interaction."""

    supplier_name: str
    order_value: float
    category: str
    supplier_history_count: int = 0
    alternative_quotes: int = 0
    round_number: int = 1
    urgency: str = "normal"
    quote_price: float = 0.0


@dataclass
class Strategy:
    """A named negotiation strategy with its parameters."""

    name: str
    target_discount: float
    fallback_name: str
    description: str
    tone: str


@dataclass
class NegotiationPosition:
    """A concrete negotiation position derived from a strategy and context."""

    target_price: float
    arguments: List[str]
    tone: str
    batna_analysis: str


@dataclass
class ContinueDecision:
    """Policy-rail decision on whether to continue, escalate, accept, or walk away."""

    action: str   # continue | escalate | accept | walk_away
    reason: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NegotiationStrategyEngine:
    """Select, adapt, and operationalise negotiation strategies.

    Class-level strategy constants define the six core playbooks available.
    All public methods are stateless and deterministic given the same inputs.
    """

    # ------------------------------------------------------------------
    # Six strategy constants
    # ------------------------------------------------------------------

    STRATEGY_COOPERATIVE = Strategy(
        name="cooperative",
        target_discount=0.05,
        fallback_name="competitive",
        description=(
            "Build on an established supplier relationship to reach a "
            "mutually beneficial outcome through open dialogue."
        ),
        tone="collaborative",
    )

    STRATEGY_COMPETITIVE = Strategy(
        name="competitive",
        target_discount=0.10,
        fallback_name="anchoring",
        description=(
            "Leverage market alternatives to drive the best price through "
            "direct price competition between qualified suppliers."
        ),
        tone="assertive",
    )

    STRATEGY_ANCHORING = Strategy(
        name="anchoring",
        target_discount=0.15,
        fallback_name="competitive",
        description=(
            "Open with an ambitious anchor price to set expectations early "
            "and create room for concessions toward the real target."
        ),
        tone="firm",
    )

    STRATEGY_BUNDLING = Strategy(
        name="bundling",
        target_discount=0.12,
        fallback_name="cooperative",
        description=(
            "Consolidate multiple categories or line items into a single "
            "negotiation to unlock volume-based discounts."
        ),
        tone="analytical",
    )

    STRATEGY_TIME_LEVERAGE = Strategy(
        name="time_leverage",
        target_discount=0.08,
        fallback_name="competitive",
        description=(
            "Apply deadline pressure to accelerate supplier concessions "
            "when procurement urgency is high."
        ),
        tone="urgent",
    )

    STRATEGY_COLLABORATIVE_PROBLEM_SOLVING = Strategy(
        name="collaborative_problem_solving",
        target_discount=0.07,
        fallback_name="cooperative",
        description=(
            "When a supplier is unwilling to move on price, explore joint "
            "cost-reduction opportunities such as spec changes, payment terms, "
            "or logistics optimisation."
        ),
        tone="collaborative",
    )

    # Lookup by name for adaptation
    _STRATEGY_MAP = {
        "cooperative": None,            # set after class body
        "competitive": None,
        "anchoring": None,
        "bundling": None,
        "time_leverage": None,
        "collaborative_problem_solving": None,
    }

    def __init__(
        self,
        max_rounds: int = 3,
        auto_approve_threshold: float = 5_000.0,
        escalation_threshold: float = 50_000.0,
    ) -> None:
        self.max_rounds = max_rounds
        self.auto_approve_threshold = auto_approve_threshold
        self.escalation_threshold = escalation_threshold

        # Build the name-to-instance lookup after __init__ so subclasses can
        # override constants before the map is frozen.
        self._strategy_map: dict[str, Strategy] = {
            s.name: s
            for s in (
                self.STRATEGY_COOPERATIVE,
                self.STRATEGY_COMPETITIVE,
                self.STRATEGY_ANCHORING,
                self.STRATEGY_BUNDLING,
                self.STRATEGY_TIME_LEVERAGE,
                self.STRATEGY_COLLABORATIVE_PROBLEM_SOLVING,
            )
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_strategy(self, ctx: NegotiationContext) -> Strategy:
        """Choose the best-fit strategy for the given negotiation context.

        Selection priority (first matching rule wins):
        1. Established relationship (history >= 5) → cooperative
        2. Competitive market (alternatives >= 2) → competitive
        3. Unknown supplier with no alternatives → anchoring
        4. Multi-category order → bundling
        5. High urgency → time_leverage
        6. Default → competitive
        """
        if ctx.supplier_history_count >= 5:
            logger.debug("select_strategy: cooperative (history=%d)", ctx.supplier_history_count)
            return self.STRATEGY_COOPERATIVE

        if ctx.alternative_quotes >= 2:
            logger.debug("select_strategy: competitive (alternatives=%d)", ctx.alternative_quotes)
            return self.STRATEGY_COMPETITIVE

        if ctx.supplier_history_count == 0 and ctx.alternative_quotes == 0:
            logger.debug("select_strategy: anchoring (no history, no alternatives)")
            return self.STRATEGY_ANCHORING

        # Multiple categories detected: comma-separated or list-like value
        if "," in str(ctx.category):
            logger.debug("select_strategy: bundling (multiple categories)")
            return self.STRATEGY_BUNDLING

        if ctx.urgency == "high":
            logger.debug("select_strategy: time_leverage (urgency=high)")
            return self.STRATEGY_TIME_LEVERAGE

        logger.debug("select_strategy: competitive (default)")
        return self.STRATEGY_COMPETITIVE

    def adapt_strategy(self, current: Strategy, round_result: dict) -> Strategy:
        """Adapt the strategy based on the outcome of the most recent round.

        Rules:
        - Supplier made no movement AND we are at or past round 3
          → switch to collaborative_problem_solving
        - Supplier made any positive movement → keep the current strategy
        """
        supplier_movement = round_result.get("supplier_movement", 0)
        round_number = round_result.get("round_number", 1)

        if supplier_movement == 0 and round_number >= 3:
            logger.debug(
                "adapt_strategy: switching to collaborative_problem_solving "
                "(no movement at round %d)",
                round_number,
            )
            return self.STRATEGY_COLLABORATIVE_PROBLEM_SOLVING

        if supplier_movement > 0:
            logger.debug(
                "adapt_strategy: keeping %s (supplier moved %s)",
                current.name,
                supplier_movement,
            )
            return current

        # Default: keep current strategy
        return current

    def generate_position(
        self, strategy: Strategy, ctx: NegotiationContext
    ) -> NegotiationPosition:
        """Generate a concrete negotiation position from the strategy and context.

        The target price is the order value after the strategy discount.
        Arguments are tailored to the strategy name.
        BATNA analysis is derived from the number of alternative quotes.
        """
        target_price = round(ctx.order_value * (1.0 - strategy.target_discount), 2)

        arguments = self._build_arguments(strategy, ctx)

        tone = strategy.tone

        batna_analysis = self._build_batna(ctx)

        return NegotiationPosition(
            target_price=target_price,
            arguments=arguments,
            tone=tone,
            batna_analysis=batna_analysis,
        )

    def should_continue(self, ctx: NegotiationContext) -> ContinueDecision:
        """Evaluate policy rails to decide whether negotiation should proceed.

        Policy hierarchy (first matching rule wins):
        1. Rounds exhausted → escalate
        2. Low-value order → accept (auto-approve)
        3. High-value order → escalate (requires senior approval)
        4. Otherwise → continue
        """
        if ctx.round_number > self.max_rounds:
            return ContinueDecision(
                action="escalate",
                reason=(
                    f"Maximum rounds ({self.max_rounds}) exceeded at round "
                    f"{ctx.round_number}. Escalating to senior buyer."
                ),
            )

        if ctx.order_value < self.auto_approve_threshold:
            return ContinueDecision(
                action="accept",
                reason=(
                    f"Order value {ctx.order_value:.2f} is below the "
                    f"auto-approve threshold {self.auto_approve_threshold:.2f}. "
                    "Accepting current quote."
                ),
            )

        if ctx.order_value > self.escalation_threshold:
            return ContinueDecision(
                action="escalate",
                reason=(
                    f"Order value {ctx.order_value:.2f} exceeds the "
                    f"escalation threshold {self.escalation_threshold:.2f}. "
                    "Requires senior approval before proceeding."
                ),
            )

        return ContinueDecision(
            action="continue",
            reason=(
                f"Round {ctx.round_number} of {self.max_rounds}. "
                "Order value within normal negotiation range."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_arguments(self, strategy: Strategy, ctx: NegotiationContext) -> List[str]:
        """Return a list of negotiation arguments suited to the strategy."""

        name = strategy.name

        if name == "cooperative":
            return [
                f"We have completed {ctx.supplier_history_count} successful orders "
                f"with {ctx.supplier_name} and value this partnership.",
                "A price adjustment supports a long-term, mutually beneficial relationship.",
                "We are open to discussing extended contract terms in exchange for better pricing.",
            ]

        if name == "competitive":
            alts = ctx.alternative_quotes
            return [
                f"We have received {alts} alternative quote(s) for this requirement.",
                "Market benchmarks indicate a more competitive price is achievable.",
                "We are prepared to award to the supplier offering the best value proposition.",
            ]

        if name == "anchoring":
            discount_pct = int(strategy.target_discount * 100)
            return [
                f"Our target price reflects a {discount_pct}% reduction from the quoted price.",
                "This anchor is based on internal cost models and category benchmarks.",
                "We invite the supplier to present their cost breakdown to align expectations.",
            ]

        if name == "bundling":
            return [
                f"We are consolidating spend across multiple categories: {ctx.category}.",
                "Bundling creates volume efficiencies that should be reflected in pricing.",
                "A single consolidated order reduces administrative costs for both parties.",
            ]

        if name == "time_leverage":
            return [
                "This requirement is urgent and must be fulfilled within the current period.",
                "Suppliers who can confirm pricing immediately will be prioritised for award.",
                "Expedited order placement is contingent on agreement within this round.",
            ]

        if name == "collaborative_problem_solving":
            return [
                "We recognise the supplier's pricing constraints and seek a creative solution.",
                "We are open to adjusting specifications, payment terms, or delivery schedule.",
                "Joint cost reduction benefits both parties and secures a longer-term agreement.",
            ]

        # Generic fallback
        return [
            "We are seeking the best total cost of ownership for this requirement.",
            "Please provide your most competitive offer.",
        ]

    @staticmethod
    def _build_batna(ctx: NegotiationContext) -> str:
        """Construct a plain-text BATNA (Best Alternative to a Negotiated Agreement) statement."""

        if ctx.alternative_quotes >= 2:
            return (
                f"BATNA: {ctx.alternative_quotes} qualified alternative suppliers have been "
                "identified and are ready to receive an award. Failure to agree on price "
                "will result in the order being placed with an alternative."
            )

        if ctx.alternative_quotes == 1:
            return (
                "BATNA: One alternative supplier quote is available. Inability to reach "
                "agreement may result in a switch to the alternative supplier."
            )

        if ctx.supplier_history_count == 0:
            return (
                "BATNA: No established relationship or alternative quotes exist. "
                "The buyer will pursue market sourcing if agreement is not reached."
            )

        return (
            f"BATNA: Continued business with {ctx.supplier_name} is contingent on "
            "competitive pricing. Alternative sourcing will be explored if needed."
        )
