"""Verification network: declarative cross-field rules.

A rule is a function over the committed values dict. If it fails,
the rule's `on_fail` action determines the consequence:
    - "demote": reduce confidence on participating fields
    - "abstain": eject participating fields from the commit set
    - "warn":   log only

The network runs AFTER consensus voting and BEFORE persistence. It is
the second-level guard that catches "consistent but wrong" extractions
(e.g., subtotal that doesn't match line-items sum).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


__all__ = ["Rule", "RuleResult", "VerificationOutcome", "run_verification"]


@dataclass(frozen=True)
class Rule:
    """One verification rule."""
    name: str
    fields: tuple[str, ...]
    check: Callable[[dict[str, Any]], bool]
    on_fail: Literal["demote", "abstain", "warn"] = "demote"
    why: str = ""


@dataclass(frozen=True)
class RuleResult:
    rule_name: str
    passed: bool
    fields: tuple[str, ...]
    on_fail: str
    why: str


@dataclass
class VerificationOutcome:
    """Per-field demotions / abstentions and the rule trace."""
    rule_results: list[RuleResult] = field(default_factory=list)
    demoted_fields: set[str] = field(default_factory=set)
    abstained_fields: set[str] = field(default_factory=set)


def run_verification(
    values: dict[str, Any],
    rules: list[Rule],
) -> VerificationOutcome:
    """Run all rules. Returns per-field demotions and abstentions."""
    outcome = VerificationOutcome()
    for rule in rules:
        # Skip rule if any required field is missing
        if not all(f in values and values[f] is not None for f in rule.fields):
            continue
        try:
            passed = bool(rule.check(values))
        except Exception:
            logger.exception("rule %r raised; treating as failed", rule.name)
            passed = False

        result = RuleResult(
            rule_name=rule.name,
            passed=passed,
            fields=rule.fields,
            on_fail=rule.on_fail,
            why=rule.why if not passed else "ok",
        )
        outcome.rule_results.append(result)

        if not passed:
            if rule.on_fail == "demote":
                outcome.demoted_fields.update(rule.fields)
            elif rule.on_fail == "abstain":
                outcome.abstained_fields.update(rule.fields)
            elif rule.on_fail == "warn":
                logger.warning("rule %r failed (warn): %s", rule.name, rule.why)

    return outcome
