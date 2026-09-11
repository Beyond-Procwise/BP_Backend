"""Fetch the critic's governed inputs. Never default them.

`approvals_agent` records what happens when a governed threshold falls back to
a constant: the agent returned SUCCESS while every threshold silently became a
hardcoded 1000. An unresolvable threshold here becomes None, and the agent
degrades that test to UNASSESSED rather than inventing a number.

Both engines are read in the shape they actually return, not the shape of a
bp_ row. PolicyEngine normalises a policy's rules under ``details`` and keeps
the original row under ``raw_row``; PromptEngine.get_prompt() only accepts a
numeric id, so the prompt is found by name. A loader written against a raw row
reads an empty rule set from every live policy and never finds the prompt --
and a critic without its prompt refuses to run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

POLICY_SLUG = "opportunity_critic_thresholds"
PROMPT_SLUG = "opportunity_critic_system"


@dataclass(frozen=True)
class Thresholds:
    """The governed numbers, exactly as resolved. None means unresolved."""

    index_band_pp: Optional[float] = None
    materiality_floor_gbp: Optional[float] = None
    relative_gap_floor: Optional[float] = None
    anchor_stale_days: Optional[int] = None
    friction_bands: Dict[str, float] = field(default_factory=dict)
    shadow_detectors: Tuple[Dict[str, Any], ...] = ()
    source: Optional[Dict[str, Any]] = None


def _rules(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    # "details" on a PolicyEngine policy, "policy_details" on a raw bp_policy row.
    details = policy.get("details") or policy.get("policy_details") or {}
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules") or {}
    return rules if isinstance(rules, dict) else {}


def _source(policy: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(policy, dict):
        return None
    row = policy.get("raw_row")
    row = row if isinstance(row, dict) else policy
    return {"policy_id": row.get("policy_id"), "version": row.get("version")}


def _opt_float(rules: Dict[str, Any], key: str) -> Optional[float]:
    value = rules.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("critic threshold %s is not numeric: %r", key, value)
        return None


def load_thresholds(policy_engine: Any = None) -> Thresholds:
    """Resolve the governed thresholds. Missing fields stay None."""
    policy = None
    if policy_engine is not None:
        try:
            policy = policy_engine.get_policy(POLICY_SLUG)
        except Exception as exc:  # noqa: BLE001 - an unreadable policy is not a crash
            logger.error("critic policy %s unreadable: %s", POLICY_SLUG, exc)
            policy = None

    rules = _rules(policy)
    bands = rules.get("friction_bands") or {}
    shadow = rules.get("shadow_detectors") or []
    stale = rules.get("anchor_stale_days")

    return Thresholds(
        index_band_pp=_opt_float(rules, "index_band_pp"),
        materiality_floor_gbp=_opt_float(rules, "materiality_floor_gbp"),
        relative_gap_floor=_opt_float(rules, "relative_gap_floor"),
        anchor_stale_days=int(stale) if stale is not None else None,
        friction_bands={str(k): float(v) for k, v in bands.items()
                        if isinstance(v, (int, float))},
        shadow_detectors=tuple(s for s in shadow if isinstance(s, dict)),
        source=_source(policy),
    )


def load_system_prompt(prompt_engine: Any = None) -> Tuple[Optional[str], Optional[int]]:
    """Return ``(prompt_text, version)`` from the governed prompt row.

    ``(None, None)`` when it cannot be resolved. The agent must refuse to run
    rather than fall back to a prompt baked into code: the DB prompt is the
    system of record and a code default that silently wins is how governance
    stops being governance.
    """
    if prompt_engine is None:
        return None, None
    try:
        prompts = prompt_engine.all_prompts()
    except Exception as exc:  # noqa: BLE001
        logger.error("critic prompt %s unreadable: %s", PROMPT_SLUG, exc)
        return None, None
    row = next((p for p in prompts or []
                if isinstance(p, dict)
                and (p.get("promptName") or p.get("prompt_name")) == PROMPT_SLUG), None)
    if row is None:
        return None, None
    text = row.get("template")
    if not text:
        desc = row.get("prompts_desc")
        text = desc.get("prompt_template") if isinstance(desc, dict) else None
    return (text or None), row.get("version")
