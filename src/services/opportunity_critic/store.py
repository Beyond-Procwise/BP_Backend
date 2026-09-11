"""Persist a critique and its gap register.

The return value is load-bearing. ``record_critique`` returns the new
``critique_id`` when it wrote, and ``None`` when it did not -- and the caller
must not suppress a finding on a ``None``. Suppressing without recording buys
neither the safety nor the measurement, which is precisely the reasoning at
services/guardrail.py:467, inverted.

Never raises. Bookkeeping must not break the pipeline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_INSERT_CRITIQUE = """
    INSERT INTO proc.bp_opportunity_critique (
        opportunity_ref_id, detector_type, verdict, confidence,
        original_claim, critic_claim, negotiator_note,
        detector_proposed, critic_addressable, currency, value_basis,
        haircuts, lever, duplicate_of, tests,
        would_have_suppressed, shadowed,
        prompt_version, policy_versions, formula_versions, run_id
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s
    ) RETURNING critique_id
"""

_INSERT_GAP = """
    INSERT INTO proc.bp_opportunity_gap (
        critique_id, opportunity_ref_id, gap_id, test, gap_type,
        what_is_missing, why_it_matters, blocking, resolves_to,
        likely_source, owner_hint, effort, ordinal
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

#: Verdicts that would remove a finding from the page once enforcement is on.
SUPPRESSING_VERDICTS = ("INVALID", "DUPLICATE")


def record_critique(critique: Dict[str, Any], *, shadowed: bool) -> Optional[int]:
    """Write one critique and its gaps. Return the id, or None if nothing was written."""
    value = critique.get("value") or {}
    verdict = str(critique.get("verdict") or "")
    would_suppress = verdict in SUPPRESSING_VERDICTS

    params = (
        critique.get("opportunity_ref_id"),
        critique.get("detector_type"),
        verdict,
        critique.get("confidence"),
        critique.get("original_claim"),
        critique.get("critic_claim"),
        critique.get("negotiator_note"),
        value.get("detector_proposed"),
        value.get("critic_addressable"),
        value.get("currency"),
        value.get("basis"),
        json.dumps(value.get("haircuts_applied") or [], default=str),
        json.dumps(critique.get("lever") or {}, default=str),
        critique.get("duplicate_of"),
        json.dumps(critique.get("tests") or [], default=str),
        would_suppress,
        bool(shadowed),
        critique.get("prompt_version"),
        json.dumps(critique.get("policy_versions") or {}, default=str),
        json.dumps(critique.get("formula_versions") or {}, default=str),
        critique.get("run_id"),
    )

    try:
        from src.services.db import get_conn

        with get_conn() as conn:
            conn.autocommit = False
            cur = conn.cursor()
            try:
                cur.execute(_INSERT_CRITIQUE, params)
                critique_id = cur.fetchone()[0]
                for ordinal, gap in enumerate(critique.get("gaps") or []):
                    cur.execute(_INSERT_GAP, (
                        critique_id,
                        critique.get("opportunity_ref_id"),
                        gap.get("gap_id"),
                        gap.get("test"),
                        gap.get("type"),
                        gap.get("what_is_missing"),
                        gap.get("why_it_matters"),
                        bool(gap.get("blocking")),
                        gap.get("resolves_to"),
                        gap.get("likely_source"),
                        gap.get("owner_hint"),
                        gap.get("effort"),
                        ordinal,
                    ))
                conn.commit()
                return critique_id
            except Exception:
                conn.rollback()
                raise
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break the pipeline
        logger.error("record_critique(%s) failed: %s",
                     critique.get("opportunity_ref_id"), exc)
        return None
