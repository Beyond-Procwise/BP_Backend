#!/usr/bin/env python
"""Critique live findings in shadow, a bounded batch at a time.

Why a script and not a workflow node: the critic judges one finding per run,
each run is a GPU tool loop of roughly 2-4 minutes, and the workflow engine has
no fan-out -- a node after mining would receive the whole findings list and
fail, and 308 findings would add 10-20 hours to every mining run. This runs on
demand instead, and is safe to stop and restart.

Three rules:

  * Shadow-only. It refuses to start unless every detector in the batch is
    actively enrolled in shadow, so a batch that calls itself an observation
    run cannot be anything else.
  * Resumable. A finding that already has a critique is skipped, so an
    interrupted batch picks up where it stopped.
  * Representative. The rarest detectors come first, so a small sample spans
    every detector family instead of being all duplicates (300 of 308 are).

Usage:
    ./.venv/bin/python scripts/critic_run.py --limit 20
    ./.venv/bin/python scripts/critic_run.py --limit 20 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

_BATCH_SQL = """
    SELECT opportunity_ref_id, detector_type, supplier_id, financial_impact_gbp,
           facts_state, source_records, calculation_details, currency,
           category_id, stage
      FROM (
        SELECT o.*, count(*) OVER (PARTITION BY o.detector_type) AS family_size
          FROM proc.bp_opportunity o
         WHERE o.retired_at IS NULL
           AND NOT EXISTS (SELECT 1 FROM proc.bp_opportunity_critique c
                            WHERE c.opportunity_ref_id = o.opportunity_ref_id)
      ) s
     ORDER BY family_size, opportunity_id
     LIMIT %s
"""


def _as_obj(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true",
                        help="list the batch and the shadow check, critique nothing")
    args = parser.parse_args()

    from agents.base_agent import AgentContext, AgentNick
    from src.agents.opportunity_critic_agent import OpportunityCriticAgent
    from src.services.db import get_conn
    from src.services.opportunity_critic.governed import load_thresholds
    from src.services.opportunity_critic.batch import (
        skip_embedding_model, skip_model_preload,
    )
    from src.services.opportunity_critic.shadow import unenrolled_detectors

    # Both before AgentNick(). Its preload asks Ollama for all 49 layers on a
    # card that holds 25, and the failed load plus reload is what stalled the
    # model; its sentence-transformer lands in RAM once the GPU is hidden, and
    # that is what the OOM killer took the last run for.
    skip_model_preload()
    skip_embedding_model()
    nick = AgentNick()
    thresholds = load_thresholds(nick.policy_engine)
    if thresholds.source is None:
        print("REFUSED: the critic's policy could not be resolved, so shadow "
              "enrolment cannot be confirmed.")
        return 2

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_BATCH_SQL, (args.limit,))
        cols = [d[0] for d in cur.description]
        findings = [dict(zip(cols, row)) for row in cur.fetchall()]
    for f in findings:
        f["source_records"] = _as_obj(f.get("source_records")) or []
        f["calculation_details"] = _as_obj(f.get("calculation_details")) or {}

    detectors = sorted({f["detector_type"] for f in findings})
    missing = unenrolled_detectors(detectors, thresholds)
    print(f"batch: {len(findings)} findings across {detectors}")
    print(f"policy: {thresholds.source}")
    if missing:
        print(f"REFUSED: not actively in shadow: {missing}. Enrol them, with an "
              "expiry, before a live run.")
        return 2
    if args.dry_run:
        for f in findings:
            print(f"  {f['opportunity_ref_id']}  {f['detector_type']}  "
                  f"{f['facts_state']}  stage={f['stage']}")
        return 0

    agent = OpportunityCriticAgent(nick)
    run_id = "critic-run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tally: dict = {}
    for i, finding in enumerate(findings, 1):
        started = time.monotonic()
        # A batch has no human starter, so it has no subject: user_id is None,
        # never a stand-in name that would read as someone's identity.
        ctx = AgentContext(workflow_id=run_id, agent_id="opportunity_critic",
                           user_id=None, input_data={"finding": finding})
        try:
            out = agent.run(ctx)
            status = getattr(out.status, "value", out.status)
            verdict = (out.data or {}).get("verdict")
            detail = (out.data or {}).get("critique_id") or out.error
        except Exception as exc:  # noqa: BLE001 - one bad finding must not end the batch
            status, verdict, detail = "crashed", None, repr(exc)
        key = verdict or status
        tally[key] = tally.get(key, 0) + 1
        print(f"[{i}/{len(findings)}] {finding['opportunity_ref_id']} "
              f"{finding['detector_type']}: {status} {verdict or ''} "
              f"({time.monotonic() - started:.0f}s) {detail}", flush=True)

    print(f"\nrun {run_id}: {tally}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
