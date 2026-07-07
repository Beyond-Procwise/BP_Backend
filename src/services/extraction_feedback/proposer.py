"""Proposer: turn recurring per-vendor extraction failures into hint proposals.

Reads proc.bp_extraction_telemetry (each row carries vendor_hint == vendor_key,
doc_type, missing_required, and a discrepancy_types {issue_type: count} map).
Aggregates per (doc_type, vendor, signal); a signal that recurs across enough
documents at a high enough rate becomes a PENDING proposal (drafted by the local
:unified model, with a deterministic fallback). Nothing is applied here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from collections import defaultdict

from src.services.agent_actions import PHASE_EXTRACTION, record_action
from src.services.db import get_conn

log = logging.getLogger(__name__)

_SUMMARY_MODEL = os.getenv("PROCWISE_SUMMARY_MODEL", "BeyondProcwise/AgentNick:unified")


def _parse_missing(raw) -> set[str]:
    """missing_required is stored as text; tolerate JSON, python-repr, or CSV."""
    if not raw or raw in ("[]", ""):
        return set()
    if isinstance(raw, (list, tuple)):
        return {str(x).strip() for x in raw if str(x).strip()}
    for loader in (json.loads, ast.literal_eval):
        try:
            val = loader(raw)
            if isinstance(val, (list, tuple)):
                return {str(x).strip() for x in val if str(x).strip()}
        except Exception:
            pass
    return {p.strip() for p in str(raw).split(",") if p.strip()}


def _fetch_signals(window_days: int) -> list[dict]:
    rows: list[dict] = []
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT doc_type, vendor_hint, doc_pk, missing_required, discrepancy_types "
            "FROM proc.bp_extraction_telemetry "
            "WHERE captured_at >= now() - (%s || ' days')::interval "
            "AND vendor_hint IS NOT NULL AND doc_type IS NOT NULL",
            (str(window_days),),
        )
        for doc_type, vendor, doc_pk, missing, disc in cur.fetchall():
            disc_map = disc if isinstance(disc, dict) else (json.loads(disc) if disc else {})
            rows.append({
                "doc_type": doc_type,
                "vendor_key": vendor,
                "doc_pk": doc_pk or "",
                "missing": _parse_missing(missing),
                "issues": {str(k) for k in (disc_map or {}).keys()},
            })
    return rows


def _scope_already_covered(cur, name: str, dedup: str) -> bool:
    cur.execute(
        "SELECT 1 FROM proc.bp_prompt WHERE prompt_name=%s "
        "AND prompt_type='extraction_vendor_hint' AND prompts_status=1 LIMIT 1",
        (name,),
    )
    if cur.fetchone():
        return True
    cur.execute("SELECT 1 FROM proc.bp_extraction_hint_proposal WHERE dedup_key=%s LIMIT 1", (dedup,))
    return cur.fetchone() is not None


def _fallback_hint(doc_type, vendor, signal, kind) -> str:
    return (f"For {vendor} {doc_type}s, '{signal}' is frequently problematic; locate it carefully in the "
            f"document (typically near the header or totals block) and capture it verbatim when present.")


def _draft_hint(doc_type, vendor, signal, kind, n, total, draft: bool) -> str:
    if not draft:
        return _fallback_hint(doc_type, vendor, signal, kind)
    prompt = (
        "You are improving a procurement document extractor. For supplier "
        f"'{vendor}' {doc_type} documents, the field/issue '{signal}' was problematic in "
        f"{n} of {total} recent documents ({kind}). Write ONE concise imperative hint "
        f"(max 30 words) telling the extractor what to look for or how to capture '{signal}' "
        "correctly for this supplier's layout. Output ONLY the hint sentence."
    )
    try:
        from src.services.ollama_client import ollama_generate
        txt = ollama_generate(prompt, model=_SUMMARY_MODEL, think=False,
                              temperature=0.2, num_predict=80, timeout=60, retries=1)
        if txt and txt.strip():
            return txt.strip().splitlines()[0][:400]
    except Exception as exc:  # noqa: BLE001
        log.debug("proposer: LLM draft failed, using fallback: %s", exc)
    return _fallback_hint(doc_type, vendor, signal, kind)


def propose_all(window_days: int = 14, min_docs: int = 3, min_fail_rate: float = 0.5,
                draft: bool = True, vendors: set[str] | None = None) -> list[int]:
    """Scan telemetry and create pending hint proposals. Returns new proposal ids.

    ``vendors`` (optional) restricts processing to those vendor keys — used by
    tests to stay hermetic; production passes None (all vendors).
    """
    rows = _fetch_signals(window_days)
    totals: dict[tuple, int] = defaultdict(int)
    sig_docs: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if vendors is not None and r["vendor_key"] not in vendors:
            continue
        totals[(r["doc_type"], r["vendor_key"])] += 1
        for f in r["missing"]:
            sig_docs[(r["doc_type"], r["vendor_key"], f, "missing")].add(r["doc_pk"])
        for it in r["issues"]:
            sig_docs[(r["doc_type"], r["vendor_key"], it, "discrepancy")].add(r["doc_pk"])

    created: list[int] = []
    for (doc_type, vendor, signal, kind), docs in sig_docs.items():
        total = totals[(doc_type, vendor)]
        n = len(docs)
        if total == 0 or n < min_docs or (n / total) < min_fail_rate:
            continue
        dedup = hashlib.sha256(f"{doc_type}|{vendor}|{signal}|{kind}".encode()).hexdigest()[:32]
        name = f"vhint::{doc_type}::{vendor}::{signal}"
        with get_conn() as c:
            with c.cursor() as cur:
                if _scope_already_covered(cur, name, dedup):
                    continue
            hint = _draft_hint(doc_type, vendor, signal, kind, n, total, draft)
            evidence = {
                "doc_ids": sorted(d for d in docs if d)[:20],
                "sample_count": total, "missing_count": n,
                "failure_rate": round(n / total, 3), "signal": signal, "kind": kind,
            }
            rationale = (f"'{signal}' occurred in {n}/{total} recent {vendor} {doc_type} documents "
                         f"({round(100 * n / total)}%).")
            with c.cursor() as cur:
                cur.execute(
                    "INSERT INTO proc.bp_extraction_hint_proposal "
                    "(doc_type, vendor_key, field_name, dedup_key, evidence, proposed_hint, rationale, status) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,'pending') RETURNING proposal_id",
                    (doc_type, vendor, signal, dedup, json.dumps(evidence), hint, rationale),
                )
                pid = cur.fetchone()[0]
            c.commit()
        created.append(pid)
        record_action(
            phase=PHASE_EXTRACTION, action_type="hint_proposed", agent="extraction_feedback",
            doc_type=doc_type, field_name=signal,
            summary=f"proposed hint {pid} for {vendor}/{doc_type}/{signal}",
            details=evidence,
        )
    log.info("proposer: created %d hint proposals", len(created))
    return created
