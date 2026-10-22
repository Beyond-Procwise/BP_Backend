"""Seed proc.bp_orchestrator_registry from the tools the agent loop really offers.

Only tool.call.before is 'live'. Stage 3 builds the gate there, and it supplies exactly the
inputs registered here: tool.name, agent.name, agent.reason and each tool's own arguments.
The other three checkpoints are named so a policy can say where it belongs, but are
'planned': nothing checks them yet, so a policy pinned to one shows "Can't be enforced yet".

Run: PYTHONPATH=.:src ./venv/bin/python -m scripts.agent_policy.seed_registry [--apply]
Without --apply it prints the rows and writes nothing.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

# Tools build_tools() adds besides the agents (orchestration/agentnick_control.py).
FIXED_TOOLS = {
    "list_governance": ["agent"],
    "get_policy": ["query", "slug", "type", "agent"],
    "get_prompt": ["query"],
    "get_corpus_facts": ["query"],
    "list_calculations": ["family", "name"],
    "run_calculation": ["name", "inputs", "rows"],
    "list_skills": [],
    "use_skill": ["name", "part"],
    "check_policy": ["action_type", "annual_value", "approver", "currency", "fx_rate_to_policy_currency",
                     "history_checked", "orders", "requester", "term_months", "total_commitment", "waiver_id"],
    "search_contracts": ["query", "supplier", "status", "expires_within_days", "limit"],
    "extract_clauses": ["document_id", "clause_type", "containing"],
    "redline_compare": ["original", "revised"],
    "savings_tracker": ["since", "until"],
    "supplier_discovery": ["item", "country", "preferred_only", "limit"],
    "supplier_lookup": ["supplier"],
    "build_spend_cube": ["by", "start", "end", "currency", "top_n", "order"],
    "obligation_tracker": ["contract", "supplier", "horizon_days"],
    "calc_supplier_scorecard": ["supplier", "start", "end", "currency"],
    "screening_summary": ["supplier"],
    "screen_sanctions": ["supplier"],
    "financial_health_score": ["supplier"],
    "esg_rating_lookup": ["supplier"],
    "forced_labour_screen": ["supplier"],
    "conflict_minerals_check": ["supplier"],
    "monitor_risk_events": ["supplier"],
    "commodity_index_lookup": ["series", "start", "end"],
    "classify_spend": ["items", "query"],
    "similar_items": ["description", "min_similarity", "limit"],
    "contract_pipeline": ["expires_within_days", "supplier", "status", "detail"],
    "invoice_reconciliation": ["supplier", "invoice_id", "po_id", "start", "end"],
    "supplier_items": ["supplier", "start", "end"],
    "open_orders": ["supplier", "start", "end"],
    "invoice_status": ["supplier", "invoice_id", "start", "end"],
    "duplicate_invoices": ["supplier", "window_days", "start", "end"],
    "supplier_parents": ["supplier"],
    "render_kpi_card": ["title", "value", "unit", "movement_pct", "target", "status", "source"],
    "build_docx": ["title", "sections", "notes"],
    "build_xlsx": ["title", "sheets"],
    "build_pptx": ["title", "subtitle", "slides"],
    "render_chart": ["title", "chart_type", "data", "unit"],
    "compare_total_cost": ["options", "horizon_years", "cost_of_capital", "buyer_currency", "fx_rates", "fx_table_date", "default_insurance_rate"],
}

# Tools the two older tool loops offer (services/governance_tools/governed_reasoning.py and
# services/supplier_enrichment/research.py). Both now ask the policy gate before each of these runs.
LEGACY_LOOP_TOOLS = {
    "web_search": ["query"],
    "fetch_url": ["url"],
    "describe_platform": ["topic"],
}

CHECKPOINTS = [
    ("tool.call.before", "before a tool runs", "live"),
    ("message.send.before", "before a message is sent", "planned"),
    ("data.egress.before", "before data leaves the system", "planned"),
    ("record.write.before", "before a record is written", "planned"),
]
COMMON_INPUTS = [
    ("tool.name", "the tool being used", "string"),
    ("agent.name", "the agent acting", "string"),
    ("agent.reason", "the agent's stated reason", "string"),
]


def _agent_tools() -> Dict[str, List[str]]:
    from agents.auto_registry import AutoRegistry

    reg = AutoRegistry.from_json()
    out: Dict[str, List[str]] = {}
    for schema in reg.tool_schemas():
        fn = schema["function"]
        out[fn["name"]] = sorted(fn.get("parameters", {}).get("properties", {}).keys())
    return out


def registry_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, plain, status in CHECKPOINTS:
        rows.append({"kind": "checkpoint", "name": name, "checkpoint": None, "plain": plain,
                     "value_type": None, "source": None, "status": status})
    tools = {**_agent_tools(), **FIXED_TOOLS, **LEGACY_LOOP_TOOLS}
    cp = "tool.call.before"
    for field_name, plain, vtype in COMMON_INPUTS:
        rows.append({"kind": "input", "name": field_name, "checkpoint": cp, "plain": plain,
                     "value_type": vtype, "source": "action", "status": "live"})
    seen_args = set()
    for tool, params in sorted(tools.items()):
        rows.append({"kind": "action", "name": tool, "checkpoint": cp, "plain": tool.replace("_", " "),
                     "value_type": None, "source": None, "status": "live"})
        for p in params:
            if p in seen_args:
                continue
            seen_args.add(p)
            rows.append({"kind": "input", "name": f"args.{p}", "checkpoint": cp,
                         "plain": p.replace("_", " "), "value_type": "string",
                         "source": "action", "status": "live"})
    return rows


def main(argv: List[str]) -> int:
    rows = registry_rows()
    if "--apply" not in argv:
        print(json.dumps(rows, indent=1))
        return 0
    from services.db import get_conn

    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        for r in rows:
            cur.execute(
                "INSERT INTO proc.bp_orchestrator_registry (kind, name, checkpoint, plain, value_type, source, status, seeded_from)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,'seed_registry')"
                " ON CONFLICT (kind, name, COALESCE(checkpoint, '')) DO NOTHING",
                (r["kind"], r["name"], r["checkpoint"], r["plain"], r["value_type"], r["source"], r["status"]))
        conn.commit()
    print(f"seeded {len(rows)} rows")
    return 0


if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    sys.exit(main(sys.argv[1:]))
