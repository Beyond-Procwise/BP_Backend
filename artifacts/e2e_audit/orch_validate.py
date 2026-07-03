"""In-memory validation of the agentic orchestration layer.
No agent execution (so no action-log writes). Validates:
  - declarative WORKFLOW_REGISTRY graphs (structure, acyclicity)
  - WorkflowContext blackboard contract (record/get/shared/signals)
  - governance resolution (PromptEngine/PolicyEngine read bp_prompt/bp_policy)
"""
from __future__ import annotations
import sys, os, traceback
sys.path.insert(0, "/home/muthu/PycharmProjects/BP_Backend/src")
sys.path.insert(0, "/home/muthu/PycharmProjects/BP_Backend")
os.chdir("/home/muthu/PycharmProjects/BP_Backend")

def section(t): print(f"\n{'='*64}\n{t}\n{'='*64}")

# 1. Declarative workflow registry
section("1. Declarative WORKFLOW_REGISTRY graphs")
try:
    from src.orchestration.workflow_definitions import WORKFLOW_REGISTRY
    for name, graph in WORKFLOW_REGISTRY.items():
        nodes = list(getattr(graph, "nodes", {}).keys())
        edges = getattr(graph, "edges", [])
        entry = getattr(graph, "entry_node", None)
        # acyclicity via topo if available
        acyclic = "?"
        try:
            order = graph.topological_order()
            acyclic = f"acyclic ({len(order)} nodes)"
        except Exception as e:
            acyclic = f"TOPO-ERR {str(e)[:40]}"
        print(f"  {name:24} entry={entry} nodes={len(nodes)} edges={len(edges)} {acyclic}")
        print(f"       node agents: {[getattr(graph.nodes[n],'agent_type',n) for n in nodes]}")
except Exception:
    traceback.print_exc()

# 2. Blackboard contract
section("2. WorkflowContext blackboard contract")
try:
    from src.orchestration.workflow_context import WorkflowContext, SignalType
    ctx = WorkflowContext(workflow_id="wf-audit-1", goal="audit test")
    ctx.record_result("DataExtractionAgent", {"doc_pk": "INV-1", "fields": 30})
    ctx.record_result("SupplierRankingAgent", {"ranked": ["SUP-A", "SUP-B"]})
    ctx.update_shared("supplier_ids", ["SUP-A", "SUP-B"])
    ctx.emit_signal("SupplierRankingAgent", SignalType.CONFIDENCE_LOW, "tie between top 2", {"gap": 0.01})
    prior = ctx.get_prior_result("DataExtractionAgent")
    sig = ctx.get_signals()
    print(f"  record/get_prior_result: {prior}")
    print(f"  shared_data['supplier_ids']: {ctx.shared_data.get('supplier_ids')}")
    print(f"  agent_results order: {list(ctx.agent_results.keys())}")
    print(f"  signals emitted: {[(s.signal_type.name if hasattr(s.signal_type,'name') else s.signal_type, s.message) for s in sig]}")
    print("  BLACKBOARD OK")
except Exception:
    traceback.print_exc()

# 3. Governance resolution (read-only DB)
section("3. Governance: PromptEngine / PolicyEngine read bp_prompt/bp_policy")
try:
    from src.orchestration.prompt_engine import PromptEngine
    pe = PromptEngine()
    if hasattr(pe, "load"):
        try: pe.load()
        except Exception: pass
    allp = pe.all_prompts() if hasattr(pe, "all_prompts") else {}
    print(f"  PromptEngine.all_prompts(): {len(allp)} prompts -> {list(allp)[:8]}")
except Exception:
    traceback.print_exc()
try:
    from src.engines.policy_engine import PolicyEngine
    pol = PolicyEngine()
    if hasattr(pol, "load"):
        try: pol.load()
        except Exception: pass
    lp = pol.list_policies() if hasattr(pol, "list_policies") else []
    names = [ (p.get('policy_name') if isinstance(p,dict) else getattr(p,'name',str(p))) for p in lp ]
    print(f"  PolicyEngine.list_policies(): {len(lp)} policies -> {names[:10]}")
except Exception:
    traceback.print_exc()

# 4. Agent registry from agent_definitions.json
section("4. Agent registry (agent_definitions.json)")
try:
    import json
    p = "agent_definitions.json"
    if os.path.exists(p):
        defs = json.load(open(p))
        agents = defs if isinstance(defs, list) else defs.get("agents", defs)
        if isinstance(agents, dict): agents = list(agents.keys())
        print(f"  {len(agents)} agent defs")
        for a in (agents[:20] if isinstance(agents, list) else []):
            print("   -", a if isinstance(a, str) else (a.get('name') or a.get('agent_type') or a.get('slug')))
    else:
        print("  agent_definitions.json not found at root")
except Exception:
    traceback.print_exc()

print("\n[orch_validate] DONE")
