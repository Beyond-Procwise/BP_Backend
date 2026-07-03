"""Diagnosis: prove AgentNick can DYNAMICALLY compose an agent plan from a
plain-English instruction via the (currently un-wired) ReasoningEngine.

This does NOT touch the API or any static workflow graph. It exercises the
exact code path that would power instruction-driven, next-best-agent routing:
    free-text goal -> ReasoningEngine.reason_and_plan(use_llm_planning=True)
                   -> _llm_compose_plan -> AgentNick LLM picks agents
"""
import sys, json, textwrap
sys.path.insert(0, "src")

from agents.auto_registry import AutoRegistry
from orchestration.reasoning_engine import ReasoningEngine

registry = AutoRegistry.from_json()
print(f"Loaded {len(registry.agent_ids)} agents from agent_definitions.json:")
print("  " + ", ".join(sorted(registry.agent_ids)))
print()

# What AgentNick is shown (first part of the catalogue it reasons over)
catalogue = registry.describe_for_llm()
print("=== Catalogue handed to AgentNick (truncated) ===")
print("\n".join(catalogue.splitlines()[:14]))
print("...\n")

# ReasoningEngine only needs the registry for LLM planning.
engine = ReasoningEngine(agent_nick=None, registry=registry)

# Three PLAIN-ENGLISH instructions with NO task_type -> forces the LLM planner.
instructions = [
    "A new supplier sent us a quote for $80,000 of steel pipe. "
    "Figure out if they're any good, compare the price, and if it's worth it, negotiate.",
    "We just received an invoice PDF. Pull out the data and check it for any discrepancies.",
    "Find cost-saving opportunities across our recent purchase orders and rank the best suppliers.",
]

for i, goal in enumerate(instructions, 1):
    print(f"========== INSTRUCTION {i} ==========")
    print(textwrap.fill(goal, 78))
    task = {"goal": goal, "use_llm_planning": True}
    plan = engine.reason_and_plan(task, context={"patterns": []})
    tag = "DYNAMIC (AgentNick)" if plan.planner == "llm" else f"DEGRADED ({plan.planner})"
    print(f"\nplanner = {plan.planner}  -> {tag}")
    if plan.planning_error:
        print(f"planning_error = {plan.planning_error}")
    print(f"AgentNick composed plan -> goal: {plan.goal!r}")
    if not plan.steps:
        print("  (no steps returned)")
    for s in plan.steps:
        valid = "OK" if s.agent in registry.agent_ids else "UNKNOWN-AGENT"
        print(f"  - group {s.parallel_group}: {s.agent:24s} required={s.required}  [{valid}]")
    print()
