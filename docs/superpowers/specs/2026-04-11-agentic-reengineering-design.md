# AgentNick Agentic Layer Re-Engineering — Design Spec

**Date:** 2026-04-11
**Status:** Approved
**Scope:** Reasoning layer, execution layer, inter-agent communication, knowledge layer, negotiation engine, process awareness

---

## 1. Overview

Re-engineer AgentNick from a monolithic orchestrator into an intelligent procurement agent with a Reason→Plan→Act→Observe loop. The system replaces hardcoded workflow routing with LLM-driven dynamic orchestration while preserving all 13 existing agents and their functionality.

### Design Principles
- Pattern-based intelligence (no raw data memorization)
- Hybrid autonomy: autonomous for routine, escalates for high-value/uncertain
- Existing agents are integrated, not removed
- Lazy agent instantiation from declarative config
- Shared workflow context for inter-agent awareness

---

## 2. Reasoning Layer

### 2.1 Reasoning Loop

Every task entering the system passes through a four-phase loop:

1. **Reason** — LLM analyzes the task against procurement context: document type, supplier history patterns, category intelligence, active policies
2. **Plan** — LLM composes a dynamic workflow: which agents to invoke, in what order, with what inputs, which can run in parallel
3. **Act** — Plan dispatched to execution layer via existing DAG scheduler + Redis Streams
4. **Observe** — LLM evaluates results, decides: complete, adapt plan, retry, or escalate

### 2.2 Orchestration Prompt

A dedicated orchestration system prompt (separate from extraction prompt) containing:
- Full procurement lifecycle understanding
- Available agent capabilities (read from registry at runtime)
- Escalation policies and thresholds
- Relevant patterns retrieved from knowledge layer

### 2.3 Escalation Triggers
- Extraction confidence below 0.70
- Document/PO value above configurable threshold (default £50,000)
- New supplier with no extraction history
- Negotiation deadlock (3 rounds, no movement)
- Anomaly: invoice total > PO amount by more than 10%

### 2.4 Implementation

**New file:** `src/orchestration/reasoning_engine.py`

```python
class ReasoningEngine:
    """AgentNick's brain — Reason→Plan→Act→Observe loop."""

    def process_task(self, task: Task) -> WorkflowResult:
        context = self._build_procurement_context(task)
        plan = self._reason_and_plan(task, context)
        result = self._execute_plan(plan)
        return self._observe_and_finalize(task, plan, result)

    def _reason_and_plan(self, task, context) -> WorkflowPlan:
        """LLM composes a dynamic workflow from available agents."""

    def _execute_plan(self, plan: WorkflowPlan) -> dict:
        """Dispatches plan to DAG scheduler."""

    def _observe_and_finalize(self, task, plan, result) -> WorkflowResult:
        """Evaluates results, adapts or completes."""
```

---

## 3. Execution Layer

### 3.1 Declarative Agent Registry

**Agent discovery from config, not code.**

`agent_definitions.json` becomes the single source of truth:

```json
{
  "agents": [
    {
      "id": "data_extraction",
      "class": "agents.data_extraction_agent.DataExtractionAgent",
      "capabilities": ["extract_document", "ocr", "nlp"],
      "required_inputs": ["file_path", "doc_type"],
      "output_fields": ["header", "line_items"],
      "description": "Extracts structured data from procurement documents"
    }
  ]
}
```

**New file:** `src/agents/auto_registry.py`

- Reads `agent_definitions.json` at startup
- Dynamically imports agent classes via module paths
- Lazy instantiation: agents created on first use, cached after
- Adding a new agent: drop the class file + add entry to JSON. No code changes elsewhere.

### 3.2 Dynamic Workflow Composition

The reasoning engine produces a structured workflow plan:

```json
{
  "goal": "Process invoice INV304056 and validate against PO",
  "steps": [
    {"agent": "data_extraction", "input": {"file_path": "...", "doc_type": "Invoice"}, "parallel_group": 1},
    {"agent": "discrepancy_detection", "input": {"from_step": 0}, "parallel_group": 2},
    {"agent": "supplier_ranking", "input": {"from_step": 0}, "parallel_group": 2, "condition": "new_supplier"}
  ],
  "escalation": {"threshold_value": 50000, "confidence_min": 0.80}
}
```

Steps in the same `parallel_group` execute concurrently via DAG scheduler.

### 3.3 Fault Handling

When an agent fails, the observe step decides:
- Retry (transient failure)
- Skip (agent marked `required: false`)
- Substitute with LLM fallback
- Escalate to human

---

## 4. Inter-Agent Communication

### 4.1 Shared Workflow Context (Blackboard)

**New file:** `src/orchestration/workflow_context.py`

Every workflow has a `WorkflowContext` that all participating agents read and write:

```python
class WorkflowContext:
    workflow_id: str
    goal: str
    escalation_policy: dict
    agent_results: OrderedDict[str, AgentOutput]  # full chain of prior outputs
    shared_data: dict  # accumulated cross-agent data
    signals: list[AgentSignal]  # inter-agent signals
    procurement_brief: dict  # lifecycle position, related docs, patterns
```

### 4.2 Agent Signals

Agents emit signals during execution:

| Signal | Meaning | Action |
|--------|---------|--------|
| `NEEDS_ATTENTION` | Unusual finding | Downstream agents factor it in |
| `CONFIDENCE_LOW` | Uncertain output | Reasoning loop may add verification step |
| `RECOMMEND_ESCALATION` | Policy boundary crossed | Human notified |
| `SUGGEST_AGENT` | "Also run agent X" | Reasoning loop extends workflow dynamically |

### 4.3 Flow Awareness

Every agent receives:
- The workflow goal and why it was included
- Prior agent outputs in the chain
- Active signals from parallel agents
- Procurement context brief (lifecycle position, related documents, relevant patterns)

---

## 5. Knowledge Layer

### 5.1 Pattern Store

**New table:** `proc.procurement_patterns`

| Column | Type | Purpose |
|--------|------|---------|
| id | serial | Primary key |
| pattern_type | varchar | extraction, negotiation, category, supplier, process |
| pattern_text | text | The learned insight |
| category | varchar | Product/service category this applies to |
| confidence | numeric | 0.0–1.0, based on observation count |
| source_count | integer | Number of workflows that contributed |
| last_validated | timestamp | When last confirmed |
| deprecated | boolean | Auto-deprecated when contradicted |

### 5.2 Pattern Learning

After each completed workflow, the observe step:
1. Compares expectations against outcomes
2. Reinforces patterns that held true (increment confidence)
3. Creates new patterns from novel observations
4. Deprecates patterns contradicted by evidence

### 5.3 Pattern Retrieval

At reasoning time, relevant patterns are queried by:
- Task type (extraction, negotiation, ranking)
- Category (office furniture, IT equipment, professional services)
- Supplier (if known)
- Confidence threshold (default > 0.60)

Retrieved patterns are injected into the reasoning prompt as context, not into the Modelfile.

---

## 6. Negotiation Strategy Engine

### 6.1 Policy Rails (Layer 1)

Hard rules from PolicyEngine:
- Max rounds per negotiation (default 3)
- Auto-approve threshold (default £5,000 with known supplier)
- Escalation threshold (default £50,000)
- Minimum cooling period between rounds (default 4 hours)
- Category-specific limits

### 6.2 Strategy Playbook (Layer 2)

**New file:** `src/engines/negotiation_strategy_engine.py`

Six strategies with contextual selection:

| Strategy | Trigger Context |
|----------|----------------|
| Cooperative | Long-term supplier, repeat orders |
| Competitive | Multiple alternatives, commoditized product |
| Anchoring | First-time supplier, no benchmark |
| Bundling | Multiple line items/categories in play |
| Time Leverage | Urgent need or approaching deadline |
| Collaborative Problem-Solving | Deadlocked after 2 rounds |

### 6.3 Strategy Selection

```python
class NegotiationStrategyEngine:
    def select_strategy(self, context: NegotiationContext) -> Strategy:
        """Analyze context and select primary + fallback strategy."""

    def adapt_strategy(self, round_result: dict, current: Strategy) -> Strategy:
        """After each round, evaluate and potentially switch."""

    def generate_position(self, strategy: Strategy, context: NegotiationContext) -> NegotiationPosition:
        """Produce: target price, arguments, tone, BATNA analysis."""
```

### 6.4 Round-by-Round Adaptation

After each supplier response:
1. Evaluate movement (did they concede? how much?)
2. Compare against expected outcome range
3. Decide: continue same strategy, switch to fallback, or escalate
4. Draft next message with adapted arguments

---

## 7. Process Awareness

### 7.1 Procurement Lifecycle State Machine

AgentNick tracks where every document sits:

```
Need Identified → RFQ Generated → Quotes Received → Quotes Evaluated
→ Supplier Selected → Negotiation → PO Issued → Invoice Received
→ Invoice Matched → Payment Approved → Supplier Reviewed
```

### 7.2 Proactive Intelligence

AgentNick identifies without being asked:
- **Missing links**: PO issued but no invoice after N days
- **Opportunities**: Multiple quotes in same category → consolidation
- **Anomalies**: Invoice amount deviates from PO by >10%
- **Timing**: Quote approaching expiry

### 7.3 Context Brief Generation

For every task, AgentNick produces a procurement context brief:
- Lifecycle position of the document
- Related documents from KG (linked POs, quotes, invoices)
- Relevant patterns for this supplier/category
- Active policies that apply
- Open issues or pending actions

### 7.4 Content Generation

On-demand procurement content:
- RFQ documents with appropriate terms
- Supplier comparison reports
- Negotiation briefings with strategy and alternatives
- Approval summaries with risk assessment

---

## 8. Integration with Existing Agents

All 13 existing agents are preserved. Changes:

| Agent | Change |
|-------|--------|
| DataExtractionAgent | No change — invoked by reasoning engine instead of hardcoded orchestrator |
| SupplierRankingAgent | Receives richer context via workflow blackboard |
| QuoteEvaluationAgent | Can emit SUGGEST_AGENT signal for comparison |
| QuoteComparisonAgent | Reads ranking results from shared context |
| OpportunityMinerAgent | Feeds patterns to knowledge layer |
| EmailDraftingAgent | Receives negotiation strategy context for tone/arguments |
| NegotiationAgent | Enhanced with strategy engine, round adaptation |
| SupplierInteractionAgent | No change — gateway role preserved |
| EmailDispatchAgent | No change — dispatch mechanics preserved |
| EmailWatcherAgent | No change — inbound monitoring preserved |
| ApprovalsAgent | Receives richer escalation context from reasoning engine |
| DiscrepancyDetectionAgent | Can emit NEEDS_ATTENTION signals |
| RAGAgent | Queries pattern store in addition to vector store |

---

## 9. New Files

| File | Purpose |
|------|---------|
| `src/orchestration/reasoning_engine.py` | Reason→Plan→Act→Observe loop |
| `src/orchestration/workflow_context.py` | Shared blackboard + signals |
| `src/agents/auto_registry.py` | Declarative agent discovery + lazy instantiation |
| `src/engines/negotiation_strategy_engine.py` | Strategy selection + adaptation |
| `src/services/pattern_service.py` | Pattern CRUD + learning + retrieval |
| `src/services/procurement_context_service.py` | Lifecycle tracking + proactive intelligence |

## 10. Modified Files

| File | Change |
|------|--------|
| `src/api/main.py` | Replace manual registration with auto_registry |
| `src/agents/base_agent.py` | Add signal emission to BaseAgent, accept WorkflowContext |
| `agent_definitions.json` | Become authoritative source for agent discovery |
| `src/orchestration/orchestrator.py` | Delegate to reasoning_engine, keep as compatibility shim |
| `Modelfile` | Add orchestration-specific prompt sections |

---

## 11. Migration Strategy

The reasoning engine wraps the existing orchestrator. Old code paths continue to work:

1. ReasoningEngine becomes the new entry point
2. For tasks it can handle, it uses the new Reason→Plan→Act→Observe loop
3. For edge cases, it falls back to the existing orchestrator
4. Gradual migration: each workflow type moves to reasoning engine one by one
5. Existing orchestrator remains as fallback until all workflows are migrated
