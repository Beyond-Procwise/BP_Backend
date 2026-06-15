# AgentNick Agentic Re-Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-engineer AgentNick into an intelligent procurement agent with LLM-driven orchestration, auto agent spawning, inter-agent communication, pattern-based learning, negotiation strategy, and end-to-end process awareness.

**Architecture:** Replace monolithic orchestrator with a Reason→Plan→Act→Observe loop. Agents auto-discovered from `agent_definitions.json`, dynamically composed into workflows by the reasoning engine. Shared blackboard enables inter-agent communication. Pattern store provides learning. Negotiation engine applies strategy playbooks within policy rails.

**Tech Stack:** Python 3.12, qwen3:30b (Ollama), PostgreSQL, Redis Streams, Neo4j, pytest

**Spec:** `docs/superpowers/specs/2026-04-11-agentic-reengineering-design.md`

**Constraint:** All 13 existing agents preserved — no removals, only enhancements.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/orchestration/reasoning_engine.py` | Reason→Plan→Act→Observe loop, LLM-driven workflow composition |
| `src/orchestration/workflow_context.py` | Shared blackboard, agent signals, procurement brief |
| `src/agents/auto_registry.py` | Declarative agent discovery from JSON, lazy instantiation |
| `src/engines/negotiation_strategy_engine.py` | Strategy selection, round adaptation, BATNA analysis |
| `src/services/pattern_service.py` | Pattern CRUD, learning from outcomes, retrieval |
| `src/services/procurement_context_service.py` | Lifecycle tracking, proactive intelligence, context briefs |
| `tests/test_reasoning_engine.py` | Tests for reasoning loop |
| `tests/test_workflow_context.py` | Tests for blackboard + signals |
| `tests/test_auto_registry.py` | Tests for declarative agent discovery |
| `tests/test_negotiation_strategy_engine.py` | Tests for strategy selection |
| `tests/test_pattern_service.py` | Tests for pattern learning |
| `tests/test_procurement_context_service.py` | Tests for lifecycle tracking |

### Modified Files
| File | Change |
|------|--------|
| `agent_definitions.json` | Add `class` module paths, I/O contracts, make authoritative |
| `src/agents/base_agent.py` | Add signal emission, accept WorkflowContext |
| `src/api/main.py` | Replace manual registration with auto_registry |
| `src/orchestration/orchestrator.py` | Delegate to reasoning_engine, keep as fallback |
| `Modelfile` | Add orchestration prompt sections |

---

## Task 1: Pattern Service — Knowledge Foundation

**Files:**
- Create: `src/services/pattern_service.py`
- Test: `tests/test_pattern_service.py`

This is the foundation — other components query patterns for intelligence.

- [ ] **Step 1: Write failing tests for pattern CRUD**

```python
# tests/test_pattern_service.py
import pytest
from unittest.mock import MagicMock, patch

class TestPatternService:
    def test_record_pattern_creates_new(self):
        svc = self._make_service()
        svc.record_pattern(
            pattern_type="negotiation",
            pattern_text="Cooperative strategy yields 10% discount with repeat suppliers",
            category="office_furniture",
            confidence=0.75,
        )
        patterns = svc.get_patterns(pattern_type="negotiation", category="office_furniture")
        assert len(patterns) >= 1
        assert "Cooperative" in patterns[0]["pattern_text"]

    def test_get_patterns_filters_by_type_and_category(self):
        svc = self._make_service()
        svc.record_pattern("negotiation", "Pattern A", category="IT")
        svc.record_pattern("extraction", "Pattern B", category="IT")
        svc.record_pattern("negotiation", "Pattern C", category="furniture")
        result = svc.get_patterns(pattern_type="negotiation", category="IT")
        assert len(result) == 1
        assert result[0]["pattern_text"] == "Pattern A"

    def test_reinforce_pattern_increments_confidence(self):
        svc = self._make_service()
        svc.record_pattern("category", "VAT is always 20% for UK", confidence=0.7)
        svc.reinforce_pattern("category", "VAT is always 20% for UK")
        patterns = svc.get_patterns(pattern_type="category")
        assert patterns[0]["source_count"] > 1

    def test_deprecate_pattern(self):
        svc = self._make_service()
        svc.record_pattern("extraction", "Old pattern", confidence=0.5)
        svc.deprecate_pattern("extraction", "Old pattern")
        patterns = svc.get_patterns(pattern_type="extraction")
        assert len(patterns) == 0  # deprecated patterns excluded by default

    def test_learn_from_workflow_outcome(self):
        svc = self._make_service()
        svc.learn_from_outcome(
            workflow_type="negotiation",
            category="IT_equipment",
            outcome={"strategy": "competitive", "discount_achieved": 0.12},
            expected={"discount_target": 0.10},
        )
        patterns = svc.get_patterns(pattern_type="negotiation", category="IT_equipment")
        assert len(patterns) >= 1

    def _make_service(self):
        from services.pattern_service import PatternService
        mock_nick = MagicMock()
        # Use in-memory storage for tests
        return PatternService(mock_nick, storage="memory")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src && python -m pytest ../tests/test_pattern_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement PatternService**

```python
# src/services/pattern_service.py
"""Pattern-based intelligence store.

Stores learned procurement patterns (not raw data) that make AgentNick
progressively smarter. Patterns are retrieved at reasoning time and
injected into LLM prompts as context.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PatternService:
    """CRUD + learning for procurement patterns."""

    def __init__(self, agent_nick, *, storage: str = "postgres") -> None:
        self._agent_nick = agent_nick
        self._storage = storage
        self._memory_store: List[Dict[str, Any]] = []  # for testing

    def record_pattern(
        self,
        pattern_type: str,
        pattern_text: str,
        *,
        category: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Record a new pattern or update existing one."""
        if self._storage == "memory":
            existing = [p for p in self._memory_store
                        if p["pattern_type"] == pattern_type
                        and p["pattern_text"] == pattern_text]
            if existing:
                existing[0]["source_count"] += 1
                existing[0]["confidence"] = min(1.0, existing[0]["confidence"] + 0.05)
            else:
                self._memory_store.append({
                    "pattern_type": pattern_type,
                    "pattern_text": pattern_text,
                    "category": category,
                    "confidence": confidence,
                    "source_count": 1,
                    "deprecated": False,
                    "last_validated": datetime.now(timezone.utc),
                })
            return

        try:
            conn = self._agent_nick.get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO proc.procurement_patterns
                        (pattern_type, pattern_text, category, confidence, source_count, last_validated)
                    VALUES (%s, %s, %s, %s, 1, NOW())
                    ON CONFLICT (pattern_type, pattern_text)
                    DO UPDATE SET
                        source_count = proc.procurement_patterns.source_count + 1,
                        confidence = LEAST(1.0, proc.procurement_patterns.confidence + 0.05),
                        last_validated = NOW()
                """, (pattern_type, pattern_text, category, confidence))
            conn.close()
        except Exception:
            logger.debug("Failed to record pattern", exc_info=True)

    def get_patterns(
        self,
        *,
        pattern_type: str = "",
        category: str = "",
        min_confidence: float = 0.4,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve patterns matching filters."""
        if self._storage == "memory":
            results = [p for p in self._memory_store if not p["deprecated"]]
            if pattern_type:
                results = [p for p in results if p["pattern_type"] == pattern_type]
            if category:
                results = [p for p in results if p["category"] == category]
            results = [p for p in results if p["confidence"] >= min_confidence]
            return sorted(results, key=lambda p: -p["confidence"])[:limit]

        try:
            conn = self._agent_nick.get_db_connection()
            with conn.cursor() as cur:
                sql = """
                    SELECT pattern_type, pattern_text, category, confidence,
                           source_count, last_validated
                    FROM proc.procurement_patterns
                    WHERE deprecated = FALSE AND confidence >= %s
                """
                params: list = [min_confidence]
                if pattern_type:
                    sql += " AND pattern_type = %s"
                    params.append(pattern_type)
                if category:
                    sql += " AND category = %s"
                    params.append(category)
                sql += " ORDER BY confidence DESC LIMIT %s"
                params.append(limit)
                cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            logger.debug("Failed to get patterns", exc_info=True)
            return []

    def reinforce_pattern(self, pattern_type: str, pattern_text: str) -> None:
        """Increment confidence after pattern is confirmed."""
        self.record_pattern(pattern_type, pattern_text)

    def deprecate_pattern(self, pattern_type: str, pattern_text: str) -> None:
        """Mark a pattern as deprecated (contradicted by evidence)."""
        if self._storage == "memory":
            for p in self._memory_store:
                if p["pattern_type"] == pattern_type and p["pattern_text"] == pattern_text:
                    p["deprecated"] = True
            return
        try:
            conn = self._agent_nick.get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE proc.procurement_patterns SET deprecated = TRUE "
                    "WHERE pattern_type = %s AND pattern_text = %s",
                    (pattern_type, pattern_text),
                )
            conn.close()
        except Exception:
            logger.debug("Failed to deprecate pattern", exc_info=True)

    def learn_from_outcome(
        self,
        workflow_type: str,
        category: str,
        outcome: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> None:
        """Learn patterns by comparing expected vs actual outcomes."""
        if workflow_type == "negotiation":
            strategy = outcome.get("strategy", "")
            discount = outcome.get("discount_achieved", 0)
            target = expected.get("discount_target", 0)
            if discount >= target and strategy:
                self.record_pattern(
                    "negotiation",
                    f"{strategy} strategy achieved {discount:.0%} discount in {category}",
                    category=category,
                    confidence=0.7,
                )
        elif workflow_type == "extraction":
            pass  # extraction pattern learning handled by vendor_profile_service
```

- [ ] **Step 4: Create DB table for patterns**

```sql
-- Run via psycopg2 in PatternService.__init__ or migration
CREATE TABLE IF NOT EXISTS proc.procurement_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    pattern_text TEXT NOT NULL,
    category VARCHAR(100) DEFAULT '',
    confidence NUMERIC(4,3) DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    last_validated TIMESTAMP DEFAULT NOW(),
    deprecated BOOLEAN DEFAULT FALSE,
    UNIQUE (pattern_type, pattern_text)
);
CREATE INDEX IF NOT EXISTS idx_patterns_type_cat
    ON proc.procurement_patterns (pattern_type, category)
    WHERE deprecated = FALSE;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd src && python -m pytest ../tests/test_pattern_service.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/services/pattern_service.py tests/test_pattern_service.py
git commit -m "feat: add PatternService for pattern-based procurement intelligence"
```

---

## Task 2: Workflow Context — Shared Blackboard & Signals

**Files:**
- Create: `src/orchestration/workflow_context.py`
- Test: `tests/test_workflow_context.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_workflow_context.py
import pytest

class TestWorkflowContext:
    def test_create_context_with_goal(self):
        from orchestration.workflow_context import WorkflowContext
        ctx = WorkflowContext(workflow_id="wf-001", goal="Extract invoice")
        assert ctx.workflow_id == "wf-001"
        assert ctx.goal == "Extract invoice"

    def test_record_agent_result(self):
        from orchestration.workflow_context import WorkflowContext
        ctx = WorkflowContext(workflow_id="wf-001", goal="test")
        ctx.record_result("data_extraction", {"invoice_id": "INV001", "total": 1200})
        assert "data_extraction" in ctx.agent_results
        assert ctx.agent_results["data_extraction"]["invoice_id"] == "INV001"

    def test_shared_data_accumulates(self):
        from orchestration.workflow_context import WorkflowContext
        ctx = WorkflowContext(workflow_id="wf-001", goal="test")
        ctx.update_shared("supplier_name", "SupplyX Ltd")
        ctx.update_shared("category", "office_furniture")
        assert ctx.shared_data["supplier_name"] == "SupplyX Ltd"
        assert ctx.shared_data["category"] == "office_furniture"

    def test_emit_and_read_signals(self):
        from orchestration.workflow_context import WorkflowContext, AgentSignal
        ctx = WorkflowContext(workflow_id="wf-001", goal="test")
        ctx.emit_signal(AgentSignal(
            agent="discrepancy_detection",
            signal_type="NEEDS_ATTENTION",
            message="Invoice total exceeds PO by 15%",
        ))
        signals = ctx.get_signals()
        assert len(signals) == 1
        assert signals[0].signal_type == "NEEDS_ATTENTION"

    def test_get_signals_by_type(self):
        from orchestration.workflow_context import WorkflowContext, AgentSignal
        ctx = WorkflowContext(workflow_id="wf-001", goal="test")
        ctx.emit_signal(AgentSignal("agent_a", "NEEDS_ATTENTION", "issue 1"))
        ctx.emit_signal(AgentSignal("agent_b", "CONFIDENCE_LOW", "uncertain"))
        ctx.emit_signal(AgentSignal("agent_c", "SUGGEST_AGENT", "run ranking"))
        assert len(ctx.get_signals(signal_type="NEEDS_ATTENTION")) == 1
        assert len(ctx.get_signals(signal_type="SUGGEST_AGENT")) == 1

    def test_procurement_brief(self):
        from orchestration.workflow_context import WorkflowContext
        ctx = WorkflowContext(workflow_id="wf-001", goal="test")
        ctx.set_procurement_brief({
            "lifecycle_stage": "Invoice Matching",
            "related_documents": ["PO526809", "QUT30746"],
            "patterns": ["UK invoices typically have 20% VAT"],
        })
        assert ctx.procurement_brief["lifecycle_stage"] == "Invoice Matching"

    def test_context_serializes_to_dict(self):
        from orchestration.workflow_context import WorkflowContext
        ctx = WorkflowContext(workflow_id="wf-001", goal="test")
        ctx.update_shared("key", "value")
        d = ctx.to_dict()
        assert d["workflow_id"] == "wf-001"
        assert d["shared_data"]["key"] == "value"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src && python -m pytest ../tests/test_workflow_context.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement WorkflowContext**

```python
# src/orchestration/workflow_context.py
"""Shared workflow context (blackboard) for inter-agent communication.

Every workflow has a WorkflowContext that all participating agents can
read and write to. This enables flow awareness — agents know what
happened before them, what's happening alongside them, and what the
overall goal is.
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AgentSignal:
    """Signal emitted by an agent during execution."""
    agent: str
    signal_type: str  # NEEDS_ATTENTION, CONFIDENCE_LOW, RECOMMEND_ESCALATION, SUGGEST_AGENT
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowContext:
    """Shared blackboard for a workflow execution."""

    def __init__(
        self,
        workflow_id: str = "",
        goal: str = "",
        escalation_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.goal = goal
        self.escalation_policy = escalation_policy or {}
        self.agent_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.shared_data: Dict[str, Any] = {}
        self._signals: List[AgentSignal] = []
        self.procurement_brief: Dict[str, Any] = {}
        self.created_at = datetime.now(timezone.utc)

    def record_result(self, agent_name: str, result: Dict[str, Any]) -> None:
        """Record an agent's output into the shared context."""
        self.agent_results[agent_name] = result

    def update_shared(self, key: str, value: Any) -> None:
        """Update shared data visible to all agents."""
        self.shared_data[key] = value

    def emit_signal(self, signal: AgentSignal) -> None:
        """Emit a signal for other agents or the reasoning loop."""
        self._signals.append(signal)

    def get_signals(
        self, *, signal_type: str = "", agent: str = ""
    ) -> List[AgentSignal]:
        """Get signals, optionally filtered."""
        result = self._signals
        if signal_type:
            result = [s for s in result if s.signal_type == signal_type]
        if agent:
            result = [s for s in result if s.agent == agent]
        return result

    def has_signal(self, signal_type: str) -> bool:
        return any(s.signal_type == signal_type for s in self._signals)

    def set_procurement_brief(self, brief: Dict[str, Any]) -> None:
        """Set the procurement context brief for this workflow."""
        self.procurement_brief = brief

    def get_prior_result(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent's prior result."""
        return self.agent_results.get(agent_name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for passing to agents or logging."""
        return {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "escalation_policy": self.escalation_policy,
            "agent_results": dict(self.agent_results),
            "shared_data": self.shared_data,
            "signals": [
                {"agent": s.agent, "type": s.signal_type, "message": s.message}
                for s in self._signals
            ],
            "procurement_brief": self.procurement_brief,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd src && python -m pytest ../tests/test_workflow_context.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/workflow_context.py tests/test_workflow_context.py
git commit -m "feat: add WorkflowContext shared blackboard with agent signals"
```

---

## Task 3: Auto Registry — Declarative Agent Discovery

**Files:**
- Create: `src/agents/auto_registry.py`
- Modify: `agent_definitions.json`
- Test: `tests/test_auto_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_auto_registry.py
import pytest
import json
import tempfile
import os

class TestAutoRegistry:
    def test_load_definitions_from_json(self):
        from agents.auto_registry import AutoRegistry
        registry = AutoRegistry.from_json("agent_definitions.json")
        assert len(registry.agent_ids) >= 13

    def test_get_agent_contract(self):
        from agents.auto_registry import AutoRegistry
        registry = AutoRegistry.from_json("agent_definitions.json")
        contract = registry.get_contract("data_extraction")
        assert "extract" in " ".join(contract.capabilities).lower()
        assert "file_path" in contract.required_inputs

    def test_lazy_instantiation(self):
        from agents.auto_registry import AutoRegistry
        registry = AutoRegistry.from_json("agent_definitions.json")
        # Before first access, agent is not instantiated
        assert not registry.is_instantiated("rag")
        # Access triggers instantiation (requires agent_nick — skip in unit test)

    def test_find_agents_by_capability(self):
        from agents.auto_registry import AutoRegistry
        registry = AutoRegistry.from_json("agent_definitions.json")
        extractors = registry.find_by_capability("extract_document")
        assert "data_extraction" in extractors

    def test_list_all_capabilities(self):
        from agents.auto_registry import AutoRegistry
        registry = AutoRegistry.from_json("agent_definitions.json")
        caps = registry.all_capabilities()
        assert "extract_document" in caps
        assert "rank_suppliers" in caps

    def test_agent_description_for_llm(self):
        from agents.auto_registry import AutoRegistry
        registry = AutoRegistry.from_json("agent_definitions.json")
        desc = registry.describe_for_llm()
        assert "data_extraction" in desc
        assert "supplier_ranking" in desc
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src && python -m pytest ../tests/test_auto_registry.py -v`

- [ ] **Step 3: Update agent_definitions.json with module paths**

Add `class_path` and `contract` fields to each agent definition in `agent_definitions.json`. Keep all existing fields. Add:

```json
{
  "agents": [
    {
      "id": "data_extraction",
      "class_path": "agents.data_extraction_agent.DataExtractionAgent",
      "capabilities": ["extract_document", "ocr", "nlp", "pdf_extraction"],
      "required_inputs": ["file_path", "doc_type"],
      "output_fields": ["header", "line_items", "confidence"],
      "description": "Extracts structured data from procurement documents (PDF, DOCX, Excel)"
    },
    ...all 13 agents with proper class_path and contracts...
  ]
}
```

- [ ] **Step 4: Implement AutoRegistry**

```python
# src/agents/auto_registry.py
"""Declarative agent discovery and lazy instantiation.

Reads agent_definitions.json as the single source of truth.
Agents are instantiated on first use and cached.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFINITIONS_PATH = Path(__file__).resolve().parents[2] / "agent_definitions.json"


@dataclass
class AgentContract:
    """Declarative I/O specification for an agent."""
    id: str
    class_path: str
    capabilities: List[str]
    required_inputs: List[str]
    output_fields: List[str]
    description: str


class AutoRegistry:
    """Discover agents from JSON, instantiate lazily."""

    def __init__(self, definitions: List[Dict[str, Any]]) -> None:
        self._definitions = {d["id"]: d for d in definitions}
        self._contracts: Dict[str, AgentContract] = {}
        self._instances: Dict[str, Any] = {}
        self._agent_nick = None

        for d in definitions:
            self._contracts[d["id"]] = AgentContract(
                id=d["id"],
                class_path=d.get("class_path", ""),
                capabilities=d.get("capabilities", []),
                required_inputs=d.get("required_inputs", []),
                output_fields=d.get("output_fields", []),
                description=d.get("description", ""),
            )

    @classmethod
    def from_json(cls, path: str = "") -> "AutoRegistry":
        fpath = Path(path) if path else DEFINITIONS_PATH
        if not fpath.is_absolute():
            fpath = Path(__file__).resolve().parents[2] / fpath
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        agents = data if isinstance(data, list) else data.get("agents", [])
        return cls(agents)

    def set_agent_nick(self, agent_nick) -> None:
        self._agent_nick = agent_nick

    @property
    def agent_ids(self) -> List[str]:
        return list(self._definitions.keys())

    def get_contract(self, agent_id: str) -> AgentContract:
        return self._contracts[agent_id]

    def get_agent(self, agent_id: str):
        """Get or lazily instantiate an agent."""
        if agent_id in self._instances:
            return self._instances[agent_id]
        contract = self._contracts.get(agent_id)
        if not contract or not contract.class_path:
            raise KeyError(f"Unknown agent: {agent_id}")
        agent = self._instantiate(contract)
        self._instances[agent_id] = agent
        return agent

    def is_instantiated(self, agent_id: str) -> bool:
        return agent_id in self._instances

    def find_by_capability(self, capability: str) -> List[str]:
        return [
            aid for aid, c in self._contracts.items()
            if capability in c.capabilities
        ]

    def all_capabilities(self) -> Set[str]:
        caps = set()
        for c in self._contracts.values():
            caps.update(c.capabilities)
        return caps

    def describe_for_llm(self) -> str:
        """Generate a description of all agents for the reasoning LLM."""
        lines = []
        for aid, c in self._contracts.items():
            caps = ", ".join(c.capabilities)
            inputs = ", ".join(c.required_inputs)
            outputs = ", ".join(c.output_fields)
            lines.append(
                f"- {aid}: {c.description}\n"
                f"  Capabilities: [{caps}]\n"
                f"  Inputs: [{inputs}] → Outputs: [{outputs}]"
            )
        return "\n".join(lines)

    def _instantiate(self, contract: AgentContract):
        module_path, class_name = contract.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(self._agent_nick)
```

- [ ] **Step 5: Run tests**

Run: `cd src && python -m pytest ../tests/test_auto_registry.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/agents/auto_registry.py agent_definitions.json tests/test_auto_registry.py
git commit -m "feat: add AutoRegistry for declarative agent discovery from JSON"
```

---

## Task 4: Negotiation Strategy Engine

**Files:**
- Create: `src/engines/negotiation_strategy_engine.py`
- Test: `tests/test_negotiation_strategy_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_negotiation_strategy_engine.py
import pytest

class TestNegotiationStrategyEngine:
    def test_select_cooperative_for_repeat_supplier(self):
        from engines.negotiation_strategy_engine import NegotiationStrategyEngine, NegotiationContext
        engine = NegotiationStrategyEngine()
        ctx = NegotiationContext(
            supplier_name="SupplyX Ltd",
            supplier_history_count=10,
            alternative_quotes=0,
            order_value=5000,
            category="office_furniture",
        )
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "cooperative"

    def test_select_competitive_with_alternatives(self):
        from engines.negotiation_strategy_engine import NegotiationStrategyEngine, NegotiationContext
        engine = NegotiationStrategyEngine()
        ctx = NegotiationContext(
            supplier_name="NewSupplier Ltd",
            supplier_history_count=0,
            alternative_quotes=3,
            order_value=20000,
            category="IT_equipment",
        )
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "competitive"

    def test_select_anchoring_for_new_supplier(self):
        from engines.negotiation_strategy_engine import NegotiationStrategyEngine, NegotiationContext
        engine = NegotiationStrategyEngine()
        ctx = NegotiationContext(
            supplier_name="Unknown Corp",
            supplier_history_count=0,
            alternative_quotes=0,
            order_value=15000,
            category="stationery",
        )
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "anchoring"

    def test_adapt_to_collaborative_after_deadlock(self):
        from engines.negotiation_strategy_engine import NegotiationStrategyEngine, NegotiationContext, Strategy
        engine = NegotiationStrategyEngine()
        current = Strategy(name="competitive", target_discount=0.15)
        round_result = {"supplier_movement": 0, "round_number": 3}
        adapted = engine.adapt_strategy(current, round_result)
        assert adapted.name == "collaborative_problem_solving"

    def test_generate_position(self):
        from engines.negotiation_strategy_engine import NegotiationStrategyEngine, NegotiationContext, Strategy
        engine = NegotiationStrategyEngine()
        strategy = Strategy(name="competitive", target_discount=0.15)
        ctx = NegotiationContext(
            supplier_name="Test Ltd", order_value=10000,
            alternative_quotes=2, category="office_furniture",
        )
        position = engine.generate_position(strategy, ctx)
        assert position.target_price > 0
        assert position.arguments
        assert position.tone in ("firm", "professional", "collaborative", "urgent")

    def test_policy_rails_enforce_max_rounds(self):
        from engines.negotiation_strategy_engine import NegotiationStrategyEngine, NegotiationContext
        engine = NegotiationStrategyEngine(max_rounds=3)
        ctx = NegotiationContext(
            supplier_name="Test", order_value=5000, round_number=4,
        )
        result = engine.should_continue(ctx)
        assert result.action == "escalate"
        assert "max rounds" in result.reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src && python -m pytest ../tests/test_negotiation_strategy_engine.py -v`

- [ ] **Step 3: Implement NegotiationStrategyEngine**

Create `src/engines/negotiation_strategy_engine.py` with:
- `NegotiationContext` dataclass with: supplier_name, supplier_history_count, alternative_quotes, order_value, category, round_number, urgency
- `Strategy` dataclass with: name, target_discount, fallback_name, arguments_template
- `NegotiationPosition` dataclass with: target_price, arguments (list of strings), tone, batna_analysis
- `ContinueDecision` dataclass with: action (continue/escalate/accept/walk_away), reason
- `NegotiationStrategyEngine` class with: select_strategy(), adapt_strategy(), generate_position(), should_continue()
- Six strategy definitions: cooperative, competitive, anchoring, bundling, time_leverage, collaborative_problem_solving
- Selection logic based on: supplier_history_count (>5 → cooperative), alternative_quotes (>1 → competitive), new supplier + no alternatives → anchoring
- Adaptation: no movement after 2 rounds → switch to collaborative_problem_solving
- Policy rails: max_rounds, auto_approve_threshold, escalation_threshold

- [ ] **Step 4: Run tests**

Run: `cd src && python -m pytest ../tests/test_negotiation_strategy_engine.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/engines/negotiation_strategy_engine.py tests/test_negotiation_strategy_engine.py
git commit -m "feat: add NegotiationStrategyEngine with 6 strategies and policy rails"
```

---

## Task 5: Procurement Context Service

**Files:**
- Create: `src/services/procurement_context_service.py`
- Test: `tests/test_procurement_context_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_procurement_context_service.py
import pytest
from unittest.mock import MagicMock

class TestProcurementContextService:
    def test_determine_lifecycle_stage_invoice(self):
        from services.procurement_context_service import ProcurementContextService
        svc = ProcurementContextService(MagicMock())
        stage = svc.determine_lifecycle_stage("Invoice", {"po_id": "526809"})
        assert stage == "Invoice Matching"

    def test_determine_lifecycle_stage_new_quote(self):
        from services.procurement_context_service import ProcurementContextService
        svc = ProcurementContextService(MagicMock())
        stage = svc.determine_lifecycle_stage("Quote", {})
        assert stage == "Quotes Received"

    def test_build_context_brief(self):
        from services.procurement_context_service import ProcurementContextService
        svc = ProcurementContextService(MagicMock())
        brief = svc.build_context_brief(
            doc_type="Invoice",
            header={"invoice_id": "INV001", "po_id": "PO526809", "supplier_id": "SupplyX"},
            patterns=[{"pattern_text": "UK invoices have 20% VAT"}],
        )
        assert brief["lifecycle_stage"]
        assert brief["patterns"]
        assert "supplier_id" in brief["document_summary"]

    def test_detect_anomalies(self):
        from services.procurement_context_service import ProcurementContextService
        svc = ProcurementContextService(MagicMock())
        anomalies = svc.detect_anomalies(
            header={"invoice_total_incl_tax": 15000, "po_id": "PO001"},
            po_total=10000,
        )
        assert any("exceeds" in a.lower() for a in anomalies)

    def test_no_anomaly_when_within_tolerance(self):
        from services.procurement_context_service import ProcurementContextService
        svc = ProcurementContextService(MagicMock())
        anomalies = svc.detect_anomalies(
            header={"invoice_total_incl_tax": 10500, "po_id": "PO001"},
            po_total=10000,
        )
        assert len(anomalies) == 0
```

- [ ] **Step 2: Run and verify failure, Step 3: Implement, Step 4: Run and pass, Step 5: Commit**

Implement `ProcurementContextService` with:
- `determine_lifecycle_stage(doc_type, header)` — maps document to procurement lifecycle position
- `build_context_brief(doc_type, header, patterns, related_docs)` — generates full procurement context for agents
- `detect_anomalies(header, po_total, quote_total)` — identifies deviations
- `find_related_documents(doc_id, doc_type)` — queries KG for linked documents

```bash
git add src/services/procurement_context_service.py tests/test_procurement_context_service.py
git commit -m "feat: add ProcurementContextService for lifecycle tracking and proactive intelligence"
```

---

## Task 6: Reasoning Engine — The Core

**Files:**
- Create: `src/orchestration/reasoning_engine.py`
- Test: `tests/test_reasoning_engine.py`
- Modify: `src/orchestration/orchestrator.py` (delegate to reasoning engine)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reasoning_engine.py
import pytest
from unittest.mock import MagicMock, patch

class TestReasoningEngine:
    def test_plan_simple_extraction(self):
        from orchestration.reasoning_engine import ReasoningEngine
        engine = self._make_engine()
        plan = engine.reason_and_plan({
            "task_type": "document_extraction",
            "file_path": "documents/invoice/test.pdf",
            "doc_type": "Invoice",
        })
        assert plan.goal
        assert len(plan.steps) >= 1
        assert plan.steps[0].agent == "data_extraction"

    def test_plan_includes_ranking_for_new_supplier(self):
        from orchestration.reasoning_engine import ReasoningEngine
        engine = self._make_engine()
        plan = engine.reason_and_plan({
            "task_type": "document_extraction",
            "file_path": "documents/quote/new_supplier.pdf",
            "doc_type": "Quote",
            "is_new_supplier": True,
        })
        agent_names = [s.agent for s in plan.steps]
        assert "data_extraction" in agent_names
        assert "supplier_ranking" in agent_names

    def test_plan_negotiation_includes_strategy(self):
        from orchestration.reasoning_engine import ReasoningEngine
        engine = self._make_engine()
        plan = engine.reason_and_plan({
            "task_type": "negotiation",
            "supplier_name": "SupplyX Ltd",
            "order_value": 50000,
            "category": "office_furniture",
        })
        assert plan.negotiation_strategy is not None

    def test_observe_detects_low_confidence(self):
        from orchestration.reasoning_engine import ReasoningEngine
        engine = self._make_engine()
        observation = engine.observe({
            "status": "success",
            "confidence": 0.55,
            "data": {"invoice_id": "INV001"},
        })
        assert observation.action in ("retry", "escalate")

    def test_observe_completes_high_confidence(self):
        from orchestration.reasoning_engine import ReasoningEngine
        engine = self._make_engine()
        observation = engine.observe({
            "status": "success",
            "confidence": 0.92,
            "data": {"invoice_id": "INV001"},
        })
        assert observation.action == "complete"

    def _make_engine(self):
        from orchestration.reasoning_engine import ReasoningEngine
        mock_nick = MagicMock()
        mock_registry = MagicMock()
        mock_registry.describe_for_llm.return_value = "- data_extraction: Extracts data\n- supplier_ranking: Ranks suppliers"
        mock_registry.agent_ids = ["data_extraction", "supplier_ranking", "discrepancy_detection"]
        return ReasoningEngine(mock_nick, mock_registry)
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement ReasoningEngine**

Create `src/orchestration/reasoning_engine.py` with:
- `WorkflowPlan` dataclass: goal, steps (list of PlanStep), negotiation_strategy, escalation_policy
- `PlanStep` dataclass: agent, input_mapping, parallel_group, condition, required
- `Observation` dataclass: action (complete/retry/escalate/adapt), reason, suggested_changes
- `ReasoningEngine` class:
  - `__init__(agent_nick, registry, pattern_service, procurement_context_service)`
  - `process_task(task)` — full Reason→Plan→Act→Observe loop
  - `reason_and_plan(task)` — LLM composes workflow plan from available agents
  - `execute_plan(plan, workflow_context)` — dispatches to agents
  - `observe(results)` — evaluates and decides next action
  - `_build_reasoning_prompt(task, context)` — constructs the orchestration prompt with agent descriptions, patterns, policies

The reasoning prompt includes:
- Available agents (from `registry.describe_for_llm()`)
- Relevant patterns (from `pattern_service.get_patterns()`)
- Procurement context (from `procurement_context_service.build_context_brief()`)
- Escalation policies

For the `reason_and_plan` step, use a structured JSON output format that the engine parses into a `WorkflowPlan`.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/reasoning_engine.py tests/test_reasoning_engine.py
git commit -m "feat: add ReasoningEngine with Reason→Plan→Act→Observe loop"
```

---

## Task 7: Wire BaseAgent for Signals + WorkflowContext

**Files:**
- Modify: `src/agents/base_agent.py`
- Test: existing tests should continue passing

- [ ] **Step 1: Add signal emission to BaseAgent**

In `BaseAgent.execute()`, add:
- Accept optional `workflow_context: WorkflowContext` parameter
- Store reference to workflow_context on self during execution
- Add `emit_signal(signal_type, message, data)` helper method
- Pass workflow_context to `self.run()` via context object

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `cd src && python -m pytest ../tests/test_base_agent_ollama.py -v`

- [ ] **Step 3: Commit**

```bash
git add src/agents/base_agent.py
git commit -m "feat: add signal emission and WorkflowContext support to BaseAgent"
```

---

## Task 8: Wire main.py to use AutoRegistry

**Files:**
- Modify: `src/api/main.py`

- [ ] **Step 1: Replace manual agent registration**

In the `lifespan()` function, replace the manual agent instantiation block with:

```python
from agents.auto_registry import AutoRegistry
registry = AutoRegistry.from_json()
registry.set_agent_nick(agent_nick)
agent_nick.auto_registry = registry
# Keep legacy agents dict for backward compatibility
agent_nick.agents = registry
```

- [ ] **Step 2: Verify server starts**

Run: start uvicorn, check all agents initialize

- [ ] **Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat: wire AutoRegistry into main.py, replace manual agent registration"
```

---

## Task 9: Wire Orchestrator to Reasoning Engine

**Files:**
- Modify: `src/orchestration/orchestrator.py`

- [ ] **Step 1: Add reasoning engine delegation**

At the top of the orchestrator's main dispatch method, add:

```python
# Try reasoning engine first for supported task types
if self._reasoning_engine:
    try:
        result = self._reasoning_engine.process_task(task)
        if result and result.status != "fallback":
            return result
    except Exception:
        logger.debug("Reasoning engine fallback to legacy", exc_info=True)
# Fall through to existing legacy routing
```

- [ ] **Step 2: Run existing orchestrator tests**

Run: `cd src && python -m pytest ../tests/test_orchestrator_supplier_workflow.py -v`

- [ ] **Step 3: Commit**

```bash
git add src/orchestration/orchestrator.py
git commit -m "feat: delegate to ReasoningEngine with legacy orchestrator fallback"
```

---

## Task 10: Seed Initial Patterns + Integration Test

**Files:**
- Create: `src/services/seed_patterns.py`
- Test: `tests/test_integration_reasoning.py`

- [ ] **Step 1: Create pattern seeder**

Seed the pattern store with procurement knowledge derived from our verified extraction data:

```python
# src/services/seed_patterns.py
INITIAL_PATTERNS = [
    # Extraction patterns
    ("extraction", "UK invoices use DD/MM/YYYY dates and 20% VAT", "general", 0.95),
    ("extraction", "Supplier name appears in letterhead, not in PAYABLE TO section", "general", 0.90),
    ("extraction", "StructTree fallback needed for Canva-generated PDFs", "general", 0.85),
    # Category patterns
    ("category", "Office furniture typical markup 30-40% above wholesale", "office_furniture", 0.80),
    ("category", "IT equipment prices vary 15-25% across suppliers", "IT_equipment", 0.80),
    ("category", "Stationery high-volume orders (100+) trigger volume discount", "stationery", 0.75),
    # Negotiation patterns
    ("negotiation", "Cooperative strategy effective with suppliers having 5+ prior transactions", "general", 0.75),
    ("negotiation", "First counter-offer at 15% below ask yields average 8% discount", "office_furniture", 0.70),
    ("negotiation", "Bundling multiple categories increases discount probability by 40%", "general", 0.70),
    # Process patterns
    ("process", "Invoice without PO reference within 30 days needs follow-up", "general", 0.80),
    ("process", "Quote validity typically 7-30 days, check expiry proactively", "general", 0.85),
]
```

- [ ] **Step 2: Write integration test**

Test that the full reasoning loop works: task → reason → plan → (mock) execute → observe → complete

- [ ] **Step 3: Run integration test**

- [ ] **Step 4: Commit**

```bash
git add src/services/seed_patterns.py tests/test_integration_reasoning.py
git commit -m "feat: seed initial procurement patterns, add integration test for reasoning loop"
```

---

## Task 11: Update Modelfile with Orchestration Prompt

**Files:**
- Modify: `Modelfile`

- [ ] **Step 1: Add orchestration-aware sections**

Add to the Modelfile system prompt:
- Workflow composition instructions (how to output a plan as JSON)
- Available agent descriptions (template that gets filled at runtime)
- Procurement lifecycle state machine description
- Negotiation strategy playbook summary

- [ ] **Step 2: Rebuild model**

```bash
ollama create BeyondProcwise/AgentNick:latest -f Modelfile
```

- [ ] **Step 3: Commit**

```bash
git add Modelfile
git commit -m "feat: update Modelfile with orchestration and negotiation prompt sections"
```

---

## Task 12: Final Integration + Server Restart

- [ ] **Step 1: Run full test suite**

```bash
cd src && python -m pytest ../tests/ -v --tb=short -x
```

- [ ] **Step 2: Restart server and verify**

```bash
# Restart
kill $(lsof -ti :8000) 2>/dev/null
PYTHONPATH="..." .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir src
```

Verify:
- All agents auto-discovered from JSON
- Reasoning engine processes a document upload
- Pattern service stores and retrieves patterns
- Negotiation strategy engine selects appropriate strategy
- Procurement context service generates lifecycle brief

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete agentic re-engineering — reasoning engine, auto spawning, negotiation, patterns"
```
