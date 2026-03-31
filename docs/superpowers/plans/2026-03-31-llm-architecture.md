# LLM Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM Router that routes tasks to local Qwen2.5-32B (with LoRA adapters) or Ollama Cloud, with a self-improving training loop that uses extraction corrections as training data.

**Architecture:** A new `LLMRouter` service sits between agents and Ollama. It decides whether to route to local GPU (for deep extraction, negotiation, reasoning) or Ollama Cloud (for classification, summarization, validation) based on task type and GPU queue depth. Three LoRA adapters (extraction, negotiation, general) are managed via Ollama Modelfiles. The BackendScheduler collects remediation corrections and triggers periodic fine-tuning.

**Tech Stack:** Python 3.12, Ollama (local + cloud), Qwen2.5-32B/7B, existing QLoRA/Unsloth pipeline, Redis (queue depth monitoring)

**Spec:** `docs/superpowers/specs/2026-03-31-orchestration-rearchitecture-design.md` Section 8

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/services/llm_router.py` | Routes LLM requests to local GPU or Ollama Cloud based on task type and queue depth |
| `src/services/training_data_collector.py` | Collects remediation corrections and extraction results as fine-tuning training data |
| `src/resources/modelfiles/Modelfile.procwise-extract` | Ollama Modelfile for extraction LoRA adapter |
| `src/resources/modelfiles/Modelfile.procwise-negotiate` | Ollama Modelfile for negotiation LoRA adapter |
| `src/resources/modelfiles/Modelfile.procwise-general` | Ollama Modelfile for general LoRA adapter |
| `tests/test_llm_router.py` | LLM Router tests |
| `tests/test_training_data_collector.py` | Training data collector tests |

### Modified Files

| File | Change |
|------|--------|
| `config/settings.py` | Add Ollama Cloud URL, model names, adapter paths, training collection settings |
| `src/agents/base_agent.py` | Update `_build_phi4_fallback_models` → `_build_fallback_models` with Qwen2.5, integrate LLM Router into `call_ollama()` |
| `src/services/backend_scheduler.py` | Add training data collection + fine-tuning job |

### Unchanged Files

| File | Why |
|------|-----|
| `src/training/pipeline.py` | Existing QLoRA pipeline reused — just new configs passed in |
| `src/services/model_training_service.py` | Existing service — new training job calls it with new adapter configs |
| `src/agents/ml_extraction_pipeline.py` | Calls agent's call_ollama() which now routes via LLMRouter |

---

## Task 1: LLM Router Service

**Files:**
- Create: `src/services/llm_router.py`
- Create: `tests/test_llm_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_router.py
"""Tests for the LLM Router — routes requests to local or cloud Ollama."""
import pytest
from unittest.mock import MagicMock, patch


def test_extraction_routes_to_local():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    tier = router.route(task_type="extraction")
    assert tier == Tier.LOCAL


def test_negotiation_routes_to_local():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    tier = router.route(task_type="negotiation")
    assert tier == Tier.LOCAL


def test_classification_routes_to_cloud():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    tier = router.route(task_type="classification")
    assert tier == Tier.CLOUD


def test_summarization_routes_to_cloud():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    tier = router.route(task_type="summarization")
    assert tier == Tier.CLOUD


def test_local_busy_degrades_non_critical_to_cloud():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
        queue_depth_threshold=8,
    )
    # Simulate high queue depth
    router._get_queue_depth = MagicMock(return_value=10)
    tier = router.route(task_type="summarization")
    assert tier == Tier.CLOUD


def test_local_busy_keeps_critical_on_local():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
        queue_depth_threshold=8,
    )
    router._get_queue_depth = MagicMock(return_value=10)
    # Extraction is critical — never degrades to cloud
    tier = router.route(task_type="extraction")
    assert tier == Tier.LOCAL


def test_select_model_returns_adapter_for_extraction():
    from services.llm_router import LLMRouter

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    model = router.select_model(task_type="extraction")
    assert "extract" in model


def test_select_model_returns_adapter_for_negotiation():
    from services.llm_router import LLMRouter

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    model = router.select_model(task_type="negotiation")
    assert "negotiate" in model


def test_get_base_url_returns_correct_tier():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url="http://localhost:11434",
        cloud_base_url="https://cloud.ollama.com",
    )
    assert router.get_base_url(Tier.LOCAL) == "http://localhost:11434"
    assert router.get_base_url(Tier.CLOUD) == "https://cloud.ollama.com"


def test_cloud_only_mode_when_no_local():
    from services.llm_router import LLMRouter, Tier

    router = LLMRouter(
        local_base_url=None,
        cloud_base_url="https://cloud.ollama.com",
    )
    tier = router.route(task_type="extraction")
    assert tier == Tier.CLOUD  # Falls back to cloud when no local
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_llm_router.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement LLMRouter**

```python
# src/services/llm_router.py
"""LLM Router: routes requests to local GPU or Ollama Cloud.

Decides between local fine-tuned models (Qwen2.5-32B + LoRA adapters)
and Ollama Cloud (lightweight models) based on task type and GPU queue depth.

Spec reference: Section 8 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Tier(Enum):
    LOCAL = "local"    # Local GPU (A10G) with fine-tuned models
    CLOUD = "cloud"    # Ollama hosted API for lightweight tasks


# Task types that MUST stay on local GPU (never degrade to cloud)
CRITICAL_TASKS = frozenset({
    "extraction", "negotiation", "quote_evaluation",
    "opportunity_mining", "rag", "policy_interpretation",
})

# Task types that prefer cloud (fast, lightweight)
CLOUD_PREFERRED_TASKS = frozenset({
    "classification", "summarization", "field_validation",
    "entity_normalization", "simple_email",
})

# Model names for local LoRA adapters
LOCAL_ADAPTER_MODELS = {
    "extraction": "procwise-extract",
    "negotiation": "procwise-negotiate",
    "quote_evaluation": "procwise-general",
    "opportunity_mining": "procwise-general",
    "rag": "procwise-general",
    "policy_interpretation": "procwise-general",
}

# Default cloud model
CLOUD_MODEL = "qwen2.5:7b"


class LLMRouter:
    def __init__(
        self,
        local_base_url: Optional[str] = "http://localhost:11434",
        cloud_base_url: Optional[str] = None,
        queue_depth_threshold: int = 8,
        get_queue_depth_func: Optional[Callable] = None,
    ):
        self._local_url = local_base_url
        self._cloud_url = cloud_base_url
        self._queue_threshold = queue_depth_threshold
        self._get_queue_depth = get_queue_depth_func or (lambda: 0)

    def route(self, task_type: str) -> Tier:
        # No local GPU available — everything goes to cloud
        if not self._local_url:
            return Tier.CLOUD

        # Critical tasks always stay local
        if task_type in CRITICAL_TASKS:
            return Tier.LOCAL

        # Cloud-preferred tasks go to cloud
        if task_type in CLOUD_PREFERRED_TASKS:
            return Tier.CLOUD

        # For other tasks, check GPU queue depth
        queue_depth = self._get_queue_depth()
        if queue_depth > self._queue_threshold:
            logger.info(
                "GPU queue depth %d > threshold %d, routing %s to cloud",
                queue_depth, self._queue_threshold, task_type,
            )
            return Tier.CLOUD

        return Tier.LOCAL

    def select_model(self, task_type: str) -> str:
        tier = self.route(task_type)
        if tier == Tier.LOCAL:
            return LOCAL_ADAPTER_MODELS.get(task_type, "procwise-general")
        return CLOUD_MODEL

    def get_base_url(self, tier: Tier) -> str:
        if tier == Tier.LOCAL:
            return self._local_url or self._cloud_url or "http://localhost:11434"
        return self._cloud_url or self._local_url or "http://localhost:11434"
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_llm_router.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/llm_router.py tests/test_llm_router.py
git commit -m "feat: add LLM Router for local/cloud model routing"
```

---

## Task 2: Training Data Collector

**Files:**
- Create: `src/services/training_data_collector.py`
- Create: `tests/test_training_data_collector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training_data_collector.py
"""Tests for training data collection from extraction corrections."""
import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock


def test_collect_extraction_correction():
    from services.training_data_collector import TrainingDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrainingDataCollector(output_dir=tmpdir)
        collector.record_correction(
            doc_type="Invoice",
            document_text="Invoice #INV-001\nSupplier: Acme\nTotal: $1500",
            original_fields={"invoice_id": "INV-00", "supplier_name": "Acm"},
            corrected_fields={"invoice_id": "INV-001", "supplier_name": "Acme"},
            correction_source="remediation",
        )
        # Should write to extraction adapter training file
        path = os.path.join(tmpdir, "extraction_corrections.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            line = json.loads(f.readline())
        assert line["doc_type"] == "Invoice"
        assert line["corrected_fields"]["invoice_id"] == "INV-001"


def test_collect_negotiation_example():
    from services.training_data_collector import TrainingDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrainingDataCollector(output_dir=tmpdir)
        collector.record_negotiation(
            workflow_id="wf-001",
            supplier_name="Acme Corp",
            round_num=2,
            strategy="competitive",
            counter_offer="We can offer 10% discount",
            outcome="accepted",
        )
        path = os.path.join(tmpdir, "negotiation_examples.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            line = json.loads(f.readline())
        assert line["outcome"] == "accepted"


def test_get_training_stats():
    from services.training_data_collector import TrainingDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrainingDataCollector(output_dir=tmpdir)
        collector.record_correction(
            doc_type="Invoice",
            document_text="test",
            original_fields={},
            corrected_fields={"invoice_id": "INV-001"},
            correction_source="remediation",
        )
        collector.record_correction(
            doc_type="Quote",
            document_text="test2",
            original_fields={},
            corrected_fields={"quote_id": "Q-001"},
            correction_source="remediation",
        )
        stats = collector.get_stats()
        assert stats["extraction_corrections"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_training_data_collector.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TrainingDataCollector**

```python
# src/services/training_data_collector.py
"""Collects extraction corrections and agent outputs as training data.

Part of the self-improving loop: remediation corrections become training
examples for the next fine-tuning cycle.

Spec reference: Section 8 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    def __init__(self, output_dir: str = "data/training"):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def record_correction(
        self,
        doc_type: str,
        document_text: str,
        original_fields: Dict[str, Any],
        corrected_fields: Dict[str, Any],
        correction_source: str = "remediation",
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_type": doc_type,
            "document_text": document_text[:2000],  # Truncate for storage
            "original_fields": original_fields,
            "corrected_fields": corrected_fields,
            "correction_source": correction_source,
        }
        self._append_jsonl("extraction_corrections.jsonl", entry)
        logger.debug("Recorded extraction correction for %s", doc_type)

    def record_negotiation(
        self,
        workflow_id: str,
        supplier_name: str,
        round_num: int,
        strategy: str,
        counter_offer: str,
        outcome: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_id": workflow_id,
            "supplier_name": supplier_name,
            "round": round_num,
            "strategy": strategy,
            "counter_offer": counter_offer,
            "outcome": outcome,
        }
        self._append_jsonl("negotiation_examples.jsonl", entry)
        logger.debug("Recorded negotiation example for %s round %d", supplier_name, round_num)

    def get_stats(self) -> Dict[str, int]:
        stats = {}
        for filename, key in [
            ("extraction_corrections.jsonl", "extraction_corrections"),
            ("negotiation_examples.jsonl", "negotiation_examples"),
        ]:
            path = os.path.join(self._output_dir, filename)
            if os.path.exists(path):
                with open(path) as f:
                    stats[key] = sum(1 for _ in f)
            else:
                stats[key] = 0
        return stats

    def _append_jsonl(self, filename: str, entry: Dict) -> None:
        path = os.path.join(self._output_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_training_data_collector.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/training_data_collector.py tests/test_training_data_collector.py
git commit -m "feat: add training data collector for self-improving loop"
```

---

## Task 3: Ollama Modelfile Templates

**Files:**
- Create: `src/resources/modelfiles/Modelfile.procwise-extract`
- Create: `src/resources/modelfiles/Modelfile.procwise-negotiate`
- Create: `src/resources/modelfiles/Modelfile.procwise-general`

- [ ] **Step 1: Create the Modelfile templates**

```dockerfile
# src/resources/modelfiles/Modelfile.procwise-extract
# Procurement extraction adapter on Qwen2.5-32B base
# Usage: ollama create procwise-extract -f Modelfile.procwise-extract
FROM qwen2.5:32b

# LoRA adapter path (update after fine-tuning)
# ADAPTER /path/to/extraction-adapter

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM """You are a procurement document extraction specialist. Extract structured data from invoices, purchase orders, quotes, and contracts with high precision. Always return valid JSON with field names matching the procurement schema."""
```

```dockerfile
# src/resources/modelfiles/Modelfile.procwise-negotiate
# Procurement negotiation adapter on Qwen2.5-32B base
# Usage: ollama create procwise-negotiate -f Modelfile.procwise-negotiate
FROM qwen2.5:32b

# LoRA adapter path (update after fine-tuning)
# ADAPTER /path/to/negotiation-adapter

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192

SYSTEM """You are a procurement negotiation specialist. Analyze supplier offers, identify negotiation leverage, draft counter-offers, and recommend strategies based on market conditions and historical data."""
```

```dockerfile
# src/resources/modelfiles/Modelfile.procwise-general
# General procurement knowledge adapter on Qwen2.5-32B base
# Usage: ollama create procwise-general -f Modelfile.procwise-general
FROM qwen2.5:32b

# LoRA adapter path (update after fine-tuning)
# ADAPTER /path/to/general-adapter

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM """You are a procurement intelligence assistant. Answer questions about suppliers, contracts, policies, and procurement best practices using the provided context."""
```

- [ ] **Step 2: Commit**

```bash
git add src/resources/modelfiles/
git commit -m "feat: add Ollama Modelfile templates for LoRA adapters"
```

---

## Task 4: Settings Update

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Read current settings**

Read: `config/settings.py` to find model-related settings sections.

- [ ] **Step 2: Add LLM Router and model settings**

Add the following settings after the existing Ollama settings:

```python
# LLM Router
ollama_cloud_base_url: Optional[str] = None  # Ollama Cloud API URL
llm_router_queue_depth_threshold: int = 8  # Route non-critical to cloud above this
local_primary_model: str = "qwen2.5:32b"  # Primary local model (env: LOCAL_PRIMARY_MODEL)
local_fallback_model: str = "qwen2.5:7b"  # Fallback local model (env: LOCAL_FALLBACK_MODEL)

# LoRA Adapter Paths
extraction_adapter_path: Optional[str] = None  # Path to extraction LoRA adapter
negotiation_adapter_path: Optional[str] = None  # Path to negotiation LoRA adapter
general_adapter_path: Optional[str] = None  # Path to general LoRA adapter

# Training Data Collection
training_data_output_dir: str = "data/training"  # Where corrections are collected
```

- [ ] **Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat: add LLM Router and model migration settings"
```

---

## Task 5: Update BaseAgent Fallback Chain

**Files:**
- Modify: `src/agents/base_agent.py`

- [ ] **Step 1: Read the current fallback model builder**

Read: `src/agents/base_agent.py:47-64` (the `_build_phi4_fallback_models` function)

- [ ] **Step 2: Update fallback chain to Qwen2.5**

Replace `_build_phi4_fallback_models` with a version that prefers Qwen2.5:

```python
def _build_fallback_models() -> Tuple[str, ...]:
    """Return fallback model chain preferring Qwen2.5."""
    configured = getattr(settings, "local_primary_model", None)
    fallback = getattr(settings, "local_fallback_model", None)
    candidates: List[str] = []
    for name in (configured, fallback, "qwen2.5:32b", "qwen2.5:7b", "phi4:latest"):
        if name and name not in candidates:
            candidates.append(name)
    return tuple(candidates) if candidates else ("qwen2.5:32b",)

_OLLAMA_FALLBACK_MODELS: Tuple[str, ...] = _build_fallback_models()
```

Also update `_AGENT_MODEL_FIELD_PREFERENCES` comment to note the Qwen2.5 migration.

- [ ] **Step 3: Verify existing tests still pass**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_data_extraction_agent.py -v 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add src/agents/base_agent.py
git commit -m "feat: update model fallback chain from Phi4 to Qwen2.5"
```

---

## Task 6: Run Full Test Suite

- [ ] **Step 1: Run all new LLM tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_llm_router.py tests/test_training_data_collector.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run complete test suite**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_message_protocol.py tests/test_state_manager.py tests/test_task_dispatcher.py tests/test_dag_scheduler.py tests/test_result_collector.py tests/test_worker.py tests/test_integration_workflow.py tests/test_adaptive_ocr.py tests/test_validation_gate.py tests/test_remediation_service.py tests/test_llm_router.py tests/test_training_data_collector.py -v`
Expected: All tests PASS
