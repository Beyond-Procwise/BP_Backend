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
