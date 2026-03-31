"""LLM Router: routes requests to local GPU or Ollama Cloud.

Decides between local fine-tuned models (Qwen2.5-32B + LoRA adapters)
and Ollama Cloud (lightweight models) based on task type and GPU queue depth.
Provides authenticated Ollama clients for each tier.

Spec reference: Section 8 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional

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
        cloud_api_key: Optional[str] = None,
        queue_depth_threshold: int = 8,
        get_queue_depth_func: Optional[Callable] = None,
    ):
        self._local_url = local_base_url
        self._cloud_url = cloud_base_url
        self._cloud_api_key = cloud_api_key
        self._queue_threshold = queue_depth_threshold
        self._get_queue_depth = get_queue_depth_func or (lambda: 0)
        self._cloud_client = None
        self._local_client = None

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

    def get_client(self, tier: Tier) -> Any:
        """Return an authenticated Ollama client for the given tier.

        For CLOUD tier, the client includes the API key in the Authorization
        header. For LOCAL tier, returns a standard client pointing at localhost.
        """
        try:
            import ollama as ollama_lib
        except ImportError:
            logger.error("ollama package not installed")
            return None

        if tier == Tier.CLOUD:
            if self._cloud_client is None:
                url = self.get_base_url(Tier.CLOUD)
                headers = {}
                if self._cloud_api_key:
                    headers["Authorization"] = f"Bearer {self._cloud_api_key}"
                self._cloud_client = ollama_lib.Client(
                    host=url,
                    headers=headers,
                )
                logger.info("Initialized Ollama Cloud client at %s", url)
            return self._cloud_client

        if self._local_client is None:
            url = self.get_base_url(Tier.LOCAL)
            self._local_client = ollama_lib.Client(host=url)
            logger.info("Initialized Ollama Local client at %s", url)
        return self._local_client

    def call(
        self,
        task_type: str,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        model: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Route and execute an LLM call to the appropriate tier.

        This is the main entry point for agents that want tier-aware routing.
        """
        tier = self.route(task_type)
        target_model = model or self.select_model(task_type)
        client = self.get_client(tier)

        if client is None:
            return {"response": "", "error": "No Ollama client available"}

        try:
            if messages is not None:
                response = client.chat(
                    model=target_model,
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
            else:
                call_kwargs = dict(kwargs)
                if format:
                    call_kwargs["format"] = format
                response = client.generate(
                    model=target_model,
                    prompt=prompt or "",
                    stream=False,
                    **call_kwargs,
                )
            logger.info(
                "LLM call routed to %s (model=%s, task=%s)",
                tier.value, target_model, task_type,
            )
            return response
        except Exception as exc:
            logger.error(
                "LLM call failed on %s tier (model=%s): %s",
                tier.value, target_model, exc,
            )
            # If cloud fails, try local as fallback (and vice versa)
            fallback_tier = Tier.LOCAL if tier == Tier.CLOUD else Tier.CLOUD
            fallback_client = self.get_client(fallback_tier)
            if fallback_client is not None and fallback_client is not client:
                try:
                    fallback_model = target_model if tier == Tier.LOCAL else CLOUD_MODEL
                    if messages is not None:
                        return fallback_client.chat(
                            model=fallback_model,
                            messages=messages,
                            stream=False,
                            **kwargs,
                        )
                    call_kwargs = dict(kwargs)
                    if format:
                        call_kwargs["format"] = format
                    return fallback_client.generate(
                        model=fallback_model,
                        prompt=prompt or "",
                        stream=False,
                        **call_kwargs,
                    )
                except Exception as fallback_exc:
                    logger.error("Fallback to %s also failed: %s", fallback_tier.value, fallback_exc)

            return {"response": "", "error": str(exc)}
