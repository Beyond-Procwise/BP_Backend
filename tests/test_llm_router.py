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
