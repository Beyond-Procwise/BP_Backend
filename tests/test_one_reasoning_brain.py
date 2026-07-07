"""One reasoning brain: all non-extraction routing defaults to AgentNick:unified."""
UNIFIED = "BeyondProcwise/AgentNick:unified"
LATEST = "BeyondProcwise/AgentNick:latest"


def test_settings_reasoning_defaults_are_unified():
    from config.settings import Settings
    f = Settings.model_fields
    assert f["local_primary_model"].default == UNIFIED
    assert f["local_fallback_model"].default == UNIFIED
    assert f["extraction_model"].default == UNIFIED
    assert f["rag_model"].default == UNIFIED


def test_base_agent_universal_fallback_is_unified():
    import agents.base_agent as ba
    assert ba._UNIVERSAL_LOCAL_MODEL == UNIFIED
    chain = ba._build_fallback_models()
    assert UNIFIED in chain
    assert LATEST not in chain


def test_reasoning_engine_model_is_unified():
    import orchestration.reasoning_engine as re_mod
    assert re_mod._OLLAMA_MODEL == UNIFIED


def test_extraction_routing_untouched():
    # Extraction must still target the specialist, not the reasoning brain.
    from src.services.ollama_client import DEFAULT_MODEL
    from src.services.extraction.context_layer import _LLM_MODEL
    assert DEFAULT_MODEL == "BeyondProcwise/AgentNick:extract"
    assert _LLM_MODEL == "BeyondProcwise/AgentNick:extract"
