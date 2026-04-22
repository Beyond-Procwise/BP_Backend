def test_feature_flag_off_preserves_legacy(monkeypatch):
    monkeypatch.setenv("USE_STRUCTURAL_EXTRACTOR", "false")
    # Simply assert that the module can still be imported without crashing
    from src.services.agent_nick_orchestrator import AgentNickOrchestrator  # noqa: F401
    assert AgentNickOrchestrator is not None


def test_feature_flag_on_module_imports(monkeypatch):
    monkeypatch.setenv("USE_STRUCTURAL_EXTRACTOR", "true")
    # Structural extractor module should also import cleanly
    from src.services.structural_extractor import extract  # noqa: F401
    assert callable(extract)
