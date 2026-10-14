# test_feature_flag_off_preserves_legacy asserted that
# src.services.agent_nick_orchestrator imports cleanly. That module was removed
# in 114e5ca (2026-05-09) with the rest of the legacy extraction stack, so the
# test could only ever fail, and had been failing since. Removed rather than
# repointed: there is no "legacy" side left for the flag to preserve.


def test_feature_flag_on_module_imports(monkeypatch):
    monkeypatch.setenv("USE_STRUCTURAL_EXTRACTOR", "true")
    # Structural extractor module should also import cleanly
    from src.services.structural_extractor import extract  # noqa: F401
    assert callable(extract)
