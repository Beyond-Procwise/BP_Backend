"""Tests for the shared capability-degradation registry.

This module exists so that when a feature loses a dependency it can never
have (e.g. a DB table that was never created), the rest of the codebase has
ONE honest place to (a) record that fact, (b) log it exactly once instead of
spamming a traceback on every call, and (c) surface it to a human via
GET /health — instead of either crashing or silently pretending the feature
still works.
"""
import logging

import pytest


def _fresh_module():
    """Import a clean copy of the module so tests don't leak state into each
    other via the module-level registry singletons."""
    import importlib
    import services.capability_status as mod

    importlib.reload(mod)
    return mod


@pytest.fixture
def capability_status():
    return _fresh_module()


def test_mark_degraded_is_reflected_in_get_degraded(capability_status):
    capability_status.mark_degraded("thing", "reason for absence")
    assert capability_status.get_degraded() == [
        {"capability": "thing", "reason": "reason for absence"}
    ]


def test_get_degraded_is_empty_when_nothing_marked(capability_status):
    assert capability_status.get_degraded() == []


def test_mark_degraded_logs_a_warning_the_first_time(capability_status, caplog):
    with caplog.at_level(logging.WARNING):
        capability_status.mark_degraded("thing", "reason for absence")
    assert "thing" in caplog.text
    assert "reason for absence" in caplog.text


def test_mark_degraded_is_idempotent_and_does_not_relog(capability_status, caplog):
    """Calling mark_degraded repeatedly (e.g. once per workflow run) must not
    spam the log — only the FIRST call for a given capability should log."""
    capability_status.mark_degraded("thing", "reason for absence")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        capability_status.mark_degraded("thing", "reason for absence")
        capability_status.mark_degraded("thing", "reason for absence")
    assert caplog.text == ""


def test_mark_available_clears_a_degraded_capability(capability_status):
    capability_status.mark_degraded("thing", "reason for absence")
    capability_status.mark_available("thing")
    assert capability_status.get_degraded() == []


def test_multiple_capabilities_are_all_reported(capability_status):
    capability_status.mark_degraded("a", "reason a")
    capability_status.mark_degraded("b", "reason b")
    degraded = capability_status.get_degraded()
    assert {"capability": "a", "reason": "reason a"} in degraded
    assert {"capability": "b", "reason": "reason b"} in degraded
    assert len(degraded) == 2


def test_log_once_only_emits_a_single_log_line_per_key(capability_status, caplog):
    with caplog.at_level(logging.WARNING):
        capability_status.log_once("k1", logging.WARNING, "first message")
        capability_status.log_once("k1", logging.WARNING, "second message (should not appear)")
    assert caplog.text.count("first message") == 1
    assert "second message" not in caplog.text


def test_log_once_different_keys_both_log(capability_status, caplog):
    with caplog.at_level(logging.WARNING):
        capability_status.log_once("k1", logging.WARNING, "message one")
        capability_status.log_once("k2", logging.WARNING, "message two")
    assert "message one" in caplog.text
    assert "message two" in caplog.text
