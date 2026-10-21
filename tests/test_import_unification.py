"""One copy of a module that holds process-wide state, however it is imported.

This codebase imports its own modules two ways: ``services.X`` and ``src.services.X``. Python loads those
as two separate modules, each with its own globals. ``ollama_client`` keeps a count of foreground calls
(which background work yields to), a queue, and a memory of GPU layout refusals; with two copies the
foreground work done through one is invisible through the other. ``src/services/__init__.py`` makes the
modules it lists share ONE copy. Module identity depends on import order, so each check runs in a fresh
interpreter.
"""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(code: str, *, path=("", "src"), cwd=None):
    """Run ``code`` in a fresh interpreter. ``path`` entries are relative to the repo root."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(os.path.join(ROOT, p) if p else ROOT for p in path),
           "CUDA_VISIBLE_DEVICES": "", "OLLAMA_BASE_URL": "http://127.0.0.1:1"}
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)], cwd=cwd or ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    return out.stdout.strip()


def test_the_two_names_are_one_module_whichever_is_imported_first():
    assert run("""
        import services.ollama_client as a
        import src.services.ollama_client as b
        print(a is b)""") == "True"
    assert run("""
        import src.services.ollama_client as b
        import services.ollama_client as a
        print(a is b)""") == "True"


def test_the_from_import_and_attribute_forms_agree():
    assert run("""
        from services import ollama_client as a
        from src.services import ollama_client as b
        from services.ollama_client import ollama_generate as g1
        from src.services.ollama_client import ollama_generate as g2
        print(a is b, g1 is g2)""") == "True True"


def test_the_foreground_state_and_the_refusal_memory_are_shared():
    assert run("""
        import services.ollama_client as a
        import src.services.ollama_client as b
        a.note_layout_rejection("test", now=1000.0)
        seen_by_b = b._LAYOUT_REJECTED_UNTIL > 0
        b.clear_layout_rejection()
        cleared_for_a = a._LAYOUT_REJECTED_UNTIL == 0.0
        a._foreground_count = 3
        print(seen_by_b, cleared_for_a, b.foreground_busy(), a._foreground_lock is b._foreground_lock)""") == \
        "True True True True"


def test_modules_that_are_not_listed_are_left_exactly_as_they_were():
    # Only the listed modules are aliased; anything else still loads under whichever name asks for it.
    assert run("""
        import services.agentnick_driver as a
        print(a.__name__)""") == "services.agentnick_driver"


def test_a_listed_module_has_always_needed_src_on_the_path_and_still_does(tmp_path):
    """ollama_client imports ``src.services.egress`` itself, so with only ``src`` on the path it cannot load,
    hook or no hook. The hook adds no new requirement."""
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", "import services.ollama_client"], cwd=str(tmp_path),
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "ModuleNotFoundError" in out.stderr and "src" in out.stderr


def test_every_module_on_the_shared_list_exists():
    import importlib.util
    pkg = importlib.util.spec_from_file_location("_svc_init", os.path.join(ROOT, "src", "services", "__init__.py"))
    mod = importlib.util.module_from_spec(pkg)
    pkg.loader.exec_module(mod)
    missing = [m for m in mod._SHARED_STATE_MODULES if not os.path.isfile(os.path.join(ROOT, "src", "services", f"{m}.py"))
               and not os.path.isdir(os.path.join(ROOT, "src", "services", m))]
    assert mod._SHARED_STATE_MODULES and missing == []


def test_callers_that_import_it_either_way_now_hold_the_same_function():
    assert run("""
        import services.style.exemplars as one          # imports services.ollama_client lazily
        import src.services.analytics.insight as two    # imports src.services.ollama_client lazily
        import services.ollama_client as a
        import src.services.ollama_client as b
        from services.ollama_client import ollama_generate as g1
        from src.services.ollama_client import ollama_generate as g2
        print(g1 is g2 is a.ollama_generate is b.ollama_generate)""") == "True"


# ---- event_bus ----------------------------------------------------------------------------------------

def test_the_event_bus_is_one_module_and_one_bus_whichever_name_is_used():
    assert run("""
        import services.event_bus as a
        import src.services.event_bus as b
        print(a is b, a.get_event_bus() is b.get_event_bus())""") == "True True"
    assert run("""
        import src.services.event_bus as b
        import services.event_bus as a
        print(a is b, a.get_event_bus() is b.get_event_bus())""") == "True True"


def test_a_subscriber_on_one_name_hears_an_event_published_through_the_other():
    """The mail-intake sweep publishes through src.services.event_bus; subscribers used services.event_bus."""
    assert run("""
        import services.event_bus as a
        import src.services.event_bus as b
        heard = []
        a.get_event_bus().subscribe("document.ingested", heard.append)
        b.get_event_bus().publish("document.ingested", {"id": 7})
        a.get_event_bus().publish("document.ingested", {"id": 8})
        print(heard)""") == "[{'id': 7}, {'id': 8}]"


def test_a_one_shot_subscriber_is_used_up_once_across_both_names():
    assert run("""
        import services.event_bus as a
        import src.services.event_bus as b
        heard = []
        a.get_event_bus().subscribe("e", heard.append, once=True)
        b.get_event_bus().publish("e", {"n": 1})
        b.get_event_bus().publish("e", {"n": 2})
        print(len(heard))""") == "1"


def test_the_workflow_context_set_through_one_name_is_seen_through_the_other():
    assert run("""
        import services.event_bus as a
        import src.services.event_bus as b
        with a.workflow_scope(workflow_id="wf-1", agent_name="x"):
            seen = b.get_current_workflow()
        print(seen and seen.get("workflow_id"), b.get_current_workflow())""") == "wf-1 None"


def test_a_handler_that_raises_does_not_stop_the_others_across_both_names():
    assert run("""
        import logging; logging.disable(logging.CRITICAL)
        import services.event_bus as a
        import src.services.event_bus as b
        got = []
        a.get_event_bus().subscribe("e", lambda p: 1 / 0)
        b.get_event_bus().subscribe("e", got.append)
        a.get_event_bus().publish("e", {"ok": True})
        print(got)""") == "[{'ok': True}]"
