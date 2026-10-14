"""The last code paths that wrote a stand-in where a person belongs, and a guard.

Clearing the old rows (deploy/sql/2026-09-11_clear_standin_actors.sql) is
pointless while code still writes new ones. Two repositories defaulted
created_by to the literal "system", and the canvas's "save workflow" endpoint
never passed a created_by at all -- which is why all ten saved workflows in
bp_testdb say they were created by "system".
"""

from __future__ import annotations

import ast
import inspect
import pathlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import require_user

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALLER = "sub-real-caller"

_ACTOR_PARAMS = {
    "created_by", "modified_by", "last_modified_by", "answered_by", "actioned_by",
    "resolved_by", "reviewed_by", "confirmed_by", "rejected_by", "triggered_by",
    "added_by", "stated_by", "decided_by", "approved_by", "requested_by",
    "initiated_by", "user_id", "user_name", "reviewer", "approver",
}


class _Principal:
    def __init__(self, subject):
        self.subject = subject


def _save(monkeypatch, subject):
    from api.routers import agent_workflows as aw

    seen = {}
    monkeypatch.setattr(aw, "gate", lambda *a, **k: None)
    monkeypatch.setattr(aw, "validate_saved_graph", lambda graph: None)
    monkeypatch.setattr(aw, "_entry_of", lambda graph: "start")
    monkeypatch.setattr(aw.repo, "create", lambda **k: seen.update(k) or 11)

    app = FastAPI()
    app.include_router(aw.router)
    app.dependency_overrides[require_user] = (
        (lambda: _Principal(subject)) if subject else (lambda: None))
    r = TestClient(app).post("/agent-workflows",
                             json={"name": "w", "graph": {"nodes": [], "edges": []}})
    return r, seen


def test_a_saved_canvas_workflow_is_created_by_the_token(monkeypatch):
    r, seen = _save(monkeypatch, CALLER)

    assert r.status_code == 200, r.text
    assert seen.get("created_by") == CALLER, (
        f"the saved workflow names {seen.get('created_by')!r}, not the caller")


def test_a_canvas_workflow_saved_by_nobody_is_created_by_nobody(monkeypatch):
    r, seen = _save(monkeypatch, None)

    assert "created_by" in seen and seen["created_by"] is None, seen


def test_the_repositories_do_not_default_to_a_standin():
    from repositories import agent_group_repo, agent_workflow_repo

    for module in (agent_workflow_repo, agent_group_repo):
        default = inspect.signature(module.create).parameters["created_by"].default
        assert default is None, f"{module.__name__}.create defaults created_by to {default!r}"


def _standin_defaults() -> list:
    """Function parameters naming an actor, defaulted to a name."""
    hits = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.args + node.args.kwonlyargs
            defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                        + list(node.args.defaults) + list(node.args.kw_defaults))
            for arg, default in zip(args, defaults):
                if (arg.arg in _ACTOR_PARAMS and isinstance(default, ast.Constant)
                        and isinstance(default.value, str) and default.value):
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                f"{node.name}({arg.arg}={default.value!r})")
    return hits


def test_no_actor_parameter_defaults_to_a_name():
    hits = _standin_defaults()
    assert not hits, (
        "an actor parameter that defaults to a name rather than a person -- "
        "default it to None:\n  " + "\n  ".join(hits))


def test_the_default_guard_can_see_one():
    tree = ast.parse('def create(name, created_by="system"): pass')
    fn = tree.body[0]
    assert fn.args.args[1].arg in _ACTOR_PARAMS
    assert fn.args.defaults[0].value == "system"
