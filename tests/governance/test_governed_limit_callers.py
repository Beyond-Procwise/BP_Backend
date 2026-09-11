"""A governed limit is a function now, and every caller has to call it.

59fd971 turned the module constants holding thirty-three limits into functions,
because a module-level read runs at import, before anything knows whether the
governance store answered. Two callers in the promotion code kept comparing
against the NAME:

    if link["F"] < MIN_LINK_SCORE:

Python 3 refuses a float compared with a function -- at runtime, and only on the
path that reaches the line. That path is the promotion decision itself, and the
loop around it has no per-document guard, so the first staged invoice that found
its PO and cleared the confidence bar would have taken the whole promotion pass
down with it, and the human review queue with it. Nothing noticed, because no
document on this box has reached that line since.

The scan below is what stops the next one. It is written against the source, not
a fixture, because the failure only exists on paths the suite does not walk.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _module_name(path: pathlib.Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _norm(module: str) -> str:
    return module[4:] if module.startswith("src.") else module


def _reads_a_governed_limit(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_governed_limit":
                return True
            if (isinstance(f, ast.Attribute) and f.attr == "limit"
                    and isinstance(f.value, ast.Name)
                    and f.value.id in ("governed_limits", "GL")):
                return True
    return False


def _accessors() -> dict:
    """{module: {function names}} for every module-level governed-limit reader."""
    found: dict = {}
    for path in (ROOT / "src").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if (isinstance(node, ast.FunctionDef) and not node.decorator_list
                    and _reads_a_governed_limit(node)):
                found.setdefault(_module_name(path), set()).add(node.name)
    return found


def _uncalled_uses(path: pathlib.Path, accessors: dict) -> list:
    """Every load of a governed-limit accessor in ``path`` that is not a call.

    Resolved through the file's own imports, so a module that happens to have
    its own MIN_CONFIDENCE constant is not mistaken for a caller of the
    governed one.
    """
    tree = ast.parse(path.read_text())
    bare = {name: name for name in accessors.get(_module_name(path), set())}
    modules: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            src = _norm(node.module)
            for a in node.names:
                if a.name in accessors.get(src, ()):
                    bare[a.asname or a.name] = a.name
                sub = f"{src}.{a.name}"
                if sub in accessors:
                    modules[a.asname or a.name] = accessors[sub]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _norm(a.name) in accessors and a.asname:
                    modules[a.asname] = accessors[_norm(a.name)]

    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    hits = []
    for node in ast.walk(tree):
        if id(node) in called:
            continue
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in bare):
            hits.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.id}")
        elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
              and isinstance(node.value, ast.Name)
              and node.attr in modules.get(node.value.id, ())):
            hits.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"{node.value.id}.{node.attr}")
    return hits


def test_the_scan_finds_the_accessors_it_is_meant_to_police():
    """A guard that finds nothing to check passes forever. Pin that it sees them."""
    acc = _accessors()
    assert "MIN_LINK_SCORE" in acc.get("services.linking_engine", set()), acc
    assert sum(len(v) for v in acc.values()) >= 25, acc


def test_no_governed_limit_is_used_without_being_called():
    acc = _accessors()
    hits = []
    for folder in ("src", "tests", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            try:
                hits += _uncalled_uses(path, acc)
            except (SyntaxError, UnicodeDecodeError):
                continue
    assert not hits, "governed limits used as values, not called:\n  " + \
        "\n  ".join(sorted(hits))


# ---------------------------------------------------------------------------
# the promotion decision, which the uncalled comparison sat inside
# ---------------------------------------------------------------------------
def _signals(identity_conflict: bool):
    return [
        {"id": "po_ref", "status": "OK", "tier": 1, "c": 0.3},
        {"id": "supplier", "status": "CONFLICT" if identity_conflict else "OK",
         "tier": 1, "c": 0.4},
        {"id": "amount", "status": "CONFLICT", "tier": 2, "c": 0.5},
    ]


def _stub_scoring(monkeypatch, LE, F, identity_conflict):
    monkeypatch.setattr(LE, "_find_parent_po", lambda cur, ref: {"po_id": "PO1"})
    monkeypatch.setattr(LE, "_resolve_po_supplier_id", lambda cur, po: None)
    monkeypatch.setattr(LE, "_rows", lambda cur, sql, params=None: [])
    monkeypatch.setattr(LE, "_set_amount_for_invoice", lambda cur, po_id: None)
    monkeypatch.setattr(LE, "score_link", lambda *a, **k: {
        "F": F, "decision": "review", "signals": _signals(identity_conflict)})


@pytest.mark.parametrize("identity_conflict, expected", [
    (True, "low_link_score"),   # belongs somewhere else: held
    (False, None),              # right PO, commercial disagreement: filed, flagged
])
def test_a_document_below_the_link_bar_is_decided_not_crashed(
        monkeypatch, identity_conflict, expected):
    from src.services import linking_engine as LE

    _stub_scoring(monkeypatch, LE, F=50.0, identity_conflict=identity_conflict)
    pk = LE._DOC["invoice"]["pk"]
    row = {pk: "INV-1", "po_id": "PO1", "confidence_score": 99}

    _po, link, reason = LE._evaluate(object(), "invoice", row)

    assert link["F"] == 50.0
    assert reason == expected


def test_the_review_band_is_measured_against_the_governed_link_bar(monkeypatch):
    from src.services import linking_engine as LE

    pk = LE._DOC["invoice"]["pk"]
    row = {pk: "INV-2", "po_id": "PO1", "confidence_score": 99}
    link = {"F": 70.0, "decision": "review", "signals": _signals(False)}
    monkeypatch.setattr(LE, "_rows", lambda cur, sql, params=None: [row])
    monkeypatch.setattr(LE, "_evaluate",
                        lambda cur, dt, r: ({"po_id": "PO1"}, link, "low_link_score"))

    class _Conn:
        def cursor(self):
            return object()

    items = LE._review_queue(_Conn(), ("invoice",), floor=65.0)

    assert [i["doc_pk"] for i in items] == ["INV-2"]
