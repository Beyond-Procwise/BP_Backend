"""No new ungated write endpoint (a ratchet, modelled on test_gateway_boundary).

Every route that changes state (POST / PUT / PATCH / DELETE) under ``src/api`` must consult the
gate -- ``endpoint_gate.require``, ``guardrail.authorize``, or one of the router-local role/permit
helpers -- inside its own handler. ``require_user`` alone only proves the caller is *someone* in the
pool; it does not decide whether they may do this (audit findings RT-1..RT-8).

``tests/endpoint_gate_baseline.txt`` lists every write handler that does NOT consult the gate today.
The list may only shrink:
  * a NEW write handler that does not consult the gate fails the first test (add the gate, or -- only
    for something genuinely public such as a webhook or login callback -- add the handler to PUBLIC);
  * a handler that has since been gated must be deleted from the baseline (second test), so the
    ratchet tightens.
Regenerate only when gating routes:  python tests/test_endpoint_gate_ratchet.py --write
"""
import ast
import os
import re

import pytest
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "tests", "endpoint_gate_baseline.txt")
WRITE_METHODS = {"post", "put", "patch", "delete"}

# A handler counts as gated only if it CALLS one of these (found in the syntax tree, so a mention in a
# comment, docstring or string never counts, and neither does merely reading the caller's role).
GATE_CALLS = {"endpoint_gate.require", "guardrail.authorize", "rbac.may"}
# Router-local helpers that refuse (raise) when the caller may not: sales._require_margin_permit,
# approvals._require_capability, agents._require_role, agent_policies._require, email_learning._gate.
GATE_HELPER = re.compile(r"^(_gate|_require|_require_\w+|require_role\w*)$")

# Deliberately public write endpoints (no signed-in user exists yet). Keep this tiny and justified.
PUBLIC: dict = {}

# Write handlers whose gate lives INSIDE the service they call, where the scanner cannot see it.
# Each entry names where the gate is, so nobody "gates" the handler again with a second action name
# and double-audits. A handler listed here must still exist and still look ungated to the scanner;
# the moment it gains a visible gate the entry is stale and must go.
GUARDED_IN_SERVICE: dict = {
    "src/api/routers/value_summary.py::POST router /value-summary/findings/{finding_id}/query-send::post_query_send":
        "services.value_query_service.send_query runs email_dispatch_guard.check_recipient_and_sensitivity "
        "and guardrail.authorize('email.send', 'communicate', principal) before anything is sent",
    "src/api/routers/reports.py::POST router /presentation-mode::presentation_set":
        "services.report_data.mode.activate/deactivate raise PermissionError (mapped to 403) for anyone "
        "who is not an Admin; the mode is per session and ends with the sign-in",
}

# Registration styles the scanner cannot see. None exists today; if one appears the scan must be taught it.
UNSEEN_REGISTRATION = re.compile(r"\.api_route\(|\.add_api_route\(|\.mount\(|\.add_websocket_route\(")


def _dotted(node):
    """'a.b.c' for a Name/Attribute chain, else ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _file_gate_names(tree):
    """How THIS file spells endpoint_gate.require: ``require as gate`` -> {'gate'}; ``import ... as eg`` -> {'eg.require'}."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("endpoint_gate"):
            for alias in node.names:
                if alias.name == "require":
                    names.add(alias.asname or "require")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("endpoint_gate") and alias.asname:
                    names.add(alias.asname + ".require")
    return names


def _calls_a_gate(func, file_names, module_funcs=None, _depth=0):
    """Does ``func`` consult the gate, directly or through ONE module-level helper it calls?

    Routers wrap the gate in small local helpers (``_edit_gate``, ``_decide``, ``_authorize``) so
    the action name lives in one place. Following one level of same-module helper keeps the scan
    honest about those without growing a list of helper names; it does not follow into other
    modules, so a gate inside a service still needs a GUARDED_IN_SERVICE entry with its evidence.
    """
    module_funcs = module_funcs or {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name:
            continue
        if name in GATE_CALLS or name in file_names or GATE_HELPER.match(name.rsplit(".", 1)[-1]):
            return True
        helper = module_funcs.get(name)
        if helper is not None and _depth == 0 and _calls_a_gate(helper, file_names, module_funcs, _depth + 1):
            return True
    return False


def _py_files():
    api = os.path.join(ROOT, "src", "api")
    for base, _dirs, files in os.walk(api):
        if "__pycache__" in base:
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(base, name)


def _handlers():
    """Yield (relative file, METHOD, 'router.var' + path literal, function name, gated?) for every write route."""
    for path in _py_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        file_names = _file_gate_names(tree)
        module_funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                    continue
                method = dec.func.attr.lower()
                if method not in WRITE_METHODS:
                    continue
                route = ""
                if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                    route = dec.args[0].value
                # The router variable is part of the key: `router` and `orchestrator_router` can serve the same
                # path and function name with different prefixes.
                owner = _dotted(dec.func.value) or "?"
                yield rel, method.upper(), f"{owner} {route}", node.name, _calls_a_gate(node, file_names, module_funcs)


def _ungated():
    return sorted(f"{rel}::{m} {route}::{fn}" for rel, m, route, fn, gated in _handlers() if not gated)


def _baseline():
    with open(BASELINE, encoding="utf-8") as fh:
        return sorted(line.strip() for line in fh if line.strip() and not line.startswith("#"))


def test_no_new_write_endpoint_skips_the_gate():
    new = sorted(set(_ungated()) - set(_baseline()) - set(PUBLIC) - set(GUARDED_IN_SERVICE))
    assert not new, (
        "write endpoints that never consult the gate (endpoint_gate.require / guardrail.authorize / a router "
        f"role helper): {new}. Gate them; require_user only proves identity."
    )


def test_baseline_only_lists_handlers_that_are_still_ungated():
    stale = sorted(set(_baseline()) - set(_ungated()))
    assert not stale, f"these handlers are now gated or gone; remove them from the baseline: {stale}"


def test_no_route_is_registered_in_a_way_the_scan_cannot_see():
    seen = []
    for path in _py_files():
        with open(path, encoding="utf-8", errors="ignore") as fh:
            if UNSEEN_REGISTRATION.search(fh.read()):
                seen.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    assert not seen, f"routes registered without a decorator the ratchet reads: {seen}. Teach tests/test_endpoint_gate_ratchet.py about them."


def test_the_scan_actually_finds_routes():
    # A scanner that silently finds nothing would pass both tests above forever.
    handlers = list(_handlers())
    assert len(handlers) > 50
    assert any(gated for *_rest, gated in handlers), "no handler recognised as gated: GATE pattern is stale"


if __name__ == "__main__":
    if "--write" in sys.argv:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            fh.write("# Write handlers under src/api that never consult the gate. May only shrink. See tests/test_endpoint_gate_ratchet.py\n")
            fh.write("\n".join(_ungated()) + "\n")
        print(f"wrote {len(_ungated())} entries")


@pytest.mark.parametrize("handler", sorted(GUARDED_IN_SERVICE))
def test_a_handler_guarded_inside_its_service_still_exists_and_still_looks_ungated(handler):
    """The exemption may not rot: the handler must exist, and must NOT have grown a visible gate
    (then it is gated twice, under two names, and the entry here is stale)."""
    ungated = set(_ungated())
    assert handler in ungated, f"{handler} is gone or now calls the gate itself: remove it from GUARDED_IN_SERVICE"
    assert handler not in set(_baseline()), f"{handler} is in the baseline AND exempted: pick one"


def test_the_scan_follows_one_level_of_local_helper_and_no_further():
    """A handler that gates through a same-module helper counts as gated; a helper that only calls
    another helper does not (the scan is deliberately shallow, so a gate cannot hide two hops away)."""
    src = (
        "from api.endpoint_gate import require as gate\n"
        "def _edit_gate(p):\n    gate('report.generate', p)\n"
        "def _twice(p):\n    _edit_gate(p)\n"
        "def direct(p):\n    _edit_gate(p)\n"
        "def indirect(p):\n    _twice(p)\n"
        "def naked(p):\n    return 1\n"
    )
    tree = ast.parse(src)
    names = _file_gate_names(tree)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert _calls_a_gate(funcs["direct"], names, funcs) is True
    assert _calls_a_gate(funcs["indirect"], names, funcs) is False
    assert _calls_a_gate(funcs["naked"], names, funcs) is False
