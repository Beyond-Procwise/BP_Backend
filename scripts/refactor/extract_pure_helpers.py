"""Move provably-pure methods off a god class into a module, mechanically.

A method that never mentions `self` cannot call a sibling method, read an
attribute, or mutate instance state. It is a leaf function wearing a method's
clothes. Those are the safe ones to move, and this moves them by rewriting the
source text rather than by hand, because 1,600 lines of hand-transcription is
how a "pure refactor" quietly changes behaviour.

What it leaves behind on the class is a one-line delegator, not a staticmethod.
That matters: the test suite calls several of these unbound, passing None for
self (`NegotiationAgent._extract_batch_inputs(None, payload)`), and a
staticmethod would silently read that None as the first real argument.

Usage:  extract_pure_helpers.py <source.py> <ClassName> <target_module.py> <name> [name ...]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


def _decorator_names(node: ast.AST) -> list[str]:
    return [ast.unparse(d) for d in getattr(node, "decorator_list", [])]


def collect(source: str, class_name: str, wanted: set[str]):
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    found = {}
    for m in cls.body:
        if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if m.name not in wanted:
            continue
        if any(isinstance(n, ast.Name) and n.id in {"self", "cls"} for n in ast.walk(m)):
            raise SystemExit(f"{m.name} references self/cls — not pure, refusing to move")
        found[m.name] = m
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"not found on {class_name}: {sorted(missing)}")
    return found


def method_source(lines: list[str], node) -> str:
    """The verbatim block, decorators included, dedented by one level."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    block = lines[start:node.end_lineno]
    out = []
    for ln in block:
        out.append(ln[4:] if ln.startswith("    ") else ln)
    return "\n".join(out)


def rewrite(block: str, name: str, alias: str) -> tuple[str, str, str]:
    """Return (public_name, module_function_source, delegator_source)."""
    public = name.lstrip("_")
    # Drop @staticmethod: at module level it is meaningless.
    body = "\n".join(
        l for l in block.splitlines() if l.strip() not in {"@staticmethod", "@classmethod"}
    )
    sig = re.search(r"^(async )?def\s+" + re.escape(name) + r"\s*\((.*?)\)\s*(->.*?)?:",
                    body, re.S | re.M)
    if not sig:
        raise SystemExit(f"could not read the signature of {name}")
    params_raw = sig.group(2)
    # Strip a leading self/cls if the signature carries one (it is unused by
    # definition here, but several of these were written as instance methods).
    params = params_raw
    lead = re.match(r"\s*(self|cls)\s*(,\s*)?", params)
    if lead:
        params = params[lead.end():]
    body = body[:sig.start(2)] + params + body[sig.end(2):]
    body = re.sub(r"^(async )?def\s+" + re.escape(name), r"\1def " + public, body, count=1,
                  flags=re.M)

    # Call-through arguments: names only, defaults and annotations dropped.
    call_args = []
    if params.strip():
        tree = ast.parse(f"def _f({params}): pass")
        fn = tree.body[0]
        for a in fn.args.posonlyargs + fn.args.args:
            call_args.append(a.arg)
        if fn.args.vararg:
            call_args.append("*" + fn.args.vararg.arg)
        for a in fn.args.kwonlyargs:
            call_args.append(f"{a.arg}={a.arg}")
        if fn.args.kwarg:
            call_args.append("**" + fn.args.kwarg.arg)
    # Dropping the leading underscore can collide with a name the method already
    # called. _normalise_lever_category was a one-line delegate to an imported
    # normalise_lever_category; renaming it turned that into infinite recursion,
    # and nothing about the diff looked wrong. Refuse instead of shipping it.
    for node in ast.walk(ast.parse(body)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == public):
            raise SystemExit(
                f"{name} calls {public!r}, so renaming it would make it call itself. "
                "It is probably already a delegate to an imported function — leave it "
                "on the class."
            )
    joined = ", ".join(call_args)
    is_async = bool(sig.group(1))
    aw = "await " if is_async else ""
    pre = "async " if is_async else ""
    delegator = (
        f"    {pre}def {name}(self, {params_raw[lead.end():] if lead else params_raw}):\n"
        if lead else f"    {pre}def {name}(self, {params_raw}):\n"
    )
    delegator += f"        return {aw}{alias}.{public}({joined})\n"
    return public, body, delegator


def main(argv: list[str]) -> int:
    src_path, class_name, target, *names = argv
    alias = "_" + Path(target).stem
    source = Path(src_path).read_text()
    lines = source.splitlines()
    found = collect(source, class_name, set(names))

    # Replace from the bottom up so earlier line numbers stay valid.
    pieces, exports = [], []
    for name in sorted(found, key=lambda n: found[n].lineno, reverse=True):
        node = found[name]
        block = method_source(lines, node)
        public, fn_src, delegator = rewrite(block, name, alias)
        pieces.append(fn_src)
        exports.append(public)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        lines[start:node.end_lineno] = delegator.rstrip("\n").split("\n")

    Path(src_path).write_text("\n".join(lines) + "\n")
    Path(target).write_text(
        "\n\n\n".join(reversed(pieces)) + "\n"
    )
    print(f"moved {len(found)} helpers -> {target}")
    print("exports:", ", ".join(sorted(exports)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
