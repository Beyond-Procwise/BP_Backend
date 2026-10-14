"""What a freshly extracted module still needs from outside itself.

Guessing this list is how an extraction goes red: models.py was given a
hand-written header that omitted `uuid` and `DEFAULT_NEGOTIATION_SUBJECT`, and
sixteen tests failed on a NameError far from the edit. Derive it instead.

Reports every name the module loads but never defines, imports, or receives as
an argument. Builtins and names bound by nested functions are excluded.

Usage:  missing_imports.py <module.py>
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    path = Path(argv[0])
    tree = ast.parse(path.read_text())

    top = {
        n.name for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for n in tree.body:
        if isinstance(n, ast.Assign):
            top.update(t.id for t in n.targets if isinstance(t, ast.Name))

    imported: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported.update((a.asname or a.name).split(".")[0] for a in n.names)

    # A function defined inside another is bound where it is defined.
    nested = {
        f.name for node in tree.body for f in ast.walk(node)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    free: set[str] = set()
    for node in tree.body:
        local = {a.arg for a in ast.walk(node) if isinstance(a, ast.arg)}
        local |= {
            x.id for x in ast.walk(node)
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)
        }
        # `except X as exc` binds exc without a Name node.
        local |= {
            h.name for h in ast.walk(node)
            if isinstance(h, ast.ExceptHandler) and h.name
        }
        local |= nested
        for x in ast.walk(node):
            if not (isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)):
                continue
            if x.id in local or x.id in top or x.id in imported:
                continue
            if hasattr(builtins, x.id):
                continue
            free.add(x.id)

    if free:
        print(f"{path.name} still needs: {', '.join(sorted(free))}")
        return 1
    print(f"{path.name}: nothing missing")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
