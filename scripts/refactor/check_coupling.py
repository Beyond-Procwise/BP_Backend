"""Which pure helpers can actually leave, and which only look as though they can.

A method with no `self` cannot touch instance state, which is what makes it safe
to move. That is necessary and not sufficient: it may still read a constant or
call a function defined at module level in the file it lives in. Moving one of
those means the new module imports the old one, which already imports the new
one, and the circular import surfaces as an ImportError a long way from the
edit that caused it.

This separates the two. Run it before choosing a cluster to extract.

Usage:  check_coupling.py <source.py> <ClassName>
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    src_path, class_name = argv
    source = Path(src_path).read_text()
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )

    # Anything this module defines itself: importing it back would be circular.
    module_level: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_level.add(n.name)
        elif isinstance(n, ast.Assign):
            module_level.update(
                t.id for t in n.targets if isinstance(t, ast.Name)
            )
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            module_level.add(n.target.id)

    # Names it imports are fine — a new module can import them the same way.
    imported: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported.update((a.asname or a.name).split(".")[0] for a in n.names)

    clean: dict[str, int] = {}
    blocked: dict[str, tuple[int, list[str]]] = {}
    for m in cls.body:
        if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(isinstance(n, ast.Name) and n.id in {"self", "cls"} for n in ast.walk(m)):
            continue
        local = {a.arg for a in ast.walk(m) if isinstance(a, ast.arg)}
        local |= {
            n.id for n in ast.walk(m)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        back = sorted({
            n.id for n in ast.walk(m)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id not in local and n.id not in imported
            and not hasattr(builtins, n.id) and n.id in module_level
        })
        size = m.end_lineno - m.lineno + 1
        if back:
            blocked[m.name] = (size, back)
        else:
            clean[m.name] = size

    total = len(clean) + len(blocked)
    print(f"pure helpers on {class_name}: {total}")
    print(f"  movable now:                   {len(clean):3}  ({sum(clean.values())} lines)")
    print(f"  blocked by module-level names: {len(blocked):3}  ({sum(v[0] for v in blocked.values())} lines)")
    print("\nMOVABLE")
    for k, v in sorted(clean.items(), key=lambda kv: -kv[1]):
        print(f"   {k:36} {v:4}")
    print("\nBLOCKED — would import back from this module")
    for k, (size, back) in sorted(blocked.items(), key=lambda kv: -kv[1][0]):
        print(f"   {k:36} {size:4}  needs {', '.join(back[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
