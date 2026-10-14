"""Two test files that import under the same module name make one of them vanish.

pytest's default import mode (`prepend`) derives a test file's module name by
walking UP from the file while each directory holds an `__init__.py`, then taking
the dotted path from there. A directory with no `__init__.py` is the stopping
point, so `tests/services/rga/test_models.py` and
`tests/services/analytics/test_models.py` both import as plain ``test_models``.
Collection then aborts:

    import file mismatch:
    imported module 'test_models' has this __file__ attribute:
      tests/services/analytics/test_models.py
    which is not the same as the test file we want to collect:
      tests/services/rga/test_models.py

That aborts the WHOLE directory sweep, not just the clashing file, which is how
`tests/services/rga/test_models.py` went three weeks without executing in any
`pytest tests/services/` run — the two files have coexisted since 2026-09-14 and
nothing said so. The error is easy to read past: a sweep still reports thousands
of passes beside its "1 error".

Seventeen basenames are duplicated under `tests/`. Most are harmless because the
directories holding them are packages, which is what makes their module names
distinct. This test is the thing that keeps that true as files are added, and it
names the fix in its own failure message.
"""
from __future__ import annotations

import collections
import pathlib

TESTS_ROOT = pathlib.Path(__file__).resolve().parent


def _module_name(path: pathlib.Path) -> str:
    """The name pytest's prepend import mode will give this file.

    Walk up while the directory is a package; the dotted path from there down is
    the module name. This mirrors `_pytest.pathlib.resolve_pkg_root_and_module_name`.
    """
    parts = [path.stem]
    parent = path.parent
    while (parent / "__init__.py").is_file():
        parts.append(parent.name)
        parent = parent.parent
    return ".".join(reversed(parts))


def test_no_two_test_files_import_under_the_same_module_name():
    by_module: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for path in TESTS_ROOT.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        by_module[_module_name(path)].append(path)

    clashes = {name: paths for name, paths in by_module.items() if len(paths) > 1}
    if not clashes:
        return

    lines = []
    for name, paths in sorted(clashes.items()):
        rel = [str(p.relative_to(TESTS_ROOT.parent)) for p in sorted(paths)]
        lines.append(f"  {name!r} is claimed by {len(paths)} files:")
        lines += [f"      {r}" for r in rel]
        # The fix, named rather than left to the reader.
        dirs = sorted({str(p.parent.relative_to(TESTS_ROOT.parent)) for p in paths})
        lines.append(
            f"    fix: add an empty __init__.py to {dirs}, or rename one file. "
            f"Until then collection ABORTS and every test under the shared "
            f"directory is skipped silently."
        )
    raise AssertionError(
        "test files collide on module name, so pytest cannot collect them "
        "together:\n" + "\n".join(lines)
    )


def _first_party_module_targets(path: pathlib.Path):
    """Every `src.…` module a test file imports, as a dotted name.

    Deliberately a TEXT scan and a FILESYSTEM check, not an import: importing
    would also fail on optional third-party packages that are legitimately
    absent here (paddle, dateparser), and those are not this test's business.
    Only first-party `src.` paths are checked, because only those can be deleted
    by someone working in this repo.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "src":
                yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "src":
                    yield alias.name


def test_no_test_imports_a_first_party_module_that_was_deleted():
    """A test importing a module nobody deleted the test for can never pass.

    It dies in one of two ways, and the quieter one is worse. A module-level
    import is a COLLECTION ERROR: pytest reports one line, carries on, and the
    whole file's coverage disappears while the suite still looks healthy. A
    function-level import is an ordinary failing test, which at least shows up
    in the count.

    Both shapes were present and both had been for five months: five files under
    tests/extraction_v2 and one assertion in tests/structural_extractor, all
    naming modules removed in 114e5ca (2026-05-09, "remove legacy extraction
    stack, hard-wire dispatch to v3").

    Checks the filesystem rather than importing, so a missing optional
    dependency cannot make this red.
    """
    repo_root = TESTS_ROOT.parent
    missing: dict[str, list[str]] = {}
    for path in TESTS_ROOT.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        for dotted in _first_party_module_targets(path):
            rel = pathlib.Path(*dotted.split("."))
            if (repo_root / rel).with_suffix(".py").is_file():
                continue
            if (repo_root / rel / "__init__.py").is_file():
                continue
            missing.setdefault(str(path.relative_to(repo_root)), []).append(dotted)

    assert not missing, (
        "these test files import first-party modules that no longer exist, so "
        "pytest cannot collect them and their coverage is silently gone:\n"
        + "\n".join(
            f"  {f}\n      {', '.join(sorted(set(mods)))}"
            for f, mods in sorted(missing.items())
        )
        + "\n  fix: delete the test if its subject was deleted, or re-point it "
        "at the module that replaced it."
    )
