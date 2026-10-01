"""Display labels are not aliases.

Build spec principle 7 and section 3.6. proc.bp_translation holds UI strings per
locale. If code that decides what a document or concept *is* ever read it, a
document would resolve differently depending on the viewer's language, and the
failure would be invisible in English.

Scope note: i18n/registry.py's ``_exact`` matches LANGUAGE NAMES for the
language picker. That is not concept matching and is legitimate; nothing here
concerns it.

HOW THE CHECKED SET IS DERIVED (nothing below is a hand-picked list of the files
to check):

* The readers of the table are discovered: every module under src/ whose source
  mentions the table, plus the whole i18n package.
* The matching modules are discovered: the concepts package, the type resolver,
  and every module that imports either of them, found by parsing imports. A new
  module that starts resolving types onto concepts is therefore picked up by
  importing the vocabulary, with no edit to this file.
* From those roots the test follows the import graph (including function-level
  imports) and fails if anything reachable is a reader of the table.

WHAT THIS CAN CATCH: a literal mention of the table anywhere outside the allowed
files; a matching module importing, directly or through any chain of helpers, a
module that reads the table or lives in the i18n package.

WHAT THIS CANNOT CATCH: a table name assembled at run time ("bp_" + "translation"),
code reached through importlib / __import__ / getattr on a string, a read that
goes through a database view or a differently named alias of the table, and a
matching module that receives translated text as an argument from a caller.
Those need a runtime guard (e.g. a DB role without SELECT on the table), which
this source-level test is not.

No database, no network, no model: it only parses files.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Files that may mention the translation tables. store.py is the only reader;
#: audit.py and the router mention the table in prose / docstrings. Any other
#: file is a violation. If a genuine new i18n module needs this, add it here with
#: a reason; never widen the regex.
_ALLOWED = {
    "src/services/i18n/store.py": "the translation store itself",
    "src/services/i18n/audit.py": "docstring mention only",
    "src/api/routers/i18n.py": "docstring mention only",
}

_TABLES = re.compile(r"bp_translation(?:_language_status)?\b")
_I18N_PACKAGE = "src.services.i18n"

#: Modules whose presence in the matching set is a floor: if the derivation ever
#: stops finding them, the guard has silently gone blind.
_FLOOR = {
    "src.services.concepts.vocabulary",
    "src.services.concepts.routing",
    "src.services.extraction.type_resolver",
    "src.services.extraction.dispatch",
}


def _modules(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in (root / "src").rglob("*.py"):
        parts = list(p.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = p
    return out


def _import_graph(mods: dict[str, Path]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for name, path in mods.items():
        tree = ast.parse(path.read_text(errors="ignore"))
        pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
        found: set[str] = set()
        for node in ast.walk(tree):  # walk, so imports inside functions count
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    up = pkg.split(".")
                    up = up[: len(up) - (node.level - 1)]
                    base = ".".join(up + ([base] if base else []))
                found.add(base)
                found.update(f"{base}.{a.name}" for a in node.names)
        graph[name] = {m for m in found if m in mods}
    return graph


def _is_i18n(mod: str) -> bool:
    return mod == _I18N_PACKAGE or mod.startswith(_I18N_PACKAGE + ".")


def table_references(root: Path) -> dict[str, int]:
    """Every file under src/ that mentions the table, outside the allowed set."""
    hits: dict[str, int] = {}
    for p in sorted((root / "src").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel in _ALLOWED:
            continue
        n = len(_TABLES.findall(p.read_text(errors="ignore")))
        if n:
            hits[rel] = n
    return hits


def matching_roots(mods: dict[str, Path], graph: dict[str, set[str]]) -> set[str]:
    core = {m for m in mods if m.startswith("src.services.concepts")}
    core |= {m for m in mods if m.endswith(".type_resolver")}
    return core | {m for m, deps in graph.items() if deps & core}


def reader_modules(root: Path, mods: dict[str, Path]) -> set[str]:
    readers = {m for m in mods if _is_i18n(m)}
    for p in (root / "src").rglob("*.py"):
        if _TABLES.search(p.read_text(errors="ignore")):
            readers |= {m for m, q in mods.items() if q == p}
    return readers


def matching_paths_reaching_readers(root: Path) -> dict[str, list[str]]:
    """root module -> an import chain ending at a reader of the table."""
    mods = _modules(root)
    graph = _import_graph(mods)
    readers = reader_modules(root, mods)
    bad: dict[str, list[str]] = {}
    for start in sorted(matching_roots(mods, graph)):
        prev = {start: None}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            if cur in readers:
                chain, n = [], cur
                while n is not None:
                    chain.append(n)
                    n = prev[n]
                bad[start] = chain[::-1]
                break
            for nxt in sorted(graph[cur]):
                if nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
    return bad


# --------------------------------------------------------------------- the guard

def test_only_the_i18n_store_mentions_the_translation_tables():
    offenders = table_references(ROOT)
    assert not offenders, (
        "files outside the allowed set reference the translation store: "
        f"{offenders}. Display labels are not aliases (build spec principle 7)."
    )


def test_allowed_files_exist_so_the_allowlist_cannot_go_stale():
    missing = [rel for rel in _ALLOWED if not (ROOT / rel).exists()]
    assert not missing, f"allowlist names files that no longer exist: {missing}"


def test_no_matching_module_can_reach_the_translation_store():
    bad = matching_paths_reaching_readers(ROOT)
    assert not bad, (
        "matching code reaches display-label machinery (import chain shown): "
        + "; ".join(" -> ".join(c) for c in bad.values())
    )


def test_the_derivation_still_finds_the_matching_modules():
    """A guard over a set that silently became empty checks nothing."""
    mods = _modules(ROOT)
    roots = matching_roots(mods, _import_graph(mods))
    assert _FLOOR <= roots, f"derivation lost: {sorted(_FLOOR - roots)}"
    readers = reader_modules(ROOT, mods)
    assert "src.services.i18n.store" in readers, "the real reader was not found"


def test_the_vocabulary_migrations_carry_no_label_column():
    """If a label column appeared on bp_document_type, matching on it would be
    one join away. The vocabulary holds codes, definitions and aliases only."""
    files = sorted((ROOT / "deploy" / "sql").glob("*concept_vocabulary*.sql"))
    assert files, "no vocabulary migrations found; the glob has gone blind"
    for f in files:
        sql = re.sub(r"--[^\n]*", "", f.read_text()).lower()
        for banned in ("locale", "label", "display_name"):
            assert banned not in sql, (
                f"{f.name} declares {banned!r}; display labels belong in "
                "proc.bp_translation and are never matched on"
            )


# ------------------------------------------- the guard itself is shown to work

def _fake_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return tmp_path


_BASE = {
    "src/__init__.py": "",
    "src/services/__init__.py": "",
    "src/services/concepts/__init__.py": "",
    "src/services/concepts/vocabulary.py": "X = 1\n",
    "src/services/i18n/__init__.py": "",
    "src/services/i18n/store.py": "Q = 'SELECT * FROM proc.bp_translation'\n",
}


def test_a_clean_tree_passes(tmp_path):
    root = _fake_tree(tmp_path, dict(_BASE))
    assert table_references(root) == {}
    assert matching_paths_reaching_readers(root) == {}


def test_a_new_matching_module_with_a_direct_read_is_caught(tmp_path):
    files = dict(_BASE)
    files["src/services/newmatcher.py"] = (
        "from src.services.concepts import vocabulary\n"
        "SQL = 'select * from proc.bp_translation'\n"
    )
    root = _fake_tree(tmp_path, files)
    assert "src/services/newmatcher.py" in table_references(root)
    assert "src.services.newmatcher" in matching_paths_reaching_readers(root)


def test_a_read_hidden_behind_a_helper_chain_is_caught(tmp_path):
    files = dict(_BASE)
    files["src/services/newmatcher.py"] = (
        "def f():\n    from src.services.helper_a import g\n    return g()\n"
        "from src.services.concepts.vocabulary import X\n"
    )
    files["src/services/helper_a.py"] = "from src.services.helper_b import g\n"
    files["src/services/helper_b.py"] = "from src.services.i18n.store import Q\ndef g(): return Q\n"
    root = _fake_tree(tmp_path, files)
    # the helpers never name the table, so only the import chain can find this
    assert "src/services/helper_b.py" not in table_references(root)
    chain = matching_paths_reaching_readers(root)["src.services.newmatcher"]
    assert chain[0] == "src.services.newmatcher"
    assert chain[-1] == "src.services.i18n.store"


def test_a_relative_import_of_the_i18n_package_is_caught(tmp_path):
    files = dict(_BASE)
    files["src/services/concepts/routing.py"] = "from ..i18n import store\n"
    root = _fake_tree(tmp_path, files)
    assert "src.services.concepts.routing" in matching_paths_reaching_readers(root)


def test_the_known_blind_spot_is_really_a_blind_spot(tmp_path):
    """Documented, not hidden: an assembled table name is NOT caught."""
    files = dict(_BASE)
    files["src/services/newmatcher.py"] = (
        "from src.services.concepts import vocabulary\n"
        "T = 'bp_' + 'translation'\n"
    )
    root = _fake_tree(tmp_path, files)
    assert table_references(root) == {}
