"""One owner for a quote's version rule: src/services/version_collapse.py (2026-10-08).

Analyses that kept their own copy (supplier_ranking_agent's r"\\(\\s*V(\\d+)", board papers
ordering "latest" by date) disagreed with each other on which version is current. Extraction
reads the marker from DOCUMENT TEXT (context_layer), which is a different job and allowed.
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
ALLOWED = {"services/version_collapse.py", "services/extraction/context_layer.py"}
# 'V(\d+)' as written in a raw regex or in a Python string holding SQL ('V(\\d+)').
PRIVATE_RULE = re.compile(r"[Vv]\((?:\\\\|\\)d\+\)")


def offenders(root=SRC):
    out = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if rel in ALLOWED:
            continue
        if PRIVATE_RULE.search(p.read_text(encoding="utf-8", errors="ignore")):
            out.append(rel)
    return sorted(out)


def test_no_module_spells_its_own_quote_version_rule():
    assert offenders() == []
