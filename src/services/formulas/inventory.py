"""``model_inventory()`` --- the report the registry can generate about itself.

A hand-maintained inventory is stale the day after it is written. This one is
derived: every field comes from the registry, and "dependents" comes from
scanning the tree for call sites by name, so a formula nobody calls any more
shows up as orphaned rather than quietly persisting.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .registry import REGISTRY, VALIDATED_AT, FormulaSpec

#: Root of the source tree scanned for call sites.
_SRC = Path(__file__).resolve().parents[2]
_REPO = _SRC.parent
_SCAN_DIRS = ("src", "scripts", "tests")
_SKIP_PARTS = {"__pycache__", "venv", ".venv", "node_modules", ".git", "worktrees"}


def _python_files() -> List[Path]:
    files: List[Path] = []
    for d in _SCAN_DIRS:
        root = _REPO / d
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if _SKIP_PARTS & set(p.parts):
                continue
            files.append(p)
    return files


def _dependents(names: Sequence[str]) -> Dict[str, List[str]]:
    """Files that name each formula, excluding the module that defines it.

    A literal-string scan rather than an import graph: ``evaluate`` takes the
    name as data, so the name in quotes *is* the dependency edge.
    """
    found: Dict[str, set] = {n: set() for n in names}
    patterns = {n: re.compile(re.escape(f'"{n}"') + "|" + re.escape(f"'{n}'")) for n in names}
    definition_files = {
        spec.name: spec.source_module.replace(".", "/") + ".py"
        for spec in REGISTRY.values()
    }
    for path in _python_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover
            continue
        rel = str(path.relative_to(_REPO))
        for name, pat in patterns.items():
            if pat.search(text) and not rel.endswith(definition_files.get(name, "\0")):
                found[name].add(rel)
    return {n: sorted(v) for n, v in found.items()}


def model_inventory(include_dependents: bool = True) -> List[Dict[str, Any]]:
    """One row per registered formula, derived entirely from the registry."""
    specs: List[FormulaSpec] = [REGISTRY[n] for n in sorted(REGISTRY)]
    deps = _dependents([s.name for s in specs]) if include_dependents else {}
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        validated: Optional[datetime] = VALIDATED_AT.get(spec.name)
        rows.append(
            {
                "name": spec.name,
                "version": spec.version,
                "version_hash": spec.version_hash,
                "kind": spec.kind.value,
                "owner": spec.owner,
                "purpose": spec.purpose,
                "contract": spec.contract.to_dict(),
                "gpss_version": spec.gpss_version,
                "effective_from": spec.effective_from.isoformat(),
                "last_validated": validated.isoformat() if validated else None,
                "golden_vectors": len(spec.golden),
                "source": f"{spec.source_module}:{spec.source_line}",
                "replaces": list(spec.replaces),
                "notes": spec.notes,
                "dependents": deps.get(spec.name, []),
            }
        )
    return rows


def _fmt_contract(contract: Dict[str, Any]) -> str:
    ins = "<br>".join(
        f"`{t['name']}` · {t['unit']} · {t['range']}"
        + ("" if t["required"] else " · *optional*")
        for t in contract["inputs"]
    )
    out = contract["output"]
    return f"{ins}<br>**→** `{out['type']}` · {out['unit']}"


def render_markdown(rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """The committed ``docs/model-inventory.md`` body."""
    rows = rows if rows is not None else model_inventory()
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [
        "# Model Inventory",
        "",
        f"Generated from the formula registry by `model_inventory()` — {generated}.",
        "Do not hand-edit: regenerate with `python -m src.services.formulas.inventory`.",
        "",
        f"**{len(rows)} registered formulas.**",
        "",
    ]

    orphans = [r["name"] for r in rows if not r["dependents"]]
    if orphans:
        lines += [
            f"**{len(orphans)} with no call site outside their own module** — "
            "either not yet migrated onto `evaluate`, or dead: "
            + ", ".join(f"`{o}`" for o in orphans),
            "",
        ]

    lines += [
        "| Formula | Ver | Owner | Purpose | GPSS | Last validated | Vectors | Dependents |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        validated = (r["last_validated"] or "never")[:19].replace("T", " ")
        deps = f"{len(r['dependents'])}" if r["dependents"] else "—"
        lines.append(
            f"| `{r['name']}` | {r['version']}+{r['version_hash'][:8]} | {r['owner']} "
            f"| {r['purpose']} | {r['gpss_version'] or '—'} | {validated} "
            f"| {r['golden_vectors']} | {deps} |"
        )

    lines += ["", "---", "", "## Contracts", ""]
    for r in rows:
        lines += [
            f"### `{r['name']}`",
            "",
            f"*{r['purpose']}* — {r['kind']} formula, owner **{r['owner']}**, "
            f"effective from {r['effective_from']}, source `{r['source']}`.",
            "",
        ]
        if r["replaces"]:
            lines += [
                "**Replaces:** " + ", ".join(f"`{x}`" for x in r["replaces"]),
                "",
            ]
        if r["notes"]:
            lines += [r["notes"], ""]
        lines += ["| Input | Unit | Range | Required |", "|---|---|---|---|"]
        for t in r["contract"]["inputs"]:
            lines.append(
                f"| `{t['name']}` | {t['unit']} | {t['range']} "
                f"| {'yes' if t['required'] else 'no'} |"
            )
        out = r["contract"]["output"]
        lines += [
            "",
            f"**Output:** `{out['type']}` in {out['unit']} — {out['description']}",
            "",
            "**Dependents:** "
            + (", ".join(f"`{d}`" for d in r["dependents"]) if r["dependents"] else "*none*"),
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - CLI
    # Parts of this codebase import as `services.x` and parts as
    # `src.services.x`; both roots have to be importable before the definition
    # modules will load.
    import sys

    for root in (str(_REPO), str(_REPO / "src")):
        if root not in sys.path:
            sys.path.insert(0, root)

    from . import definitions  # noqa: F401  (registers every formula)

    print(render_markdown())


if __name__ == "__main__":  # pragma: no cover
    main()

__all__ = ["model_inventory", "render_markdown"]
