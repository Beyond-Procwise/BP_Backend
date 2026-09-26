"""Which UI components have a backend behind them, and which only look as if they do.

The companion check (check_ui_backend_coverage.py) asks whether every endpoint the
UI calls exists. It cannot see the more expensive failure: a component that calls
nothing at all, because it was wired to a literal array instead. That component
renders, scrolls, filters and sorts, and every endpoint it uses -- none -- is
present and correct.

So this walks the other way. For each component it records whether the file (or
the module's own data layer) reaches a backend, whether it carries fixture or
sample data, and how many controls it offers. A component with controls, fixtures
and no reachable backend is the thing worth looking at.

It reports, it does not judge: a presentational component legitimately has no
backend, and its module's data hook is the one making the call. The module-level
rollup is there for exactly that reason.

Usage:  check_ui_component_backing.py <ui_src_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SKIP = (".test.", "__tests__", ".stories.", ".contract.")

# Reaching a backend: axios/http/fetch, or one of the SpendIQ bridges.
CALLS = re.compile(
    r"""\b(?:axios|http|api|client)\s*\.\s*(?:get|post|put|patch|delete)\s*\("""
    # A bridge counts wherever it is named: it is often aliased or captured
    # before use, not called on the spot.
    r"""|__SPENDIQ_API(?:_AI)?(?:_[A-Z]+)?__"""
    # fetch(`${API}/workflows/ask`) -- the path starts with an interpolation,
    # so requiring a literal "/" or "http" right after the quote missed the
    # whole Chat module.
    r"""|\bfetch\s*\(\s*[`'"]?(?:\$\{|http|/)"""
    # Cognito is a backend. Sign-in, reset and the rest of the auth screens
    # reach it through Amplify rather than over axios, and reporting them as
    # unbacked is just wrong.
    # A client built on the spot: (await http()).get(url('/languages')). The
    # token before .get is a ")", so matching on a named client alone missed
    # the whole i18n adapter and, through it, every screen that reads from it.
    r"""|\)\s*\.\s*(?:get|post|put|patch|delete)\s*\("""
    r"""|from\s+['"]aws-amplify(?:/\w+)?['"]"""
    r"""|\b(?:signIn|signOut|resetPassword|confirmResetPassword|fetchAuthSession|confirmSignIn)\s*\(""",
    re.I | re.X,
)

IMPORTS = re.compile(r"""from\s+['"](\.[^'"]*)['"]""")

# Data that ships with the code rather than arriving from one.
FIXTURES = re.compile(
    r"""from\s+['"][^'"]*fixtures?['"]"""
    r"""|\b(?:SAMPLE|DEMO|PREVIEW|FALLBACK|MOCK|SEED|PLACEHOLDER)_[A-Z_]+\s*="""
    r"""|\bconst\s+\w*(?:Sample|Demo|Preview|Mock|Fallback)\w*\s*=\s*\[""",
    re.X,
)

CONTROLS = re.compile(r"onClick\s*=|onSubmit\s*=|onChange\s*=|<button\b", re.I)
IS_COMPONENT = re.compile(r"export\s+(?:default\s+)?function\s+[A-Z]|=>\s*\(?\s*<|return\s*\(?\s*<")


def main(argv: list[str]) -> int:
    ui_src = Path(argv[0])
    files = [
        f for f in sorted(ui_src.rglob("*.js*"))
        if not any(x in str(f) for x in SKIP)
    ]

    rows = []
    text_of: dict[str, str] = {}
    for f in files:
        text = f.read_text(errors="ignore")
        rel = f.relative_to(ui_src)
        text_of[str(rel)] = text
        parts = rel.parts
        module = parts[1] if parts[0] == "modules" and len(parts) > 1 else parts[0]
        rows.append({
            "file": str(rel),
            "module": module,
            "path": f,
            "calls": len(CALLS.findall(text)),
            "fixtures": len(FIXTURES.findall(text)),
            "controls": len(CONTROLS.findall(text)),
            "component": bool(IS_COMPONENT.search(text)),
        })

    # A component that hands the work to a service it imports is backed by it.
    # Without following imports, every screen whose fetching lives in a sibling
    # hook or in lib/ is reported as having no backend, which is the opposite
    # of true and would bury the one module that really has none.
    direct = {r["file"] for r in rows if r["calls"]}
    def resolve(src: Path, spec: str) -> str | None:
        base = (src.parent / spec).resolve()
        for cand in (base, base.with_suffix(".js"), base.with_suffix(".jsx"),
                     base / "index.js", base / "index.jsx"):
            try:
                rel = cand.relative_to(ui_src.resolve())
            except ValueError:
                continue
            if str(rel) in text_of:
                return str(rel)
        return None

    edges = {r["file"]: [e for e in
                         (resolve(r["path"], m.group(1)) for m in IMPORTS.finditer(text_of[r["file"]]))
                         if e] for r in rows}
    backed, changed = set(direct), True
    while changed:
        changed = False
        for f, deps in edges.items():
            if f not in backed and any(d in backed for d in deps):
                backed.add(f)
                changed = True
    for r in rows:
        r["backed"] = r["file"] in backed

    # Roll up: a presentational component's data usually arrives from a sibling.
    by_module: dict[str, dict] = {}
    for r in rows:
        m = by_module.setdefault(r["module"], {"calls": 0, "fixtures": 0, "controls": 0,
                                               "files": 0, "backed": 0})
        m["calls"] += r["calls"]
        m["fixtures"] += r["fixtures"]
        m["controls"] += r["controls"]
        m["files"] += 1
        m["backed"] += 1 if r["backed"] else 0

    print(f"files scanned: {len(rows)}   modules: {len(by_module)}\n")
    print(f"{'module':22} {'files':>5} {'backed':>7} {'fixtures':>9} {'controls':>9}  verdict")
    for name, m in sorted(by_module.items(), key=lambda kv: (kv[1]["backed"], -kv[1]["controls"])):
        if m["backed"] == 0 and m["controls"] > 0:
            verdict = "NO BACKEND — controls but nothing to act on"
        elif m["backed"] == 0:
            verdict = "presentational (no controls, no calls)"
        elif m["fixtures"] > 0:
            verdict = "backed, ships fixture data too"
        else:
            verdict = "backed"
        print(f"{name:22} {m['files']:5} {m['backed']:7} {m['fixtures']:9} {m['controls']:9}  {verdict}")

    # The sharp end: a component offering controls whose module reaches no
    # backend at all, directly or through anything it imports.
    stranded = [
        r for r in rows
        if r["controls"] > 0 and r["component"] and by_module[r["module"]]["backed"] == 0
    ]
    if stranded:
        print(f"\nCOMPONENTS WITH CONTROLS WHOSE MODULE NEVER CALLS A BACKEND ({len(stranded)})")
        for r in sorted(stranded, key=lambda r: -r["controls"])[:30]:
            print(f"  {r['file']:58} controls={r['controls']:3} fixtures={r['fixtures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
