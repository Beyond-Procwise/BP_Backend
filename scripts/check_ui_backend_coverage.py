"""Every endpoint the UI asks for, and whether a backend answers it.

The UI reaches two backends -- the Node gateway and this one -- and calls them
through several shapes: axios directly, a shared `http` client, and the
`window.__SPENDIQ_API_*__` bridges the SpendIQ engine publishes. An earlier,
looser version of this check matched on a path prefix and reported
/agents/tool-options as covered because /agents/tools existed. It is not the
same endpoint, and the tool picker had been quietly falling back to invented
placeholder data for months.

So this matches on method AND full path, treats a path parameter as matching
only one segment, and says which backend answers. A path no backend serves is
reported; that list is the thing worth acting on.

Usage:
  check_ui_backend_coverage.py <ui_src_dir> <bp_routes.txt> <gw_routes.txt>

Route files are one "METHOD /path" per line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

VERBS = ("get", "post", "put", "patch", "delete")

# Paths the router owns: pages, not endpoints.
IN_APP = {
    "/spendiq", "/home", "/login", "/analyse", "/support", "/actions",
    "/dashboard", "/forgot-password", "/reset-password", "/new-password",
    "/unauthorized", "/settings/mail", "/translation-audit", "/whats-new",
    "/report-builder", "/", "/quotes", "/suppliers", "/deals", "/records",
    # Actions/access.js SCREEN_PATH: which screen of the Action Centre is
    # showing, not a request. They are path-shaped and go nowhere near HTTP.
    "/actions/queue", "/actions/focus", "/actions/overview",
}

# Fragments that are concatenated onto a base elsewhere, not endpoints on
# their own. Reported separately rather than counted as misses.
FRAGMENT = re.compile(r"^/(confirm|reject|refuse|dismiss|deck|draft|preview|page|"
                      r"messages|versions|signoff|resume|sibling|empty|meta|month|"
                      r"strings|preference|languages|audit|inbox|brief|insights|"
                      r"contracts|demand|negotiations|won|toggle-status)(/.*)?$")

CALL = re.compile(
    r"""(?:axios|http|api|client)\s*\.\s*(get|post|put|patch|delete)\s*\(\s*"""
    r"""|__SPENDIQ_API(?:_AI)?(?:_(GET|POST|PUT|PATCH|DELETE|BLOB))?__\s*\(\s*""",
    re.I | re.X,
)


def first_argument(text: str, start: int) -> str:
    """The source of the call's first argument, from just after its "(".

    Needed because a path is often built by concatenation --
    '/reports/jobs/' + encodeURIComponent(jobId) + '/dismiss' -- and reading
    only the opening literal yields '/reports/jobs', which is a different
    endpoint that happens to exist. That is exactly the false "covered" this
    script was written to stop, in the other direction.
    """
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            quote, i = c, i + 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return text[start:i]
            depth -= 1
        elif c == "," and depth == 0:
            return text[start:i]
        i += 1
    return text[start:i]


LITERAL = re.compile(r"[\"'`]([^\"'`]*)[\"'`]")

# Aliases that hold across files, if any are ever introduced.
CALL_ALIASES: dict[str, str] = {}


TERNARY = re.compile(
    r"\(\s*[^()?:]*\?\s*([\"'`][^\"'`]*[\"'`])\s*:\s*([\"'`][^\"'`]*[\"'`])\s*\)")


def path_from_argument(arg: str) -> list[str]:
    """Stitch the literals of a concatenated path, expressions becoming "{}".

    A ternary picking between two suffixes -- (page ? '/page' : '/deck') --
    is two endpoints, not one. Merging them produced the nonsense path
    '/reports/jobs/{}/page{}/deck', reported as missing while both real routes
    existed. Each branch is expanded instead.
    """
    variants = [arg]
    m = TERNARY.search(arg)
    if m:
        variants = [arg[:m.start()] + m.group(1) + arg[m.end():],
                    arg[:m.start()] + m.group(2) + arg[m.end():]]

    paths = []
    for variant in variants:
        out, pos = [], 0
        for lit in LITERAL.finditer(variant):
            gap = variant[pos:lit.start()]
            if out and gap.strip(" +\n\t"):
                out.append("{}")
            out.append(lit.group(1))
            pos = lit.end()
        if out and variant[pos:].strip(" +\n\t"):
            out.append("{}")
        joined = "".join(out)
        if joined:
            paths.append(joined)
    return paths


def normalise(raw: str) -> str:
    p = raw.split("?")[0].split("#")[0]
    p = re.sub(r"\$\{[^}]*\}", "{}", p)       # template holes
    p = re.sub(r"\{[^}]*\}", "{}", p)         # FastAPI-style params
    # NestJS writes a parameter as :name. Without this the gateway's
    # /policies/update/:id was compared as a literal segment called ":id",
    # so a real call to /policies/update/7 looked unserved -- and, worse, a
    # path that genuinely had no route could have matched one that did.
    p = re.sub(r"/:[A-Za-z_][\w]*", "/{}", p)
    p = re.sub(r"/+", "/", p).rstrip("/")
    return p or "/"


def looks_like_endpoint(p: str) -> bool:
    if not p.startswith("/") or p in IN_APP:
        return False
    if re.search(r"\.(js|jsx|css|png|svg|ico|json|woff2?)$", p):
        return False
    return len(p) > 3


def load_routes(path: Path) -> set[tuple[str, str]]:
    out = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() in {v.upper() for v in VERBS}:
            out.add((parts[0].upper(), normalise(parts[1])))
    return out


def served_by(method: str | None, path: str, table: set[tuple[str, str]]) -> bool:
    """A path parameter matches exactly one segment -- never several."""
    want = path.strip("/").split("/")
    for m, route in table:
        if method and m != method:
            continue
        have = route.strip("/").split("/")
        if len(have) != len(want):
            continue
        if all(a == b or b == "{}" or a == "{}" for a, b in zip(want, have)):
            return True
    return False


def collect(ui_src: Path) -> dict[tuple[str | None, str], set[str]]:
    found: dict[tuple[str | None, str], set[str]] = {}
    for f in sorted(ui_src.rglob("*.js*")):
        s = str(f)
        if any(x in s for x in (".test.", "__tests__", ".stories.", "/fixtures", "fixtures.")):
            continue
        text = f.read_text(errors="ignore")
        # A bridge is often aliased before use -- `const P = window.__SPENDIQ_API_AI_PATCH__`
        # and then `P('/agents/'+slug, body)`. Matching only the bridge's own name
        # missed every call made through an alias, including the one route this
        # check was rerun to confirm.
        alias = re.compile(
            r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*window\.__SPENDIQ_API"
            r"(?:_AI)?(?:_(GET|POST|PUT|PATCH|DELETE|BLOB))?__")
        per_file = dict(CALL_ALIASES)
        for a in alias.finditer(text):
            per_file[a.group(1)] = (a.group(2) or "GET").upper()
        if per_file:
            names = "|".join(re.escape(k) for k in per_file)
            for am in re.finditer(rf"\b({names})\s*\(\s*", text):
                verb = per_file[am.group(1)]
                if verb == "BLOB":
                    verb = "GET"
                for raw in path_from_argument(first_argument(text, am.end())):
                    if "/" not in raw:
                        continue
                    p = normalise(raw[raw.index("/"):])
                    if looks_like_endpoint(p):
                        found.setdefault((verb, p), set()).add(str(f.relative_to(ui_src)))
        for m in CALL.finditer(text):
            verb = (m.group(1) or m.group(2) or "").upper() or None
            if verb == "BLOB":
                verb = "GET"
            for raw in path_from_argument(first_argument(text, m.end())):
                # Keep only the path part of a template like `${API}/spendiq/deals`.
                if "/" not in raw:
                    continue
                p = normalise(raw[raw.index("/"):])
                if looks_like_endpoint(p):
                    found.setdefault((verb, p), set()).add(str(f.relative_to(ui_src)))
    return found


def main(argv: list[str]) -> int:
    ui_src, bp_file, gw_file = (Path(a) for a in argv)
    bp, gw = load_routes(bp_file), load_routes(gw_file)
    found = collect(ui_src)

    missing, fragments, covered = [], [], []
    for (verb, path), files in sorted(found.items(), key=lambda kv: (kv[0][1], kv[0][0] or "")):
        if FRAGMENT.match(path):
            fragments.append((verb, path, files))
            continue
        on_bp = served_by(verb, path, bp)
        on_gw = served_by(verb, path, gw)
        if on_bp or on_gw:
            covered.append((verb, path, "BP" if on_bp else "", "GW" if on_gw else ""))
        else:
            missing.append((verb, path, files))

    # Second pass, method-agnostic. The first pass only sees a path written at
    # the call itself; the i18n adapter builds one through a url() helper and a
    # ternary picks between two. Rather than teach the parser every wrapper, scan
    # every path-shaped literal in the source and report the ones no route
    # matches. Some of these will be innocent -- the point is that nothing
    # path-shaped goes unexamined.
    seen = {p for (_, p) in found}
    literal = re.compile(r"[\"'`](/[a-z][\w\-/{}$%.]*)[\"'`]", re.I)
    unresolved: dict[str, set[str]] = {}
    for f in sorted(ui_src.rglob("*.js*")):
        s = str(f)
        if any(x in s for x in (".test.", "__tests__", ".stories.", "/fixtures", "fixtures.")):
            continue
        for m in literal.finditer(f.read_text(errors="ignore")):
            p = normalise(m.group(1))
            if p in seen or not looks_like_endpoint(p) or FRAGMENT.match(p):
                continue
            if served_by(None, p, bp) or served_by(None, p, gw):
                continue
            if p.startswith("/ws/"):
                continue  # a WebSocket, not an HTTP route
            # A helper-built path often appears without its prefix; try each
            # backend prefix the UI is known to mount under.
            if any(served_by(None, pre + p, bp) or served_by(None, pre + p, gw)
                   for pre in ("/i18n", "/spendiq", "/workflows", "/value")):
                continue
            # A literal that is only the fixed half of "'/prompts/update/' + id".
            if served_by(None, p + "/{}", bp) or served_by(None, p + "/{}", gw):
                continue
            unresolved.setdefault(p, set()).add(str(f.relative_to(ui_src)))

    print(f"UI call sites resolved : {len(found)}")
    print(f"  served by a backend  : {len(covered)}")
    print(f"  URL fragments (joined elsewhere, not endpoints): {len(fragments)}")
    print(f"  NO BACKEND           : {len(missing)}")
    if missing:
        print("\nNOT SERVED BY EITHER BACKEND")
        for verb, path, files in missing:
            print(f"  {(verb or '?'):6} {path:52} {sorted(files)[0]}")
    if unresolved:
        print(f"\nPATH-SHAPED LITERALS NO ROUTE MATCHES ({len(unresolved)}) — review each")
        for p, files in sorted(unresolved.items()):
            print(f"  {p:52} {sorted(files)[0]}")
    print("\nSERVED")
    for verb, path, b, g in covered:
        print(f"  {(verb or '?'):6} {(b+g) or '??':3} {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
