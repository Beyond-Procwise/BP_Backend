"""Save a typed version of a prompt that was imported as it was.

    python scripts/prompt_convert_to_typed.py <key> --names a,b,c --author NAME
    python scripts/prompt_convert_to_typed.py <key> --names a,b,c --author NAME --apply

Takes the version in use, marks each `{name}` blank as `{{name}}`, declares it,
and (with --apply) saves that as a NEW version. It changes nothing in use: a
second person promotes it with scripts/prompt_pointer.py.

Refuses unless the conversion is lossless and filling the typed template gives
the same text, character for character, as the older fill does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from services.prompt_engine import convert, registry, templates  # noqa: E402

_PROBES = ("plain", "", "with {braces} and }} and {{doubles}}", "multi\nline £ é",
           '{"json": [1, 2]}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key")
    parser.add_argument("--names", required=True, help="the blanks, comma separated")
    parser.add_argument("--author", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    names = [n.strip() for n in args.names.split(",") if n.strip()]

    live = registry.resolve(args.key)
    if live["syntax"] != "legacy":
        print(f"{args.key} version {live['version']} is already a typed template")
        return 1
    typed, declared = convert.braces_to_typed(live["template"], names)
    used = [d["name"] for d in declared]

    for probe in _PROBES:
        older = live["template"]
        for name in used:
            older = older.replace("{" + name + "}", probe)
        newer = templates.render_text(typed, declared, {n: probe for n in used}, {})
        if older != newer:
            print(f"REFUSED: the typed template fills differently for a value like {probe!r}")
            return 2

    print(f"{args.key}: version {live['version']} in use, {len(live['template'])} characters")
    print(f"blanks declared ({len(used)}): {', '.join(used)}")
    unused = [n for n in names if n not in used]
    if unused:
        print(f"named but not in the wording: {', '.join(unused)}")
    print("fills identically to the older way for every probe value")
    if not args.apply:
        print("nothing saved (run again with --apply)")
        return 0
    settings = {k: v for k, v in (live["settings"] or {}).items() if k != registry.LEGACY_DESC}
    number = registry.save_version(
        args.key, template=typed, author=args.author, variables=declared, settings=settings,
        change_note=f"version {live['version']} as a typed template; wording unchanged")
    print(f"saved as version {number}. Not in use until promoted by someone other than "
          f"{args.author}.")
    return 0


if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    raise SystemExit(main())
