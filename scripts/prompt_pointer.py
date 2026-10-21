"""Show, promote or roll back the version of a prompt in use.

    python scripts/prompt_pointer.py show <key>
    python scripts/prompt_pointer.py promote <key> <version> --approved-by NAME [--note ...]
    python scripts/prompt_pointer.py rollback <key> --approved-by NAME [--note ...] [--to N]

A promotion is refused if the approver wrote the version. A rollback returns to
a version that was in use before. Both take effect on the very next render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from services.prompt_engine import registry  # noqa: E402


def _show(key: str) -> None:
    try:
        live = registry.resolve(key)["version"]
    except registry.PointerUnresolved:
        live = None
    for v in registry.versions(key):
        mark = "  <- in use" if v["version"] == live else ""
        print(f"  v{v['version']}  {v['syntax']:6}  by {v['author']}  {v['created_at']:%Y-%m-%d %H:%M}"
              f"  {v['change_note'] or ''}{mark}")
    for m in registry.pointer_history(key):
        print(f"  {m['moved_at']:%Y-%m-%d %H:%M}  {m['change']:11}  "
              f"v{m['from_version']} -> v{m['to_version']}  by {m['approved_by']}  {m['note'] or ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("show").add_argument("key")
    promote = sub.add_parser("promote")
    promote.add_argument("key")
    promote.add_argument("version", type=int)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("key")
    rollback.add_argument("--to", type=int, default=None)
    for p in (promote, rollback):
        p.add_argument("--approved-by", required=True)
        p.add_argument("--note", default=None)
    args = parser.parse_args()

    try:
        if args.action == "promote":
            now = registry.promote(args.key, args.version, approved_by=args.approved_by,
                                   note=args.note)
            print(f"{args.key}: version {now} is now in use")
        elif args.action == "rollback":
            now = registry.rollback(args.key, approved_by=args.approved_by, note=args.note,
                                    to_version=args.to)
            print(f"{args.key}: rolled back; version {now} is now in use")
        _show(args.key)
    except registry.RegistryError as exc:
        print(f"refused ({exc.code}): {exc}")
        return 1
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
