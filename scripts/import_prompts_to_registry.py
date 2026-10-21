"""Import the prompts in proc.bp_prompt into the prompt registry.

    python scripts/import_prompts_to_registry.py            # say what it would do
    python scripts/import_prompts_to_registry.py --apply    # do it

Uses the database the environment points at. Run once per database.
Needs deploy/sql/2026-10-13_bp_prompt_registry.sql applied first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from services.prompt_engine import legacy_import  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the import")
    args = parser.parse_args()

    report = legacy_import.run(apply=args.apply)
    verb = "imported" if args.apply else "would import"
    print(f"{verb}: {len(report['imported'])}")
    for key in report["imported"]:
        print(f"  {key}")
    print(f"already in the registry: {len(report['already_there'])}")
    for label in ("drifted", "refused"):
        if report[label]:
            print(f"{label}: {len(report[label])}")
            for line in report[label]:
                print(f"  {line}")
    return 1 if report["refused"] or report["drifted"] else 0


if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    raise SystemExit(main())
