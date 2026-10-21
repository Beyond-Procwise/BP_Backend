"""Save a version of negotiation_playbook_system that mentions rivals only when there are some.

    python scripts/negotiation_playbook_prompt_bidders.py            # show the change
    python scripts/negotiation_playbook_prompt_bidders.py --apply    # save it as a new version

Uses the database the environment points at (DB_NAME). Run once per database.

The wording told the model to say "We're evaluating multiple proposals..." in
round 2, and to "reference alternatives subtly", on every deal -- including deals
with one bidder, where it is not true. This makes both depend on the brief, the
same edits as email_drafting_agent.NEGOTIATION_PLAYBOOK_SYSTEM (2026-10-15 lever
audit, rec. 5).

It builds on the newest saved version, which must already carry the lever edits
(scripts/negotiation_playbook_prompt_levers.py, saved as v2 on 2026-10-15), so
promoting this one puts both changes in use. Saving does not put it in use. A
second person promotes it:

    python scripts/prompt_pointer.py promote negotiation_playbook_system <N> --approved-by NAME
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

KEY = "negotiation_playbook_system"
AUTHOR = "claude (negotiation-honest-asks)"

EDITS = (
    ("4. **Use Leverage Wisely**: Reference alternatives and market data subtly",
     "4. **Use Leverage Wisely**: Reference alternatives and market data only when the brief states them"),
    ("- Reference competitive pressure subtly: \"We're evaluating multiple proposals...\"",
     "- Mention other proposals only when the brief says other suppliers are bidding"),
)


def _lever_edits():
    spec = importlib.util.spec_from_file_location(
        "np_levers", ROOT / "scripts" / "negotiation_playbook_prompt_levers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EDITS


def revised(template: str) -> str:
    """Apply EDITS; refuse unless each old line is there exactly once."""
    for old, new in EDITS:
        if template.count(old) != 1:
            raise SystemExit(f"expected exactly one {old!r}; found {template.count(old)}")
        template = template.replace(old, new)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="save the new version")
    args = parser.parse_args()

    # Imported here so EDITS can be read where the registry is not installed.
    from services.prompt_engine import registry

    newest = max(v["version"] for v in registry.versions(KEY))
    base = registry.get_version(KEY, newest)
    old = base["template"]
    missing = [new for _, new in _lever_edits() if new not in old]
    if missing:
        raise SystemExit(f"v{newest} does not carry the lever edits; save those first "
                         "(scripts/negotiation_playbook_prompt_levers.py)")
    new = revised(old)
    sys.stdout.writelines(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True),
        f"{KEY} v{newest} (newest saved)", f"{KEY} (new)"))
    if not args.apply:
        print("\ndry run: nothing saved")
        return 0
    settings = dict(base.get("settings") or {})
    settings[registry.LEGACY_DESC] = {"prompt_template": new}
    number = registry.save_version(
        KEY, template=new, author=AUTHOR, syntax="legacy", settings=settings,
        change_note=f"v{newest} plus: other proposals mentioned only when other "
                    "suppliers are bidding (lever audit rec. 5)")
    print(f"\nsaved {KEY} v{number}; not in use until a second person promotes it:\n"
          f"  python scripts/prompt_pointer.py promote {KEY} {number} --approved-by NAME")
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
