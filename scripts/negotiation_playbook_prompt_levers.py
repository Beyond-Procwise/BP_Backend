"""Save a version of negotiation_playbook_system without its stock levers and figures.

    python scripts/negotiation_playbook_prompt_levers.py            # show the change
    python scripts/negotiation_playbook_prompt_levers.py --apply    # save it as a new version

Uses the database the environment points at (DB_NAME). Run once per database.

The wording in use carried example emails and a "common levers" list -- a
12->24 month term, Net-30 -> Net-15, "£X" -- and the model copied them into
negotiations they had nothing to do with. This removes them, the same edits as
email_drafting_agent.NEGOTIATION_PLAYBOOK_SYSTEM (2026-10-15 lever audit, rec. 3),
and keeps everything else in the version in use, including the output-style
section added to the database copy on 2026-07-24.

Saving does not put the version in use. A second person promotes it:

    python scripts/prompt_pointer.py promote negotiation_playbook_system <N> --approved-by NAME

and promotion copies it to proc.bp_prompt, where the email drafter reads it.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

KEY = "negotiation_playbook_system"
AUTHOR = "claude (negotiation-levers)"

EDITS = (
    ("- Offer 2-3 commercial levers they can adjust (payment terms, volume, contract length)\n",
     "- Raise only the commercial levers listed in the brief (the terms the supplier raised, "
     "and plays this deal's evidence supports)\n"),
    ('**Example Opening**: "Thank you for your proposal. We\'re impressed with the solution, '
     "though our budget positioning is around £X. We'd value discussing how we might bridge "
     'this gap through adjusted payment terms or extended contract commitment."\n\n', ""),
    ('- Introduce specific commercial levers: "If we commit to 24 months instead of 12..."\n',
     "- Answer the specific terms the supplier raised, using only the levers listed in the brief\n"),
    ('**Example Middle**: "We appreciate your revised position. To help us move forward, could '
     "you share the cost breakdown? We're prepared to commit to a 24-month term and Net-15 "
     'payment if we can align on £X per unit."\n\n', ""),
    ("- Offer face-saving concessions: small scope additions, faster payment\n",
     "- Offer a face-saving concession only from the levers listed in the brief\n"),
    ('**Example Closing**: "We\'ve reached our maximum flexibility at £X with Net-15 payment '
     "terms. This represents our best and final offer. If you can accommodate these terms, "
     "please confirm by close of business Thursday, and we'll proceed with the award "
     'immediately."\n\n', ""),
    ("Common levers to offer:\n"
     "- Extended contract duration (12→24 months)\n"
     "- Volume commitments or guaranteed orders\n"
     "- Payment terms (Net-30 → Net-15 with discount)\n"
     "- Simplified scope or phased delivery\n"
     "- Marketing/reference opportunities\n"
     "- Exclusivity or preferred supplier status\n",
     "Use only the levers listed in the brief: the terms the supplier raised in their own "
     "words, and plays this deal's evidence supports.\n"
     "- Quote the supplier's own figures for the terms they raised\n"
     "- Never introduce a term length, payment term, volume or percentage that is not in "
     "the brief\n"
     "- If the brief lists none, negotiate on price alone\n"),
    ("Subtly reference:\n"
     '- "Market rates we\'re seeing are around £X..."\n'
     '- "Competitive proposals are positioning at £X..."\n'
     '- "Industry benchmarks suggest £X is achievable..."\n',
     "Reference market rates, competing proposals or benchmarks only when the brief gives "
     "the figure; never invent one.\n"),
    ("4. **Commercial Levers**: Offer 2-3 specific adjustments you can make\n",
     "4. **Commercial Levers**: The levers listed in the brief, if any\n"),
)


def revised(template: str) -> str:
    for old, new in EDITS:
        if template.count(old) != 1:
            raise SystemExit(f"the wording in use does not contain exactly one of:\n{old}")
        template = template.replace(old, new)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="save the new version")
    args = parser.parse_args()

    # Imported here so EDITS can be read where the registry is not installed.
    from services.prompt_engine import registry

    live = registry.resolve(KEY)
    old = live["template"]
    new = revised(old)
    sys.stdout.writelines(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True),
        f"{KEY} v{live['version']} (in use)", f"{KEY} (new)"))
    if not args.apply:
        print("\ndry run: nothing saved")
        return 0
    settings = dict(live.get("settings") or {})
    settings[registry.LEGACY_DESC] = {"prompt_template": new}
    number = registry.save_version(
        KEY, template=new, author=AUTHOR, syntax="legacy", settings=settings,
        change_note="stock levers and example figures removed (lever audit rec. 3)")
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
