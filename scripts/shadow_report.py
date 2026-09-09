#!/usr/bin/env python
"""What the rules would have done, over a date range.

Read by someone deciding whether it is safe to start enforcing, so it answers
that question and not a more interesting one:

  * how many decisions were observed at all -- because "no denials" and "not
    observing" look identical without it
  * how many would have been denied, by policy, by action, by person
  * which enrolments are still live, and which have expired

Usage:
    ./venv/bin/python scripts/shadow_report.py [--days 7] [--action supplier.read]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:  # noqa: BLE001
    pass


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="how far back to look")
    parser.add_argument("--action", help="restrict to one action name")
    args = parser.parse_args()

    from src.services.db import get_conn

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    where = "observed_at >= %s"
    params: list = [since]
    if args.action:
        where += " AND action = %s"
        params.append(args.action)

    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM proc.bp_policy_observation WHERE {where}", params)
        total = cur.fetchone()[0]

        print(f"\nShadow report — {args.days} day(s) to {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
        if args.action:
            print(f"action: {args.action}")

        if total == 0:
            print(
                "\n  NOTHING WAS OBSERVED.\n"
                "  This is not the same as 'nothing would have been denied'. Either the gate\n"
                "  was not reached in this window, or observations are not being written.\n"
                "  Check /health -> shadow_mode and the service log before concluding.\n"
            )
            return 0

        cur.execute(
            f"SELECT COUNT(*) FILTER (WHERE would_have_denied), "
            f"COUNT(*) FILTER (WHERE shadowed) "
            f"FROM proc.bp_policy_observation WHERE {where}",
            params,
        )
        denied, shadowed = cur.fetchone()
        print(f"\n  observed          {total}")
        print(f"  would have denied {denied}  ({100 * denied // max(total, 1)}%)")
        print(f"  allowed through by shadow mode  {shadowed}")
        if denied and not shadowed:
            print("  (none were shadowed: these were real refusals)")

        _rule("Would have been denied — by action")
        cur.execute(
            f"SELECT action, COUNT(*), COUNT(*) FILTER (WHERE shadowed) "
            f"FROM proc.bp_policy_observation WHERE {where} AND would_have_denied "
            f"GROUP BY action ORDER BY 2 DESC LIMIT 25",
            params,
        )
        rows = cur.fetchall()
        if not rows:
            print("  none")
        for action, count, shad in rows:
            print(f"  {count:>6}  {action:<34} ({shad} shadowed)")

        _rule("Would have been denied — by policy")
        cur.execute(
            f"SELECT COALESCE(policy_name, '(no policy — default deny)'), COUNT(*) "
            f"FROM proc.bp_policy_observation WHERE {where} AND would_have_denied "
            f"GROUP BY 1 ORDER BY 2 DESC LIMIT 25",
            params,
        )
        for name, count in cur.fetchall() or [("none", 0)]:
            print(f"  {count:>6}  {name}")

        _rule("Would have been denied — by person")
        cur.execute(
            f"SELECT COALESCE(principal_subject, '(no principal)'), COALESCE(role, '?'), COUNT(*) "
            f"FROM proc.bp_policy_observation WHERE {where} AND would_have_denied "
            f"GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 25",
            params,
        )
        for subject, role, count in cur.fetchall() or [("none", "", 0)]:
            print(f"  {count:>6}  {subject:<40} {role}")

        _rule("Most common reasons")
        cur.execute(
            f"SELECT reason, COUNT(*) FROM proc.bp_policy_observation "
            f"WHERE {where} AND would_have_denied GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            params,
        )
        for reason, count in cur.fetchall():
            print(f"  {count:>6}  {(reason or '')[:100]}")

    _rule("Enrolments")
    try:
        from services import guardrail

        status = guardrail.shadow_status()
        if not status.get("enrolled"):
            print("  nothing enrolled — every rule is enforcing")
        for entry in status.get("enrolled", []):
            state = "ACTIVE" if entry.get("active") else "EXPIRED — now enforcing"
            print(f"  {entry.get('action'):<34} until {entry.get('until')}  [{state}]")
        print(f"  never shadowable: {', '.join(status.get('never_shadowed', []))}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read enrolments: {exc}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
