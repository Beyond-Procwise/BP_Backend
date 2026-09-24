#!/usr/bin/env python
"""Undo one triage run (spec §9.1).

    set -a; . ./.env; set +a
    ./.venv/bin/python scripts/triage_rollback.py --run-id <uuid>

Removes the Action Centre findings that run created which nobody has touched (still
open, no owner, no due date, no resolver) and the run's audit rows. Findings a person
has acted on are kept.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from src.services.db import get_conn  # noqa: E402
from src.services.triage import writer  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run-id", required=True)
    a = p.parse_args(argv)
    with get_conn() as conn:
        result = writer.rollback_run(conn, a.run_id)
    print(f"Run {a.run_id}: removed {result['findings_removed']} findings nobody had touched, "
          f"kept {result['findings_kept']} that a person had acted on, "
          f"removed {result['audit_rows_removed']} audit rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
