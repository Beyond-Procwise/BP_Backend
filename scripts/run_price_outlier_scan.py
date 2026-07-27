"""Scan for extreme prices. Dry run by default.

    .venv/bin/python scripts/run_price_outlier_scan.py --target bp_testdb
    .venv/bin/python scripts/run_price_outlier_scan.py --target bp_testdb --write

Reports how many findings it WOULD raise before writing any, so the volume can
be judged before the Action Centre fills up.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Running as `python scripts/run_price_outlier_scan.py` sets sys.path[0] to the
# scripts/ directory itself, not the repo root, so `scripts.testdata` would
# otherwise fail to import. Derive the root from this file's own location
# (not the caller's cwd) so the script works from any invocation directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.testdata.db import connect  # noqa: E402
from services.price_outlier import find_outliers, persist_findings  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="bp_testdb")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    conn = connect(args.target)
    try:
        with conn.cursor() as cur:
            findings = find_outliers(cur)
            by_severity = Counter(f.verdict.severity for f in findings)
            print(f"{len(findings)} findings: {dict(by_severity)}")
            for finding in findings[:20]:
                print(f"  [{finding.verdict.severity}] {finding.note}")
            if not args.write:
                print("\ndry run — nothing written. Pass --write to persist.")
                return 0
            written = persist_findings(cur, findings)
        conn.commit()
        print(f"wrote {written} new findings")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
