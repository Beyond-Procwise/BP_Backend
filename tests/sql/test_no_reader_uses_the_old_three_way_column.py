"""Nothing reads `three_way_match` any more, and the column is gone.

The column was a VALUE reconciliation wearing the name of a delivery check
(see tests/sql/test_deal_overview_three_way.py). Renaming it is only half the
job: while both names exist, a new reader can pick the wrong one and nobody
notices, which is how the original misnaming survived for months.

So two assertions. No source file names the old column as a SQL identifier,
and the DATABASE no longer has it -- the second is what makes the first more
than a style rule.
"""
from __future__ import annotations

import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Where the string is allowed to survive, and why.
#:   two_way_match.py  -- profile_registry_version="three_way_match/line_v1" is
#:                        hashed into the resolution reproducibility
#:                        fingerprint. Renaming it would invalidate every
#:                        stored result for a cosmetic gain (design §12).
#:   three_way_match.py / its tests / dispatch.py -- the MODULE is called that,
#:                        and now genuinely is one.
#:   this file and test_deal_overview_three_way.py / the reconciliation test --
#:                        they are about the rename itself.
_ALLOWED = {
    "src/services/extraction/two_way_match.py",
    "src/services/extraction/three_way_match.py",
    "src/services/extraction/dispatch.py",
    "tests/sql/test_no_reader_uses_the_old_three_way_column.py",
    "tests/sql/test_deal_overview_three_way.py",
    "tests/sql/test_bp_deal_overview_reconciliation_sql.py",
    "tests/extraction/test_three_way_match.py",
    "tests/extraction/test_three_way_match_live.py",
}

#: `three_way_match` NOT followed by another word character -- so
#: `three_way_matched` and `three_way_match.py` do not count.
_OLD = re.compile(r"\bthree_way_match(?![\w./])")


def _sources():
    for base in ("src", "tests"):
        for path in sorted((ROOT / base).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in _ALLOWED or "/__pycache__/" in rel:
                continue
            yield rel, path


def test_no_source_file_reads_the_old_column():
    offenders = []
    for rel, path in _sources():
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if _OLD.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


@pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live db")
def test_the_view_no_longer_offers_the_old_column():
    from src.services.db import get_conn

    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT column_name FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name='bp_deal_overview'
                          AND column_name IN ('three_way_match','value_reconciled',
                                              'three_way_matched')
                        ORDER BY column_name""")
        names = [r[0] for r in cur.fetchall()]
    assert names == ["three_way_matched", "value_reconciled"], names
