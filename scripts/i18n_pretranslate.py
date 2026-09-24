"""Pre-translate the whole English catalog for one language, so no user waits for it.

  ./.venv/bin/python scripts/i18n_pretranslate.py --lang ja --catalog out/en.json [--limit N] [--dry-run]

Only strings with no cached translation for the current prompt version and model are sent;
re-running after an English edit translates just the edited strings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.i18n import get_service  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--lang-name", default=None, help="display name for a custom x- code")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--limit", type=int, default=0, help="translate at most N strings (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    catalog = json.loads(Path(a.catalog).read_text(encoding="utf-8"))
    svc = get_service()
    hits, missing = svc.cached(a.lang, catalog)
    if a.limit:
        missing = missing[: a.limit]
    print(f"{a.lang}: {len(catalog)} strings, {len(hits)} cached, {len(missing)} to translate")
    if a.dry_run or not missing:
        return 0
    failed = 0
    step = svc.batch_size * 5
    for i in range(0, len(missing), step):
        t0 = time.monotonic()
        chunk = {k: catalog[k] for k in missing[i:i + step]}
        r = svc.translate(a.lang, chunk, lang_name=a.lang_name)
        failed += len(r.failed)
        print(f"  {min(i + step, len(missing))}/{len(missing)}  failed so far {failed}  ({time.monotonic() - t0:.0f}s)",
              flush=True)
    print(f"{a.lang}: done, {failed} served in English (see the log for reasons)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
