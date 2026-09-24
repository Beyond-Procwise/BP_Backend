"""Load human translations as reviewed rows (served ahead of, and never overwritten by, the model).

  ./.venv/bin/python scripts/i18n_import_reviewed.py --lang es --catalog out/en.json --translations out/es.json
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.i18n import audit  # noqa: E402
from src.services.i18n.store import PgTranslationStore, source_hash  # noqa: E402


def pairs(en: dict, tr: dict) -> dict[str, tuple[str, str]]:
    return {source_hash(en[k]): (en[k], v) for k, v in tr.items()
            if k in en and isinstance(v, str) and v.strip() and en[k].strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--catalog", required=True, help="flat English catalog (key -> English)")
    ap.add_argument("--translations", required=True, help="flat translations (key -> text)")
    a = ap.parse_args(argv)
    en = json.loads(Path(a.catalog).read_text(encoding="utf-8"))
    tr = json.loads(Path(a.translations).read_text(encoding="utf-8"))
    rows = pairs(en, tr)
    store = PgTranslationStore()
    # Audited BEFORE the write, with any reviewed text it replaces: an import that cannot be
    # traced does not happen.
    try:
        audit.record_reviewed_import(lang=a.lang, imported_by=f"cli:{getpass.getuser()}",
                                     added=len(rows), changed=store.reviewed_changes(a.lang, rows))
    except audit.AuditWriteError as exc:
        print(f"{a.lang}: refused, the import could not be audited: {exc}", file=sys.stderr)
        return 2
    n = store.import_reviewed(a.lang, rows)
    print(f"{a.lang}: {n} reviewed translations imported ({len(tr) - len(rows)} skipped: no English source, "
          f"empty, or a duplicate English sentence)")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
