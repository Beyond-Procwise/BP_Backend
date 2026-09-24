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
from src.services.i18n.registry import build_registry  # noqa: E402
from src.services.i18n.settings import load_settings  # noqa: E402
from src.services.i18n.store import PgTranslationStore, source_hash  # noqa: E402
from src.services.i18n.validate import check_pair  # noqa: E402


def invalid(en: dict, tr: dict, lang: str) -> dict[str, str]:
    """Human text held to the same rules as the model's: key -> why it was refused."""
    out = {}
    for k, v in tr.items():
        if k in en and isinstance(v, str) and v.strip() and en[k].strip():
            reason = check_pair(en[k], v, lang)
            if reason:
                out[k] = reason
    return out


def pairs(en: dict, tr: dict) -> dict[str, tuple[str, str]]:
    return {source_hash(en[k]): (en[k], v) for k, v in tr.items()
            if k in en and isinstance(v, str) and v.strip() and en[k].strip()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--catalog", required=True, help="flat English catalog (key -> English)")
    ap.add_argument("--translations", required=True, help="flat translations (key -> text)")
    a = ap.parse_args(argv)
    language = build_registry(load_settings().model).get(a.lang)
    if language is None:
        print(f"{a.lang}: refused, not a known language code", file=sys.stderr)
        return 2
    lang = language.code  # "ES" / "es_ES" are stored the way the service looks them up
    en = json.loads(Path(a.catalog).read_text(encoding="utf-8"))
    tr = json.loads(Path(a.translations).read_text(encoding="utf-8"))
    bad = invalid(en, tr, lang)
    for k, why in sorted(bad.items()):
        print(f"  skipped {k}: {why}")
    rows = pairs(en, {k: v for k, v in tr.items() if k not in bad})
    store = PgTranslationStore()
    # Audited BEFORE the write, with any reviewed text it replaces: an import that cannot be
    # traced does not happen.
    try:
        audit.record_reviewed_import(lang=lang, imported_by=f"cli:{getpass.getuser()}",
                                     added=len(rows), changed=store.reviewed_changes(lang, rows))
    except audit.AuditWriteError as exc:
        print(f"{lang}: refused, the import could not be audited: {exc}", file=sys.stderr)
        return 2
    n = store.import_reviewed(lang, rows)
    print(f"{lang}: {n} reviewed translations imported ({len(tr) - len(rows)} skipped: no English source, "
          f"empty, failing the placeholder check, or a duplicate English sentence)")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
