"""Pre-translate the whole English catalog for one language, so no user waits for it.

  ./.venv/bin/python scripts/i18n_pretranslate.py --lang ja --catalog out/en.json [--limit N] [--dry-run]
  ./.venv/bin/python scripts/i18n_pretranslate.py --publish-public --catalog out/en.json

--publish-public replaces the list of keys a signed-out visitor may read translated
(GET /i18n/public/{lang}) with the catalog keys matching config/i18n/public.json. It is
audited first; if the audit cannot be written, nothing changes.

Only strings with no cached translation for the current prompt version and model are sent;
re-running after an English edit translates just the edited strings. Like the background
filler, it yields the GPU: before each chunk it waits while extraction (or any foreground
model work) wants the card.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.i18n import audit, get_service  # noqa: E402
from src.services.i18n.gpu_gate import GpuGate  # noqa: E402
from src.services.i18n.settings import load_settings  # noqa: E402


def _gate():
    s = load_settings()
    return GpuGate(util_threshold=s.yield_gpu_util)

PUBLIC_CONFIG = Path(__file__).resolve().parents[1] / "config" / "i18n" / "public.json"


def publish_public(svc, catalog: dict) -> int:
    prefixes = tuple(json.loads(PUBLIC_CONFIG.read_text(encoding="utf-8"))["key_prefixes"])
    keys = {k: v for k, v in catalog.items() if k.startswith(prefixes)}
    before = svc.store.public_keys()
    try:
        audit.record_public_keys(published_by=f"cli:{getpass.getuser()}", total=len(keys),
                                 added=sorted(set(keys) - set(before)), removed=sorted(set(before) - set(keys)))
    except audit.AuditWriteError as exc:
        print(f"refused, the change could not be audited: {exc}", file=sys.stderr)
        return 2
    svc.store.set_public_keys(keys)
    print(f"signed-out key list: {len(keys)} keys ({', '.join(prefixes)})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", help="language to pre-translate (optional with --publish-public)")
    ap.add_argument("--lang-name", default=None, help="display name for a custom x- code")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--limit", type=int, default=0, help="translate at most N strings (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--publish-public", action="store_true",
                    help="replace the signed-out key list from this catalog (audited)")
    a = ap.parse_args(argv)
    if not a.lang and not a.publish_public:
        ap.error("--lang is required unless --publish-public is given")
    catalog = json.loads(Path(a.catalog).read_text(encoding="utf-8"))
    svc = get_service()
    if a.publish_public:
        rc = publish_public(svc, catalog)
        if rc or not a.lang:
            return rc
    hits, missing = svc.cached(a.lang, catalog)
    if a.limit:
        missing = missing[: a.limit]
    print(f"{a.lang}: {len(catalog)} strings, {len(hits)} cached, {len(missing)} to translate")
    if a.dry_run or not missing:
        return 0
    failed = 0
    step = svc.batch_size * 5
    gate, poll = _gate(), load_settings().yield_poll_seconds
    for i in range(0, len(missing), step):
        waited = 0.0
        while gate.busy():  # extraction goes first
            time.sleep(poll)
            waited += poll
        if waited:
            print(f"  waited {waited:.0f}s for the GPU", flush=True)
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
