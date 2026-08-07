"""The one place mail-intake hands an artifact to the extraction plane.

Extraction resolves an inbound `file_path` by trying the local filesystem and
then falling back to fetching that string as an S3 key. That fallback is the
plane crossing: given a key it fetches, and the credentials it uses can read
any key in the bucket — including the quarantine prefix, where the bytes the
safety gate rejected are kept.

So mail-intake never gives extraction a key. It writes the artifact's bytes to
a private temporary file, hands over that path, and removes it afterwards
whatever happens. Extraction's resolver takes its local-file branch, never
reaches S3, and has no string to vary. The boundary holds for this path by
construction rather than by anyone remembering the rule.

This does NOT make extraction incapable of reaching S3 in general — other
callers still pass keys, and closing that needs a deployment change (spec
§2.2). The import guard alongside this module is a ratchet over that gap, not
a cure for it. Say so plainly wherever this is documented.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# The crossings that exist today, recorded so the guard can fail on anything
# NEW while not pretending these are absent. Adding to this set is a decision
# a human makes deliberately, not a way to quiet a failing build.
KNOWN_CROSSINGS: Dict[str, set] = {
    "src/services/extraction/parser.py": {"boto3"},
}


def deliver_to_extraction(
    *,
    artifact_id: str,
    content: bytes,
    doc_type: str,
    process_monitor_id: Optional[int] = None,
    dispatch: Optional[Callable[..., Any]] = None,
) -> Any:
    """Give extraction the bytes, on a path it did not choose.

    `dispatch` is injectable so tests need no extraction stack; production
    resolves the real entry point lazily, which also keeps this module free of
    an import-time dependency on the extraction package.
    """
    if dispatch is None:  # pragma: no cover - exercised in integration only
        from src.services.extraction.dispatch import dispatch_document as dispatch

    # 0o600: these are attachments from arbitrary senders sitting briefly on
    # local disk, and mkstemp already creates them owner-only. Being explicit
    # keeps that true if the surrounding code ever changes umask.
    handle, path = tempfile.mkstemp(prefix=f"mailintake-{artifact_id}-", suffix=".bin")
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "wb") as fh:
            fh.write(content)
        return dispatch(
            process_monitor_id=process_monitor_id,
            file_path=path,
            doc_type=doc_type,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.warning(
                "mail_intake.handoff could not remove temporary file for artifact=%s",
                artifact_id,
            )
