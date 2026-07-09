"""Document content hashing + post-extraction quality classification.

compute_content_hash reuses the renovation parser's local/S3 resolver so the
bytes we hash are exactly the bytes the extractor parses.
"""
from __future__ import annotations

import hashlib
import logging
import os

log = logging.getLogger(__name__)

_LOW_CONFIDENCE = 0.70


def compute_content_hash(file_path: str) -> str | None:
    """SHA-256 hex digest of the document bytes, or None if unresolvable."""
    from src.services.extraction.parser import _resolve_to_local
    try:
        resolved, needs_cleanup = _resolve_to_local(str(file_path))
    except FileNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive
        log.debug("content hash resolve failed for %s", file_path, exc_info=True)
        return None
    try:
        with open(resolved, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:  # pragma: no cover - defensive
        log.debug("content hash read failed for %s", file_path, exc_info=True)
        return None
    finally:
        if needs_cleanup:
            try:
                os.remove(resolved)
            except Exception:
                pass


def quality_action_from_result(result: dict) -> str | None:
    """Return 'needs_review' when an extraction succeeded but is low quality."""
    if not isinstance(result, dict):
        return None
    confidence = result.get("confidence", 0) or 0
    pk = result.get("pk") or result.get("doc_pk") or ""
    missing = result.get("missing") or []
    if not pk or confidence < _LOW_CONFIDENCE or missing:
        return "needs_review"
    return None
