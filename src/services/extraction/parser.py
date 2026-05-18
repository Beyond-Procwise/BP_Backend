"""L0 — unified parser entry. Delegates to extraction_v3.parsers.router.parse.

Resolves file_path → on-disk file by:
1. If file exists locally as given → use it.
2. Otherwise treat file_path as an S3 object key under settings.s3_bucket_name
   (matches the DataExtractionAgent convention used by the legacy pipeline).
   Tries the key as-is first, then the canonical-prefix fallback map for
   "documents/<category>/" keys.
3. Downloads to a NamedTemporaryFile and parses that.

S3-mode lazily imports boto3 so unit tests that pass a local fixture path
do not require AWS credentials.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from src.services.extraction_v3.parsers.router import parse as _route_parse
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument

log = logging.getLogger(__name__)

# Mirror DataExtractionAgent._process_documents canonical fallback map:
# documents/<category>/ → CapitalizedPrefix/
_S3_CANONICAL_PREFIXES: dict[str, str] = {
    "documents/invoice/": "Invoice/",
    "documents/po/": "Purchase_Order/",
    "documents/purchase_order/": "Purchase_Order/",
    "documents/quote/": "Invoice/",
    "documents/contract/": "Invoice/",
    "documents/spend/": "Invoice/",
}


def _s3_bucket() -> Optional[str]:
    try:
        from config.settings import Settings
        bucket = getattr(Settings(), "s3_bucket_name", None)
        if bucket:
            return str(bucket)
    except Exception:  # pragma: no cover - defensive
        pass
    return os.getenv("S3_BUCKET_NAME") or None


def _try_download_key(key: str, bucket: str, dest: str) -> bool:
    """Attempt one S3 GetObject for (bucket, key) → dest path.
    Returns True if downloaded a non-empty body."""
    try:
        import boto3
    except ImportError:  # pragma: no cover - boto3 must be present in prod
        log.warning("boto3 not installed; cannot download s3://%s/%s", bucket, key)
        return False
    try:
        client = boto3.client("s3")
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp.get("Body") if isinstance(resp, dict) else None
        if body is None:
            return False
        data = body.read()
        try:
            body.close()
        except Exception:
            pass
        if not data:
            return False
        with open(dest, "wb") as fh:
            fh.write(data)
        return True
    except Exception as exc:
        log.debug("S3 GetObject failed for s3://%s/%s: %s", bucket, key, exc)
        return False


def _resolve_to_local(file_path: str) -> str:
    """Resolve an inbound file_path to a path that exists on disk.

    Tries the literal path first, then S3 (object key = file_path) using the
    settings.s3_bucket_name, then canonical-prefix fallbacks. Returns the
    resolved local path. Raises FileNotFoundError if no resolution succeeds.
    """
    # 1. Local file present
    if os.path.isfile(file_path):
        return file_path

    bucket = _s3_bucket()
    if not bucket:
        raise FileNotFoundError(
            f"File not found locally and S3 bucket is not configured: {file_path}"
        )

    # 2. Download to temp file
    suffix = Path(file_path).suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="renov_s3_")
    os.close(fd)

    # 2a. Try the file_path as-is as an S3 key
    if _try_download_key(file_path, bucket, tmp):
        log.info("Resolved %r via S3 key %r (bucket=%s)", file_path, file_path, bucket)
        return tmp

    # 2b. Try canonical prefix mapping (e.g. documents/invoice/foo → Invoice/foo)
    for doc_prefix, s3_prefix in _S3_CANONICAL_PREFIXES.items():
        if file_path.startswith(doc_prefix):
            canonical_key = s3_prefix + file_path[len(doc_prefix):]
            if _try_download_key(canonical_key, bucket, tmp):
                log.info(
                    "Resolved %r via canonical S3 key %r (bucket=%s)",
                    file_path, canonical_key, bucket,
                )
                return tmp

    # 2c. Try basename in each canonical prefix (handles UI paths with stray
    # subdirs and lets us recover when the key wasn't laid down with the
    # documents/<cat>/ prefix at all)
    basename = Path(file_path).name
    seen_prefixes: set[str] = set()
    for s3_prefix in _S3_CANONICAL_PREFIXES.values():
        if s3_prefix in seen_prefixes:
            continue
        seen_prefixes.add(s3_prefix)
        candidate = s3_prefix + basename
        if _try_download_key(candidate, bucket, tmp):
            log.info(
                "Resolved %r via basename S3 key %r (bucket=%s)",
                file_path, candidate, bucket,
            )
            return tmp

    # Cleanup empty temp file
    try:
        os.unlink(tmp)
    except Exception:
        pass
    raise FileNotFoundError(
        f"File not found locally or in S3 (bucket={bucket}): {file_path}"
    )


def parse(file_path: Union[str, Path]) -> ParsedDocument:
    """Return a ParsedDocument for the given file path.

    file_path may be:
      - an absolute or relative local path (used directly when it exists)
      - an S3 object key (downloaded to a temp file and parsed)
      - a "documents/<category>/<filename>" path that resolves to an S3
        canonical prefix (Invoice/, Purchase_Order/, etc.)

    Supported formats: PDF (native + scanned), DOCX, PNG, JPG, JPEG.
    Raises FileNotFoundError when the file cannot be resolved.
    Raises ValueError for unsupported formats.
    """
    resolved = _resolve_to_local(str(file_path))
    return _route_parse(Path(resolved))
