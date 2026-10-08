"""Policy documents: issue an upload, register it as a version, read its text.

The gateway signs the S3 PUT (fix round 1, 2026-10-08): this module only validates the request
and issues an upload id and a safe name, and never returns a URL or a key, which the output
scrubber would withhold. The key is rebuilt here from the id and the name when registering.

A document is recognised by its normalised filename (or named explicitly by revisionOf);
the same bytes uploaded twice are the same version, so nothing is written the second time.
Every object lives under ``agent-policy-documents/`` in the backend's bucket. Upload limits
are the ``document_intake_authority`` policy's, read-only; a missing value refuses.

get_conn() is AUTOCOMMIT: registering switches autocommit off so the version row and the
document's latest_version move together or not at all, and restores it afterwards.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ACCEPTED = {".pdf", ".docx", ".txt", ".md"}
PREFIX = "agent-policy-documents/"
UPLOAD_PREFIX = PREFIX + "uploads/"
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
}
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")
# A version mark only counts after a separator (or as a parenthesised number), so a word that
# merely ends in one -- "Overdraft", "Semifinal" -- is left whole and cannot match another document.
_SUFFIX = re.compile(r"(?:[\s_-]+|(?=\())(v\d+|\(\d+\)|final|draft|rev\d*)$")
_HASH_CONSTRAINT = "ux_bp_policy_document_version_hash"
_MAX_NAME = 120


class DocumentUnreadable(Exception):
    """The document produced no text, so there is nothing to extract policies from."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------- limits, names, S3

def intake_limits() -> Tuple[int, int]:
    """(files per request, bytes per file) from document_intake_authority. Refuses if unset."""
    from api.routers.documents import _intake_limits  # local: the router module is heavy

    return _intake_limits()


def safe_name(name: str) -> str:
    base = re.split(r"[\\/]", str(name or ""))[-1]
    cleaned = _SAFE_CHARS.sub("_", base)[:_MAX_NAME]
    return cleaned or "upload"


def normalise(name: str) -> str:
    """'Refund Policy v2.docx' -> 'refund policy'; 'refund_policy (1).pdf' -> 'refund policy'."""
    stem = Path(re.split(r"[\\/]", str(name or ""))[-1]).stem.lower().strip()
    while True:
        cut = _SUFFIX.sub("", stem).strip()
        if cut == stem or not cut:  # never strip a name down to nothing
            break
        stem = cut
    return re.sub(r"[\s_]+", " ", stem).strip()


def _suffix(name: str) -> str:
    return Path(str(name or "")).suffix.lower()


def _bucket() -> str:
    from config.settings import settings

    return settings.s3_bucket_name


_REGION_CACHE: Dict[str, str] = {}


def _bucket_region(bucket: str) -> Optional[str]:
    """The bucket's own region, from the x-amz-bucket-region header S3 sends even on a refusal.

    A presigned URL is signed for one region and S3 rejects it if that is not the bucket's
    (the backend's default region is not the bucket's), so the URL must be signed for it.
    """
    if bucket in _REGION_CACHE:
        return _REGION_CACHE[bucket]
    from botocore.exceptions import ClientError

    from services import egress

    probe = egress.aws_client("s3", purpose=egress.Purpose.OBJECT_STORAGE)
    try:
        meta = probe.head_bucket(Bucket=bucket).get("ResponseMetadata") or {}
    except ClientError as exc:
        meta = exc.response.get("ResponseMetadata") or {}
    region = (meta.get("HTTPHeaders") or {}).get("x-amz-bucket-region")
    if region:
        _REGION_CACHE[bucket] = region
    return region


def _s3():
    from botocore.config import Config

    from services import egress

    region = _bucket_region(_bucket())
    kwargs: Dict[str, Any] = {"config": Config(signature_version="s3v4")}
    if region:
        kwargs["region_name"] = region
    return egress.aws_client("s3", purpose=egress.Purpose.OBJECT_STORAGE, **kwargs)


# ---------------------------------------------------------------- issue

def _validated(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The whole request against the intake limits. First violation refuses all."""
    max_files, max_bytes = intake_limits()
    files = list(files or [])
    if not files:
        raise ValueError("No files were sent.")
    if len(files) > max_files:
        raise ValueError(f"At most {max_files} files can be uploaded at once; {len(files)} were sent.")
    for f in files:
        name = str(f.get("name") or "")
        if _suffix(name) not in ACCEPTED:
            raise ValueError(f"{name or 'A file'}: only {', '.join(sorted(ACCEPTED))} files are accepted.")
        try:
            size = int(f.get("size"))
        except (TypeError, ValueError):
            raise ValueError(f"{name}: the file size is missing.") from None
        if size <= 0:
            raise ValueError(f"{name}: the file is empty.")
        if size > max_bytes:
            raise ValueError(f"{name}: the file is larger than the {max_bytes} byte limit.")
    return files


def issue_uploads(files: List[Dict[str, Any]], *, actor: str) -> List[Dict[str, Any]]:
    """One upload id per file, with its safe name and the content type the PUT must carry.

    No URL and no key: the gateway signs the PUT for upload_key(uploadId, name).
    """
    return [{"uploadId": str(uuid.uuid4()), "safeName": safe_name(f["name"]),
             "contentType": _CONTENT_TYPES[_suffix(str(f["name"]))]}
            for f in _validated(files)]


def upload_key(upload_id: Any, name: str) -> str:
    """agent-policy-documents/uploads/<uuid4>/<safe name>; refuses anything but a canonical uuid4."""
    uid = str(upload_id or "")
    refused = ValueError(f"{name or 'An upload'} is not an agent-policy upload.")
    try:
        parsed = uuid.UUID(uid)
    except ValueError:
        raise refused from None
    if str(parsed) != uid or parsed.version != 4:
        raise refused
    return f"{UPLOAD_PREFIX}{uid}/{safe_name(name)}"


# ---------------------------------------------------------------- data access (tuple cursors)

def _doc_by_id(cur, document_id) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT document_id, title, latest_version FROM proc.bp_policy_document"
                " WHERE document_id = %s", (document_id,))
    row = cur.fetchone()
    return {"document_id": row[0], "title": row[1], "latest_version": row[2]} if row else None


def _doc_by_match(cur, match_name: str) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT document_id, title, latest_version FROM proc.bp_policy_document"
                " WHERE match_name = %s ORDER BY document_id LIMIT 1", (match_name,))
    row = cur.fetchone()
    return {"document_id": row[0], "title": row[1], "latest_version": row[2]} if row else None


def _version_by_hash(cur, document_id, content_hash: str) -> Optional[int]:
    cur.execute("SELECT version FROM proc.bp_policy_document_version"
                " WHERE document_id = %s AND content_hash = %s", (document_id, content_hash))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _lock_latest(cur, document_id) -> int:
    cur.execute("SELECT latest_version FROM proc.bp_policy_document WHERE document_id = %s FOR UPDATE",
                (document_id,))
    return int(cur.fetchone()[0])


def _insert_document(cur, title: str, match_name: str, actor: str) -> int:
    cur.execute("INSERT INTO proc.bp_policy_document (title, match_name, latest_version, created_by)"
                " VALUES (%s, %s, 1, %s) RETURNING document_id", (title, match_name, actor))
    return int(cur.fetchone()[0])


def _insert_version(cur, document_id, version, filename, key, size, content_hash, actor) -> None:
    cur.execute("INSERT INTO proc.bp_policy_document_version (document_id, version, filename, s3_key,"
                " byte_size, content_hash, uploaded_by) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (document_id, version, filename, key, size, content_hash, actor))


def _bump_latest(cur, document_id, version) -> None:
    cur.execute("UPDATE proc.bp_policy_document SET latest_version = %s WHERE document_id = %s",
                (version, document_id))


# ---------------------------------------------------------------- register

def _fetch_upload(client, bucket: str, key: str, max_bytes: int, label: str = "") -> bytes:
    label = label or "The upload"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - botocore ClientError; a missing object is a refusal
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            raise ValueError(f"{label}: the file was not uploaded.") from None
        raise
    size = int(head.get("ContentLength") or 0)
    if size <= 0:
        raise ValueError(f"{label}: the uploaded file is empty.")
    if size > max_bytes:
        raise ValueError(f"{label}: the uploaded file is larger than the {max_bytes} byte limit.")
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        data = body.read(max_bytes + 1)  # never pull more than the limit, whatever head said
    finally:
        try:
            body.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
    if len(data) > max_bytes:
        raise ValueError(f"{label}: the uploaded file is larger than the {max_bytes} byte limit.")
    if len(data) != size:
        raise ValueError(f"{label}: the uploaded file changed while it was being read.")
    return data


def _check_issued_key(key: str, name: str) -> None:
    """The key must be exactly one upload_key builds for this name: uploads/<uuid4>/<safe name>."""
    refused = ValueError(f"{name or 'An upload'} is not an agent-policy upload.")
    if not key.startswith(UPLOAD_PREFIX):
        raise refused
    parts = key[len(UPLOAD_PREFIX):].split("/")
    if len(parts) != 2:
        raise refused
    try:
        if str(uuid.UUID(parts[0])) != parts[0]:
            raise refused
    except ValueError:
        raise refused from None
    if parts[1] != safe_name(name):
        raise refused


def _is_hash_violation(exc: Exception) -> bool:
    diag = getattr(exc, "diag", None)
    return getattr(exc, "pgcode", None) == "23505" and getattr(diag, "constraint_name", None) == _HASH_CONSTRAINT


def _register_one(conn, client, bucket, max_bytes, upload, actor) -> Dict[str, Any]:
    name = str(upload.get("name") or "")
    revision_of = upload.get("revisionOf")
    if _suffix(name) not in ACCEPTED:
        raise ValueError(f"{name or 'A file'}: only {', '.join(sorted(ACCEPTED))} files are accepted.")
    key = upload_key(upload.get("uploadId"), name)
    _check_issued_key(key, name)
    data = _fetch_upload(client, bucket, key, max_bytes, label=name)
    content_hash = hashlib.sha256(data).hexdigest()
    filename = safe_name(name)
    match_name = normalise(name)
    title = Path(filename).stem or filename

    previous_autocommit = conn.autocommit
    conn.autocommit = False
    cur = conn.cursor()
    try:
        if revision_of is not None:
            doc = _doc_by_id(cur, revision_of)
            if doc is None:
                raise ValueError(f"Document {revision_of} does not exist.")
        else:
            doc = _doc_by_match(cur, match_name)

        if doc is not None:
            # Lock first, so a concurrent upload of the same bytes waits and then sees this one.
            latest = _lock_latest(cur, doc["document_id"])
            existing = _version_by_hash(cur, doc["document_id"], content_hash)
            if existing is not None:
                conn.rollback()
                return _duplicate(doc, existing)
            version = latest + 1
            try:
                _insert_version(cur, doc["document_id"], version, filename, key, len(data), content_hash, actor)
            except Exception as exc:
                if not _is_hash_violation(exc):
                    raise
                conn.rollback()  # someone registered these bytes between our check and insert
                existing = _version_by_hash(conn.cursor(), doc["document_id"], content_hash)
                conn.rollback()  # end the read's transaction; autocommit cannot change inside one
                if existing is None:
                    raise
                return _duplicate(doc, existing)
            _bump_latest(cur, doc["document_id"], version)
            document_id, doc_title = doc["document_id"], doc["title"]
        else:
            document_id = _insert_document(cur, title, match_name, actor)
            version, doc_title = 1, title
            _insert_version(cur, document_id, 1, filename, key, len(data), content_hash, actor)
        conn.commit()
        return {"documentId": document_id, "version": version, "title": doc_title,
                "isRevision": version > 1, "duplicate": False}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = previous_autocommit


def _duplicate(doc: Dict[str, Any], version: int) -> Dict[str, Any]:
    return {"documentId": doc["document_id"], "version": version, "title": doc["title"],
            "isRevision": version > 1, "duplicate": True}


def register_uploads(conn, uploads: List[Dict[str, Any]], *, actor: str) -> List[Dict[str, Any]]:
    """Record each uploaded object as a document version (or recognise it as one already held)."""
    _, max_bytes = intake_limits()
    client, bucket = _s3(), _bucket()
    return [_register_one(conn, client, bucket, max_bytes, u, actor) for u in (uploads or [])]


# ---------------------------------------------------------------- text

def _parse_bytes(filename: str, data: bytes) -> str:
    suffix = _suffix(filename)
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    if suffix in (".pdf", ".docx"):
        from services.extraction import parser

        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            return parser.parse(path).full_text or ""
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    raise DocumentUnreadable(f"{filename}: {suffix or 'this'} files are not accepted")


def document_text(conn, document_id, version) -> str:
    """The version's text: stored if already parsed, otherwise read from S3, parsed and stored."""
    cur = conn.cursor()
    cur.execute("SELECT filename, s3_key, parsed_text FROM proc.bp_policy_document_version"
                " WHERE document_id = %s AND version = %s", (document_id, version))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Document {document_id} version {version} does not exist.")
    filename, key, parsed = row
    if parsed:
        return parsed
    body = _s3().get_object(Bucket=_bucket(), Key=key)["Body"]
    try:
        data = body.read()
    finally:
        try:
            body.close()
        except Exception:  # noqa: BLE001
            pass
    text = _parse_bytes(filename, data)
    if not text or not text.strip():
        raise DocumentUnreadable(f"{filename}: no text could be read from the document")
    cur.execute("UPDATE proc.bp_policy_document_version SET parsed_text = %s, parsed_at = now()"
                " WHERE document_id = %s AND version = %s", (text, document_id, version))
    return text


# ---------------------------------------------------------------- list

def list_documents(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT document_id, title, match_name, latest_version, created_by, created_at"
                " FROM proc.bp_policy_document ORDER BY document_id")
    docs = {r[0]: {"documentId": r[0], "title": r[1], "matchName": r[2], "latestVersion": r[3],
                   "createdBy": r[4], "createdAt": r[5].isoformat() if r[5] else None, "versions": []}
            for r in cur.fetchall()}
    cur.execute("SELECT document_id, version, filename, byte_size, content_hash, uploaded_by, uploaded_at,"
                " parsed_at FROM proc.bp_policy_document_version ORDER BY document_id, version")
    for r in cur.fetchall():
        if r[0] in docs:
            docs[r[0]]["versions"].append({
                "version": r[1], "filename": r[2], "byteSize": r[3], "contentHash": r[4],
                "uploadedBy": r[5], "uploadedAt": r[6].isoformat() if r[6] else None,
                "parsed": r[7] is not None})
    return list(docs.values())
