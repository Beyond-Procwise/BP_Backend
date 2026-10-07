"""Turn the report builder's rendered page into a file: a PDF, or an Excel workbook.

The browser already assembles a complete, self-contained print document (charts are
PNG data-URIs), so the server only needs HTML + CSS -> PDF. No headless browser.

The HTML arrives from a caller, so rendering it must not become a way to make this
server fetch things. WeasyPrint follows ``<img src>``, ``<link href>`` and CSS
``url()`` by default; the fetcher below refuses everything except ``data:`` URIs, so
a ``file:///etc/passwd`` or an ``http://169.254.169.254/`` in the page is refused,
not fetched.
"""
from __future__ import annotations

import io
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_HTML_BYTES = 12 * 1024 * 1024
MAX_TABLE_ROWS = 50_000
_BUCKET_DEFAULT = "procwisemvp"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ExportRefused(ValueError):
    """The request cannot be turned into a file; the message says why."""


def _data_only_fetcher():
    # fail_on_errors: a refused URL stops the render instead of being skipped with a warning.
    from weasyprint.urls import URLFetcher
    return URLFetcher(allowed_protocols={"data"}, fail_on_errors=True, allow_redirects=False)


def render_pdf(html: str, css: str = "") -> bytes:
    if not html or not html.strip():
        raise ExportRefused("there is no page to export")
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise ExportRefused("the page is too large to export")
    from weasyprint import CSS, HTML
    from weasyprint.urls import FatalURLFetchingError
    fetcher = _data_only_fetcher()
    try:
        sheets = [CSS(string=css, url_fetcher=fetcher)] if css and css.strip() else []
        return HTML(string=html, url_fetcher=fetcher).write_pdf(stylesheets=sheets)
    except FatalURLFetchingError as exc:
        raise ExportRefused(f"the page refers to an outside resource, which is not fetched: {exc}") from exc


def render_xlsx(tables: List[Dict[str, Any]]) -> bytes:
    """One sheet per table: ``{title, columns: [..], rows: [[..], ..]}``."""
    if not tables:
        raise ExportRefused("there are no tables to export")
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    used = set()
    for i, t in enumerate(tables, 1):
        cols, rows = t.get("columns") or [], t.get("rows") or []
        if len(rows) > MAX_TABLE_ROWS:
            raise ExportRefused(f"table {i} has more than {MAX_TABLE_ROWS} rows")
        name = re.sub(r"[\[\]:*?/\\]", " ", str(t.get("title") or f"Table {i}")).strip()[:31] or f"Table {i}"
        base, n = name, 2
        while name.lower() in used:
            name = f"{base[:28]} {n}"; n += 1
        used.add(name.lower())
        ws = wb.create_sheet(name)
        ws.append([_cell(c) for c in cols])
        for r in rows:
            ws.append([_cell(v) for v in r])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _cell(v: Any) -> Any:
    # A cell starting with = + - @ is a formula in Excel; a report value must stay text.
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
        return "'" + v
    return v if isinstance(v, (int, float, bool)) or v is None else str(v)


def storage_key(report_id: Optional[str], name: str, fmt: str, stamp: str) -> str:
    return f"reports/exports/{_SAFE.sub('_', str(report_id or 'adhoc'))}/{_SAFE.sub('_', name)[:80]}_{stamp}.{fmt}"


def store(content: bytes, key: str, content_type: str) -> bool:
    """Keep a copy in S3. False (not an exception) when it cannot: the download still works."""
    try:
        from src.services.egress import Purpose, aws_client
        bucket = os.getenv("S3_BUCKET_NAME", _BUCKET_DEFAULT)
        aws_client("s3", purpose=Purpose.OBJECT_STORAGE).put_object(
            Bucket=bucket, Key=key, Body=content, ContentType=content_type)
        return True
    except Exception:
        logger.warning("report export: could not store %s", key, exc_info=True)
        return False


def record_url(report_id: str, key: str) -> bool:
    """Write the S3 key to proc.bp_reports.report_url (a column nothing wrote until now)."""
    try:
        from src.services.db import get_conn
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_reports SET report_url = %s WHERE id::text = %s",
                        (f"s3://{os.getenv('S3_BUCKET_NAME', _BUCKET_DEFAULT)}/{key}", str(report_id)))
            return cur.rowcount > 0
    except Exception:
        logger.warning("report export: could not record report_url for %s", report_id, exc_info=True)
        return False
