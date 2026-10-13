"""The logos a quote can wear: uploaded, kept, and swapped between.

Each upload is a new row; nothing is edited in place, so a quote that was sent with
one logo is not silently re-dressed when the company changes its mark. A logo is
chosen in two places: the company profile names the default (entity_profile.logoId),
and a quote's or template's letterhead section may name a different one.

Only PNG and JPEG are accepted, and only after the image library has actually decoded
them: the declared type is not believed, and SVG is refused outright because an SVG
is a document that can carry script onto a page a customer opens.
"""
from __future__ import annotations

import hashlib
import io
from typing import Any, Dict, List, Optional, Tuple

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor

MAX_BYTES = 1_000_000
MAX_SIDE = 4000
_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg"}
_COLS = ("logo_id", "logo_name", "content_type", "sha256", "size_bytes", "width", "height",
         "status", "created_by", "created_date")


def inspect(data: bytes) -> Tuple[str, int, int]:
    """(content type, width, height) of an image we will keep, or ValueError."""
    if not data:
        raise ValueError("the logo file is empty")
    if len(data) > MAX_BYTES:
        raise ValueError(f"a logo is at most {MAX_BYTES // 1000} KB")
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                       # the whole file, not just its header
        with Image.open(io.BytesIO(data)) as image:
            kind, (width, height) = image.format, image.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ValueError("that file is not a PNG or JPEG image") from None
    if kind not in _TYPES:
        raise ValueError("a logo must be a PNG or JPEG image")
    if width > MAX_SIDE or height > MAX_SIDE:
        raise ValueError(f"a logo is at most {MAX_SIDE} pixels on a side")
    return _TYPES[kind], width, height


def _meta(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": row["logo_id"], "name": row["logo_name"], "contentType": row["content_type"],
            "sizeBytes": row["size_bytes"], "width": row["width"], "height": row["height"],
            "status": row["status"], "uploadedBy": row["created_by"],
            "uploadedAt": row["created_date"], "imagePath": f"/entity/logos/{row['logo_id']}/image"}


def add(conn: Any, *, name: Optional[str], data: bytes, actor: Optional[str]) -> Dict[str, Any]:
    content_type, width, height = inspect(data)
    label = (name or "").strip()[:120] or "Logo"
    cur = dict_cursor(conn)
    try:
        cur.execute(
            "INSERT INTO proc.bp_brand_logo (logo_name, content_type, sha256, size_bytes, width, "
            f"height, image, created_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            f"RETURNING {', '.join(_COLS)}",
            (label, content_type, hashlib.sha256(data).hexdigest(), len(data), width, height,
             data, actor))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return _meta(row)


def list_logos(conn: Any) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(f"SELECT {', '.join(_COLS)} FROM proc.bp_brand_logo WHERE status = 'active' "
                "ORDER BY created_date DESC, logo_id DESC")
    return [_meta(dict(r)) for r in cur.fetchall()]


def exists(cur: Any, logo_id: int) -> bool:
    cur.execute("SELECT 1 FROM proc.bp_brand_logo WHERE logo_id = %s AND status = 'active'",
                (logo_id,))
    return cur.fetchone() is not None


def image(conn: Any, logo_id: int) -> Tuple[bytes, str]:
    """The picture itself. A removed logo is still served: quotes already sent wear it."""
    cur = dict_cursor(conn)
    cur.execute("SELECT image, content_type FROM proc.bp_brand_logo WHERE logo_id = %s", (logo_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"logo {logo_id} does not exist")
    return bytes(row["image"]), row["content_type"]


def remove(conn: Any, logo_id: int, *, default_logo_id: Optional[int]) -> None:
    """Take a logo out of the list it is chosen from. The company's current default is
    refused -- choose another first -- so the letterhead is never left pointing at nothing."""
    if default_logo_id is not None and int(default_logo_id) == int(logo_id):
        raise StateConflict("that is the company's current logo; choose another before "
                            "removing it")
    cur = dict_cursor(conn)
    try:
        cur.execute("UPDATE proc.bp_brand_logo SET status = 'removed' WHERE logo_id = %s "
                    "AND status = 'active' RETURNING logo_id", (logo_id,))
        if cur.fetchone() is None:
            raise NotFound(f"logo {logo_id} does not exist")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
