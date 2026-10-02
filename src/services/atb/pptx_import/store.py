"""The only module here that touches the database.

THE WRITE ORDER MATTERS. `get_conn()` is AUTOCOMMIT, so `rollback()` is a no-op and an import
cannot be one unit of work. The sequence is therefore:

    insert_importing()  ->  insert_layout() per layout  ->  mark_candidate()

and **no read path serves an `importing` pack or its layouts.** A crash between the first and the
last write leaves something invisible rather than something half-valid, and a re-import of the
same file supersedes it with a new version.

A re-import also INHERITS what a human decided — layout names, hand-defined rating scales, the
locale, and rejections — because re-measuring everything would throw away the one part nobody can
automate (design §5b).
"""
from __future__ import annotations

import json
from typing import Any

_PACK_COLUMNS = ('pack_id', 'pack_key', 'version', 'source_file', 'source_sha256', 'slide_count',
                 'format', 'tokens', 'evidence', 'status', 'notes', 'created_at', 'created_by',
                 'approved_at', 'approved_by')
_LAYOUT_COLUMNS = ('layout_id', 'pack_id', 'layout_key', 'proposed_name', 'name', 'slide_refs',
                   'regions', 'slots', 'example_fill', 'example_source', 'problems', 'status',
                   'created_at', 'approved_at', 'approved_by')


def _rows(cursor, columns: tuple[str, ...]) -> list[dict]:
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _js(value: Any) -> str:
    return json.dumps(value, default=str)


def next_version(conn, pack_key: str) -> int:
    cursor = conn.cursor()
    cursor.execute('SELECT coalesce(max(version), 0) + 1 FROM proc.bp_style_pack '
                   'WHERE pack_key = %s', (pack_key,))
    return int(cursor.fetchone()[0])


def insert_importing(conn, *, pack_key: str, version: int, source_file: str, source_sha256: str,
                     slide_count: int, pack: dict, evidence: dict, user: str) -> str:
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO proc.bp_style_pack (pack_key, version, source_file, source_sha256, '
        '    slide_count, format, tokens, evidence, status, created_by) '
        "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, 'importing', %s) "
        'RETURNING pack_id',
        (pack_key, version, source_file, source_sha256, slide_count,
         _js(pack.get('format', {})), _js(pack), _js(evidence), user))
    return str(cursor.fetchone()[0])


def insert_layout(conn, *, pack_id: str, layout: dict) -> str:
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO proc.bp_page_layout (pack_id, layout_key, proposed_name, name, slide_refs, '
        '    regions, slots, example_fill, example_source, problems) '
        'VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb) '
        'RETURNING layout_id',
        (pack_id, layout['id'], layout.get('proposed_name') or layout.get('name') or '',
         layout.get('name'), list(layout.get('slide_refs') or []),
         _js(layout.get('regions', [])), _js(layout.get('slots', {})),
         _js(layout.get('example_fill', {})), _js(layout.get('example_source', {})),
         _js(layout.get('problems', []))))
    return str(cursor.fetchone()[0])


def mark_candidate(conn, pack_id: str) -> None:
    conn.cursor().execute(
        "UPDATE proc.bp_style_pack SET status = 'candidate' "
        "WHERE pack_id = %s AND status = 'importing'", (pack_id,))


def packs(conn, *, include_importing: bool = False) -> list[dict]:
    cursor = conn.cursor()
    where = '' if include_importing else "WHERE status <> 'importing'"
    cursor.execute(f'SELECT {", ".join(_PACK_COLUMNS)} FROM proc.bp_style_pack {where} '
                   'ORDER BY created_at DESC, version DESC')
    return _rows(cursor, _PACK_COLUMNS)


def pack(conn, pack_id: str) -> dict | None:
    cursor = conn.cursor()
    cursor.execute(f'SELECT {", ".join(_PACK_COLUMNS)} FROM proc.bp_style_pack '
                   "WHERE pack_id = %s AND status <> 'importing'", (pack_id,))
    found = _rows(cursor, _PACK_COLUMNS)
    return found[0] if found else None


def layouts(conn, *, pack_id: str | None = None, status: str | None = None) -> list[dict]:
    cursor = conn.cursor()
    clauses = ["p.status <> 'importing'"]
    params: list[Any] = []
    if pack_id:
        clauses.append('l.pack_id = %s')
        params.append(pack_id)
    if status:
        clauses.append('l.status = %s')
        params.append(status)
    columns = ', '.join(f'l.{c}' for c in _LAYOUT_COLUMNS)
    cursor.execute(f'SELECT {columns} FROM proc.bp_page_layout l '
                   'JOIN proc.bp_style_pack p ON p.pack_id = l.pack_id '
                   f'WHERE {" AND ".join(clauses)} ORDER BY l.created_at', tuple(params))
    return _rows(cursor, _LAYOUT_COLUMNS)


def rename_layout(conn, layout_id: str, name: str, user: str) -> None:
    conn.cursor().execute('UPDATE proc.bp_page_layout SET name = %s WHERE layout_id = %s',
                          (name, layout_id))


def set_layout_status(conn, layout_id: str, status: str, user: str) -> None:
    conn.cursor().execute(
        'UPDATE proc.bp_page_layout SET status = %s, approved_by = %s, approved_at = now() '
        'WHERE layout_id = %s', (status, user, layout_id))


def set_pack_status(conn, pack_id: str, status: str, user: str) -> None:
    conn.cursor().execute(
        'UPDATE proc.bp_style_pack SET status = %s, approved_by = %s, approved_at = now() '
        'WHERE pack_id = %s', (status, user, pack_id))


def define_rating_scale(conn, pack_id: str, name: str, chips: dict, user: str) -> None:
    """A hand-defined scale. Recorded as the human's, so §5b carries it to the next import and
    nobody mistakes it for a measurement."""
    cursor = conn.cursor()
    # ARRAY[...] rather than a concatenated '{a,b}' string: jsonb_set takes a text[] path, and the
    # string form needs a cast that psycopg cannot infer — and a scale name would be interpolated
    # into SQL rather than bound.
    cursor.execute(
        'UPDATE proc.bp_style_pack SET '
        "  tokens = jsonb_set(tokens, ARRAY['rating_scales', %s], %s::jsonb, true), "
        "  evidence = jsonb_set(evidence, ARRAY['values', 'rating_scales.' || %s], %s::jsonb, "
        '    true) '
        'WHERE pack_id = %s',
        (name, _js(chips), name,
         _js({'value': chips, 'defined_by': 'user', 'by': user,
              'why': 'defined by hand on the review screen; the deck gave no cell fills to '
                     'take chip colours from'}),
         pack_id))


def inherited(conn, pack_key: str) -> dict:
    """What a human decided about the previous version of this key."""
    empty: dict = {'names': {}, 'rating_scales': {}, 'locale': None, 'rejected': []}
    cursor = conn.cursor()
    cursor.execute(
        'SELECT pack_id, tokens, evidence FROM proc.bp_style_pack '
        "WHERE pack_key = %s AND status <> 'rejected' ORDER BY version DESC LIMIT 1",
        (pack_key,))
    found = cursor.fetchall()
    if not found:
        return empty
    pack_id, tokens, evidence = found[0]
    tokens = tokens or {}
    evidence = evidence or {}

    scales = {}
    for name, chips in (tokens.get('rating_scales') or {}).items():
        record = (evidence.get('values') or {}).get(f'rating_scales.{name}') or {}
        if record.get('defined_by') == 'user':
            scales[name] = chips

    locale_record = (evidence.get('values') or {}).get('writing.locale') or {}
    locale = (tokens.get('writing') or {}).get('locale') \
        if locale_record.get('defined_by') == 'user' else None

    cursor.execute('SELECT layout_key, name, status FROM proc.bp_page_layout WHERE pack_id = %s',
                   (pack_id,))
    names: dict[str, str] = {}
    rejected: list[str] = []
    for layout_key, name, status in cursor.fetchall():
        if status == 'rejected':
            rejected.append(layout_key)
        elif name:
            names[layout_key] = name
    return {'names': names, 'rating_scales': scales, 'locale': locale, 'rejected': rejected}
