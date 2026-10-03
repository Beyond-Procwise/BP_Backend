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


def layout_status(conn, layout_id: str) -> str | None:
    """The layout's current status, or None if there is no such layout.

    Read before REJECTING one: rejecting a candidate is triage, and rejecting something a
    human already approved withdraws that approval — two different authorities (atb router).
    """
    cursor = conn.cursor()
    cursor.execute('SELECT status FROM proc.bp_page_layout WHERE layout_id = %s', (layout_id,))
    found = cursor.fetchone()
    return str(found[0]) if found else None


def rename_layout(conn, layout_id: str, name: str, user: str) -> int:
    """-> rows changed, so a rename of something that does not exist is not a 200."""
    cursor = conn.cursor()
    cursor.execute('UPDATE proc.bp_page_layout SET name = %s WHERE layout_id = %s',
                   (name, layout_id))
    return int(cursor.rowcount or 0)


def set_layout_status(conn, layout_id: str, status: str, user: str) -> int:
    """-> rows changed, so the caller can answer 404 rather than 200 for an id that is not there.

    Guarded on the layout's pack NOT being `importing`: every read path excludes an importing
    pack, and an unguarded UPDATE walked straight around that filter.
    """
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE proc.bp_page_layout l SET status = %s, approved_by = %s, approved_at = now() '
        'WHERE l.layout_id = %s AND EXISTS (SELECT 1 FROM proc.bp_style_pack p '
        "  WHERE p.pack_id = l.pack_id AND p.status <> 'importing')",
        (status, user, layout_id))
    return int(cursor.rowcount or 0)


def set_pack_status(conn, pack_id: str, status: str, user: str) -> int:
    """-> rows changed. An `importing` pack is not approvable: it has no layouts yet, and
    approving it would publish an incomplete import past the filter the invariant rests on."""
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE proc.bp_style_pack SET status = %s, approved_by = %s, approved_at = now() '
        "WHERE pack_id = %s AND status <> 'importing'", (status, user, pack_id))
    return int(cursor.rowcount or 0)


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


def promote_column(conn, *, pack_id: str, layout_key: str, column: str, scale: str) -> int:
    """Turn a text column into a rating column on a named scale. -> rows changed.

    The endpoint used to validate the promotion and then throw it away, returning 200 as though it
    had happened. The slots blob is read, mutated and written back, because a jsonb_set path into
    an ARRAY element needs the index, and the index is what we are searching for.
    """
    cursor = conn.cursor()
    cursor.execute('SELECT layout_id, slots FROM proc.bp_page_layout '
                   'WHERE pack_id = %s AND layout_key = %s', (pack_id, layout_key))
    found = cursor.fetchall()
    if not found:
        return 0
    layout_id, slots = found[0]
    slots = slots or {}
    changed = False
    for slot in slots.values():
        if not isinstance(slot, dict) or slot.get('type') != 'table':
            continue
        for definition in slot.get('columns') or []:
            if isinstance(definition, dict) and definition.get('id') == column:
                definition['type'] = 'rating'
                definition['scale'] = scale
                definition.pop('max_chars', None)
                changed = True
    if not changed:
        return 0
    cursor.execute('UPDATE proc.bp_page_layout SET slots = %s::jsonb WHERE layout_id = %s',
                   (_js(slots), layout_id))
    return int(cursor.rowcount or 0)


def inherited(conn, pack_key: str) -> dict:
    """What a human decided about the previous version of this key."""
    empty: dict = {'names': {}, 'rating_scales': {}, 'locale': None, 'rejected': []}
    cursor = conn.cursor()
    cursor.execute(
        # `importing` is excluded as well as `rejected`: a crash leaves a pack row with no
        # layouts, and inheriting from it threw away every name and hand-defined scale from the
        # last good version — the one part of this the spec says nobody can automate.
        'SELECT pack_id, tokens, evidence FROM proc.bp_style_pack '
        "WHERE pack_key = %s AND status NOT IN ('rejected', 'importing') "
        'ORDER BY version DESC LIMIT 1',
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
