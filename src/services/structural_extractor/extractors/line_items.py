from src.services.structural_extractor.parsing.model import (
    ParsedDocument, Token, BBox, CellRef, ColumnRef, NodeRef
)
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.schema import FieldType

MIN_COLUMNS = 2


def extract_line_items(doc: ParsedDocument, doc_type: str) -> list[dict[str, ExtractedValue]]:
    if doc.source_format == "pdf":
        return _pdf_line_items(doc)
    # XLSX/DOCX/CSV branches will be added in Phase 15 Tasks 51a/b/c
    return []


def _parse_num(toks):
    if not toks:
        return None
    txt = "".join(t.text for t in toks).replace(",", "").replace("£", "").replace("$", "").replace("€", "")
    try:
        return float(txt)
    except ValueError:
        return None


def _ev_extracted(value, tok):
    return ExtractedValue(
        value=value, provenance="extracted", anchor_text=str(value),
        anchor_ref=tok.anchor, source="structural", confidence=1.0, attempt=1,
    )


def _pdf_line_items(doc: ParsedDocument) -> list[dict[str, ExtractedValue]]:
    """Type-driven line-item discovery. No hardcoded keyword sets.
    - Classify each token via type detectors (MONEY / PERCENT) or local check (INTEGER).
    - Header row = first line where non-empty tokens are ALL TEXT and
      subsequent rows contain MONEY tokens at consistent X-positions.
    - Data rows between header and the first summary row (> n_cols MONEY tokens).
    - Column roles assigned by TYPE signature + arithmetic fit (qty*price=total).
    """
    money_ids = {id(c.tokens[0]) for c in find_candidates(doc, FieldType.MONEY)}
    percent_ids = {id(c.tokens[0]) for c in find_candidates(doc, FieldType.PERCENT)}

    def _classify(tok: Token) -> str:
        if id(tok) in money_ids:
            return "MONEY"
        if id(tok) in percent_ids:
            return "PERCENT"
        try:
            int(tok.text.replace(",", ""))
            return "INTEGER"
        except ValueError:
            pass
        return "TEXT"

    # Group tokens into Y-bucketed lines
    lines: dict[tuple, list[Token]] = {}
    for t in doc.tokens:
        if isinstance(t.anchor, BBox):
            y_bucket = int(t.anchor.y0 / 4)
            lines.setdefault((t.anchor.page, y_bucket), []).append(t)
    if not lines:
        return []
    ordered_keys = sorted(lines.keys())

    # Find header row: all-TEXT line followed within 3 rows by a MONEY-bearing row
    header_key = None
    for idx, k in enumerate(ordered_keys):
        line_toks = [t for t in lines[k] if t.text.strip()]
        if len(line_toks) < MIN_COLUMNS:
            continue
        types = {_classify(t) for t in line_toks}
        if types != {"TEXT"}:
            continue
        next_rows = [lines[kk] for kk in ordered_keys[idx + 1:idx + 4]]
        if any(any(_classify(t) == "MONEY" for t in row) for row in next_rows):
            header_key = k
            break
    if header_key is None:
        return []

    header_toks = sorted([t for t in lines[header_key] if t.text.strip()], key=lambda t: t.anchor.x0)
    columns = [(i, (t.anchor.x0 + t.anchor.x1) / 2) for i, t in enumerate(header_toks)]
    n_cols = len(columns)

    # Stop row = first row after header with more MONEY tokens than header has columns (likely summary)
    stop_key = None
    for k in ordered_keys:
        if k <= header_key:
            continue
        row_types = [_classify(t) for t in lines[k]]
        if row_types.count("MONEY") > n_cols:
            stop_key = k
            break

    # Collect data rows, merging adjacent buckets into logical rows.
    # A "logical row" begins at a MONEY-bearing line and includes text-only
    # lines within 4 buckets above/below (description + sub-description).
    data_bucket_keys = [k for k in ordered_keys if k > header_key and k[0] == header_key[0] and (not stop_key or k < stop_key)]

    def _has_money(k):
        return any(_classify(t) == "MONEY" for t in lines[k])

    # Group buckets: anchor each group at a money-bearing bucket, collect
    # neighbouring text-only buckets within +/- 4.
    groups: list[list[tuple]] = []
    used: set = set()
    for k in data_bucket_keys:
        if k in used:
            continue
        if _has_money(k):
            group = [k]
            used.add(k)
            # Look backward for nearby text-only buckets
            for prev in reversed(data_bucket_keys):
                if prev >= k:
                    continue
                if prev in used:
                    break
                if k[1] - prev[1] > 4:
                    break
                if _has_money(prev):
                    break
                group.insert(0, prev)
                used.add(prev)
            # Look forward for nearby text-only buckets
            for nxt in data_bucket_keys:
                if nxt <= k:
                    continue
                if nxt in used:
                    break
                if nxt[1] - k[1] > 4:
                    break
                if _has_money(nxt):
                    break
                group.append(nxt)
                used.add(nxt)
            groups.append(group)

    # Build row_tokens keyed by group index
    row_tokens: dict[int, list[Token]] = {}
    for gi, group in enumerate(groups):
        for k in group:
            row_tokens.setdefault(gi, []).extend(lines[k])

    items: list[dict[str, ExtractedValue]] = []
    seen = set()
    line_no = 1
    for y_bucket in sorted(row_tokens.keys()):
        toks = row_tokens[y_bucket]
        if not toks:
            continue
        cells: dict[int, list[Token]] = {col_idx: [] for col_idx, _ in columns}
        for t in toks:
            mid = (t.anchor.x0 + t.anchor.x1) / 2
            best = min(columns, key=lambda c: abs(c[1] - mid))
            cells[best[0]].append(t)

        # Assign column roles by type signature
        col_types: dict[int, str] = {}
        for col_idx, col_toks in cells.items():
            if not col_toks:
                col_types[col_idx] = "EMPTY"
                continue
            types_here = [_classify(t) for t in col_toks]
            col_types[col_idx] = max(set(types_here), key=types_here.count)

        text_cols = [c for c, t in col_types.items() if t == "TEXT"]
        int_cols = [c for c, t in col_types.items() if t == "INTEGER"]
        money_cols = [c for c, t in col_types.items() if t == "MONEY"]

        desc_col = max(text_cols, key=lambda c: sum(len(t.text) for t in cells[c]), default=None)
        qty_col = int_cols[0] if int_cols else None
        desc = " ".join(t.text for t in cells[desc_col]) if desc_col is not None else ""
        qty_val = _parse_num(cells[qty_col]) if qty_col is not None else None

        price_col, total_col = None, None
        if len(money_cols) >= 2 and qty_val:
            a = _parse_num(cells[money_cols[0]])
            b = _parse_num(cells[money_cols[1]])
            if a is not None and b is not None:
                if abs(qty_val * a - b) < 0.01:
                    price_col, total_col = money_cols[0], money_cols[1]
                elif abs(qty_val * b - a) < 0.01:
                    price_col, total_col = money_cols[1], money_cols[0]
                else:
                    price_col, total_col = money_cols[0], money_cols[-1]
        elif len(money_cols) == 1:
            total_col = money_cols[0]

        price_val = _parse_num(cells[price_col]) if price_col is not None else None
        total_val = _parse_num(cells[total_col]) if total_col is not None else None

        if desc and (qty_val is not None or total_val is not None):
            key = (desc, qty_val, price_val, total_val)
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, ExtractedValue] = {
                "line_no": ExtractedValue(
                    value=line_no, provenance="derived",
                    derivation_trace={"rule_id": "line_no_monotonic", "inputs": {}},
                    source="derivation_registry", confidence=1.0, attempt=1,
                ),
                "item_description": ExtractedValue(
                    value=desc, provenance="extracted", anchor_text=desc,
                    anchor_ref=cells[desc_col][0].anchor if cells.get(desc_col) else None,
                    source="structural", confidence=1.0, attempt=1,
                ),
            }
            if qty_val is not None:
                item["quantity"] = _ev_extracted(qty_val, cells[qty_col][0])
            if price_val is not None:
                item["unit_price"] = _ev_extracted(price_val, cells[price_col][0])
            if total_val is not None:
                item["line_total"] = _ev_extracted(total_val, cells[total_col][0])
            items.append(item)
            line_no += 1
    return items
