from src.services.structural_extractor.parsing.model import (
    ParsedDocument, Token, BBox, CellRef, ColumnRef, NodeRef
)
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.schema import FieldType

MIN_COLUMNS = 2

# Exclusion vocabulary for summary-row detection. These are NOT used to
# *extract* labels — they are used to structurally reject rows that are
# visually at the bottom of a table and carry a summary function
# (subtotal / tax / grand total / balance). A different concept from a
# "label vocabulary for extraction" because it's a rejection filter
# applied only when we've already identified a row as structurally
# ambiguous (e.g., very few words + a money token).
SUMMARY_VOCAB = {
    "subtotal", "sub-total", "sub total", "total", "tax", "vat",
    "discount", "balance", "amount due", "grand total", "amount",
}


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


def _ev_derived(value, rule_id: str, inputs: dict):
    return ExtractedValue(
        value=value, provenance="derived",
        derivation_trace={"rule_id": rule_id, "inputs": inputs},
        source="derivation_registry", confidence=1.0, attempt=1,
    )


def _is_summary_row(cells_text: str, n_money: int) -> bool:
    """Return True when this row looks like a summary row rather than a data row.

    Two heuristics:
    1. The description text (lowercased, stripped of punctuation) is in
       SUMMARY_VOCAB.
    2. The description column has fewer than 3 alphabetic words AND the
       row contains at least one MONEY token — classic summary pattern
       ("Tax (20%): £479.94", "Sub-total: £2,399.70").
    """
    if not cells_text:
        return False
    clean = cells_text.strip().lower().rstrip(":;,.")
    # Strip trailing punctuation inside tokens
    clean = clean.replace(":", "").strip()
    if clean in SUMMARY_VOCAB:
        return True
    # Also catch "tax (20%)" style
    for vocab in SUMMARY_VOCAB:
        if clean.startswith(vocab + " ") or clean.startswith(vocab + "("):
            return True
    # Fewer than 3 word-like tokens AND has money → likely summary
    words = [w for w in clean.split() if any(ch.isalpha() for ch in w)]
    if n_money >= 1 and len(words) < 3:
        # Additionally require a vocab hit anywhere in the words
        for vocab in SUMMARY_VOCAB:
            for w in words:
                if vocab in w or w in vocab:
                    return True
    return False


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

    # Find header row: all-TEXT line followed within 3 rows by a MONEY-bearing
    # row AND whose tokens look like column headers (short, keyword-like).
    # Addresses like 'WEST SUSSEX, RH13 5QH MANCHESTER, M1 7HY' look all-TEXT
    # and precede the line-item MONEY rows, but they are NOT headers — we
    # reject them via a 'looks like a header row' heuristic.
    # Common English function-words: a header is a column-label row, not
    # prose. Prose lines have connectors like 'and', 'for', 'of'; headers
    # don't. This filter separates 'Item Description Total' (header) from
    # 'Ongoing content creation and revisions' (prose bullet).
    _STOPWORDS = {
        "and", "or", "but", "the", "of", "to", "for", "from", "in", "on",
        "at", "by", "with", "as", "a", "an", "is", "are", "was", "were",
        "be", "been", "being", "has", "have", "had", "our", "their",
        "this", "that", "these", "those",
    }

    def _row_looks_like_header(line_toks: list[Token]) -> bool:
        # Token check: headers have 2-8 tokens; reject very long address-like rows.
        if not (2 <= len(line_toks) <= 8):
            return False
        # Reject rows containing digit tokens (addresses have postcodes/numbers).
        for t in line_toks:
            clean = t.text.strip(".,:;")
            if any(ch.isdigit() for ch in clean):
                return False
        # Reject rows where any token is > 14 chars (long descriptions).
        if any(len(t.text.strip(".,:;")) > 14 for t in line_toks):
            return False
        # Reject rows containing English function-words (prose, not headers).
        for t in line_toks:
            if t.text.strip(".,:;").lower() in _STOPWORDS:
                return False
        # Reject rows where any token ends with a comma — these are
        # prose/address runs like "Redkiln Way, Horsham, West Sussex".
        # A header column label never ends with a comma.
        for t in line_toks:
            if t.text.rstrip().endswith(","):
                return False
        # Reject rows where the MAJORITY of tokens start with a lowercase
        # letter — headers use Title Case or ALL CAPS.
        lower_starts = sum(
            1 for t in line_toks
            if t.text[:1].isalpha() and t.text[:1].islower()
        )
        if lower_starts > len(line_toks) // 2:
            return False
        # Column-structure check: a real header has its tokens positioned at
        # column midpoints, so there is at least one LARGE horizontal gap
        # between consecutive tokens (>=50pt). Prose/address rows have tight
        # word-spacing (<=5pt) throughout.
        toks_sorted = sorted(
            line_toks,
            key=lambda t: t.anchor.x0 if isinstance(t.anchor, BBox) else 0,
        )
        max_gap = 0.0
        for i in range(1, len(toks_sorted)):
            a = toks_sorted[i - 1].anchor
            b = toks_sorted[i].anchor
            if isinstance(a, BBox) and isinstance(b, BBox):
                gap = b.x0 - a.x1
                if gap > max_gap:
                    max_gap = gap
        if max_gap < 50:
            return False
        return True

    header_key = None
    for idx, k in enumerate(ordered_keys):
        line_toks = [t for t in lines[k] if t.text.strip()]
        if len(line_toks) < MIN_COLUMNS:
            continue
        types = {_classify(t) for t in line_toks}
        if types != {"TEXT"}:
            continue
        if not _row_looks_like_header(line_toks):
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

    # Stop row = first row after header that is either:
    #   (a) more MONEY tokens than columns (classic summary row), OR
    #   (b) identified by _is_summary_row on the description text
    stop_key = None
    for k in ordered_keys:
        if k <= header_key:
            continue
        row_line_toks = [t for t in lines[k] if t.text.strip()]
        row_types = [_classify(t) for t in row_line_toks]
        n_money = row_types.count("MONEY")
        if n_money > n_cols:
            stop_key = k
            break
        # Early summary detection: combine text tokens and check vocab
        text_part = " ".join(t.text for t in row_line_toks if _classify(t) == "TEXT")
        if _is_summary_row(text_part, n_money):
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

        # Assign column roles by type signature.
        # Priority: if a cell contains ANY MONEY token treat it as MONEY.
        # Then PERCENT, then INTEGER, else modal type. This prevents a
        # description continuation word ("Base") from downgrading a
        # MONEY-bearing cell to TEXT.
        col_types: dict[int, str] = {}
        for col_idx, col_toks in cells.items():
            if not col_toks:
                col_types[col_idx] = "EMPTY"
                continue
            types_here = [_classify(t) for t in col_toks]
            if "MONEY" in types_here:
                col_types[col_idx] = "MONEY"
            elif "PERCENT" in types_here:
                col_types[col_idx] = "PERCENT"
            elif "INTEGER" in types_here:
                col_types[col_idx] = "INTEGER"
            else:
                col_types[col_idx] = "TEXT"

        text_cols = [c for c, t in col_types.items() if t == "TEXT"]
        int_cols = [c for c, t in col_types.items() if t == "INTEGER"]
        money_cols = [c for c, t in col_types.items() if t == "MONEY"]

        desc_col = max(text_cols, key=lambda c: sum(len(t.text) for t in cells[c]), default=None)
        qty_col = int_cols[0] if int_cols else None
        # When a cell is classified MONEY, only keep MONEY tokens for _parse_num
        # — otherwise stray text continuations bleed into the number.
        def _money_toks(col):
            return [t for t in cells[col] if _classify(t) == "MONEY"]

        # Description: concatenate all TEXT tokens from all text-type cells
        # in reading order (left-to-right, top-to-bottom). A data row in
        # practice often spans multiple header columns (especially in
        # two-column 'Item | Description' layouts where the service title
        # flows across both). Taking only the longest-text column misses
        # the leading word ('Monthly' in ELEANOR's 'Monthly Design &
        # Marketing Package'). All MONEY/INTEGER tokens are excluded —
        # those belong to qty / price / total columns.
        desc_tokens_all: list[Token] = []
        for col_idx in text_cols:
            for t in cells[col_idx]:
                if _classify(t) == "TEXT":
                    desc_tokens_all.append(t)
        # Sort by (y0, x0) for natural reading order — preserves sub-lines.
        desc_tokens_all.sort(
            key=lambda t: (
                (t.anchor.y0, t.anchor.x0) if isinstance(t.anchor, BBox) else (0, 0)
            )
        )
        desc = " ".join(t.text for t in desc_tokens_all)
        # Back-compat: if desc_col was set but the new logic produced an
        # empty string, fall back to the old behavior (defensive — should
        # never fire in practice).
        if not desc and desc_col is not None:
            fallback_toks = [t for t in cells[desc_col] if _classify(t) == "TEXT"]
            desc = " ".join(t.text for t in fallback_toks) if fallback_toks else (
                " ".join(t.text for t in cells[desc_col])
            )
        qty_val = _parse_num(cells[qty_col]) if qty_col is not None else None

        price_col, total_col = None, None
        if len(money_cols) >= 2 and qty_val:
            a = _parse_num(_money_toks(money_cols[0]))
            b = _parse_num(_money_toks(money_cols[1]))
            if a is not None and b is not None:
                if abs(qty_val * a - b) < 0.01:
                    price_col, total_col = money_cols[0], money_cols[1]
                elif abs(qty_val * b - a) < 0.01:
                    price_col, total_col = money_cols[1], money_cols[0]
                else:
                    price_col, total_col = money_cols[0], money_cols[-1]
        elif len(money_cols) == 1:
            total_col = money_cols[0]

        price_val = _parse_num(_money_toks(price_col)) if price_col is not None else None
        total_val = _parse_num(_money_toks(total_col)) if total_col is not None else None

        # Safety net: skip rows whose description text matches the summary
        # vocabulary. Prevents "Sub-total:", "Discount:", "Tax (20%):",
        # "Total:" etc. from being emitted as data rows.
        n_money_here = sum(1 for c, t in col_types.items() if t == "MONEY" and cells[c])
        if _is_summary_row(desc, n_money_here):
            continue

        # F1 — Implicit quantity: single MONEY token + no INTEGER → qty=1
        # and unit_price=line_total=MONEY (derived). This is a universal
        # invoice convention: "Service £X" means one unit at price X.
        if qty_val is None and price_val is None and total_val is not None and (qty_col is None):
            # Derive: qty=1, unit_price=line_total
            if desc:
                key = (desc, 1, total_val, total_val)
                if key in seen:
                    continue
                seen.add(key)
                item: dict[str, ExtractedValue] = {
                    "line_no": _ev_derived(line_no, "line_no_monotonic", {}),
                    "item_description": ExtractedValue(
                        value=desc, provenance="extracted", anchor_text=desc,
                        anchor_ref=cells[desc_col][0].anchor if cells.get(desc_col) else None,
                        source="structural", confidence=1.0, attempt=1,
                    ),
                    "quantity": _ev_derived(
                        1, "implicit_single_qty",
                        {"reason": "single-money row with no qty column"},
                    ),
                    "unit_price": _ev_extracted(total_val, _money_toks(total_col)[0]),
                    "line_total": _ev_extracted(total_val, _money_toks(total_col)[0]),
                }
                items.append(item)
                line_no += 1
            continue

        # F1 — qty + single money: derive line_total = qty * unit_price
        if qty_val is not None and price_val is None and total_val is not None:
            # Money cell exists but couldn't be aligned as price/total. Keep
            # as total and skip unit_price (existing behavior).
            pass
        if qty_val is not None and len(money_cols) == 1 and total_val is not None and price_val is None:
            # Single money column — treat as unit_price, derive total.
            price_val = total_val
            derived_total = round(qty_val * price_val, 2)
            # Overwrite total_val to derived form
            total_val = derived_total

        if desc and (qty_val is not None or total_val is not None):
            key = (desc, qty_val, price_val, total_val)
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, ExtractedValue] = {
                "line_no": _ev_derived(line_no, "line_no_monotonic", {}),
                "item_description": ExtractedValue(
                    value=desc, provenance="extracted", anchor_text=desc,
                    anchor_ref=cells[desc_col][0].anchor if cells.get(desc_col) else None,
                    source="structural", confidence=1.0, attempt=1,
                ),
            }
            if qty_val is not None:
                item["quantity"] = _ev_extracted(qty_val, cells[qty_col][0])
            if price_val is not None:
                if price_col is not None:
                    item["unit_price"] = _ev_extracted(price_val, _money_toks(price_col)[0])
                else:
                    item["unit_price"] = _ev_derived(
                        price_val, "single_money_as_unit_price",
                        {"line_total": total_val, "quantity": qty_val},
                    )
            if total_val is not None:
                if total_col is not None and _money_toks(total_col):
                    # If total was derived from qty*price, use derived tag
                    raw_total = _parse_num(_money_toks(total_col))
                    if raw_total is not None and abs(raw_total - total_val) < 0.01:
                        item["line_total"] = _ev_extracted(total_val, _money_toks(total_col)[0])
                    else:
                        item["line_total"] = _ev_derived(
                            total_val, "qty_times_unit_price",
                            {"quantity": qty_val, "unit_price": price_val},
                        )
                else:
                    item["line_total"] = _ev_derived(
                        total_val, "qty_times_unit_price",
                        {"quantity": qty_val, "unit_price": price_val},
                    )
            items.append(item)
            line_no += 1
    return items
