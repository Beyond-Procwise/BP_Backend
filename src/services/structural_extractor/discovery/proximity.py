from src.services.structural_extractor.discovery.type_entities import Candidate
from src.services.structural_extractor.parsing.model import Token, BBox, CellRef, ColumnRef, NodeRef


def inferred_label(cand: Candidate, all_tokens: list[Token], max_lookback: int = 5) -> str:
    if not cand.tokens:
        return ""
    first = cand.tokens[0]
    anchor = first.anchor
    label_tokens: list[Token] = []
    if isinstance(anchor, BBox):
        line_y = (anchor.y0 + anchor.y1) / 2
        for t in all_tokens:
            if t.order >= first.order:
                break
            if not isinstance(t.anchor, BBox) or t.anchor.page != anchor.page:
                continue
            t_y = (t.anchor.y0 + t.anchor.y1) / 2
            if abs(t_y - line_y) <= 6 and t.anchor.x0 < anchor.x0:
                label_tokens.append(t)
        label_tokens = label_tokens[-max_lookback:]
    elif isinstance(anchor, CellRef):
        for t in all_tokens:
            if not isinstance(t.anchor, CellRef):
                continue
            if t.anchor.sheet == anchor.sheet and t.anchor.row == anchor.row and t.anchor.col < anchor.col:
                label_tokens.append(t)
        label_tokens = label_tokens[-1:] if label_tokens else []
    elif isinstance(anchor, ColumnRef):
        return anchor.column_name or ""
    elif isinstance(anchor, NodeRef):
        for t in all_tokens:
            if t.order >= first.order:
                break
            if isinstance(t.anchor, NodeRef):
                if (anchor.kind == "paragraph" and t.anchor.kind == "paragraph"
                        and t.anchor.paragraph_index == anchor.paragraph_index):
                    label_tokens.append(t)
                elif (anchor.kind == "table_cell" and t.anchor.kind == "table_cell"
                      and t.anchor.table_index == anchor.table_index
                      and t.anchor.row == anchor.row and t.anchor.col == anchor.col):
                    label_tokens.append(t)
        label_tokens = label_tokens[-max_lookback:]
    return " ".join(t.text for t in label_tokens)


def arithmetic_fit(subtotal: float, tax: float, total: float, tol: float = 0.01) -> float:
    return 1.0 if abs(subtotal + tax - total) <= tol else 0.0
