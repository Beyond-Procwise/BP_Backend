from src.services.structural_extractor.parsing.model import ParsedDocument, BBox
from src.services.structural_extractor.discovery.schema import FieldType, type_of, fields_for
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label
from src.services.structural_extractor.types import ExtractedValue


def _nearby_label(cand, all_tokens) -> str:
    """Find a label near the candidate — same line OR one line above (left-aligned).
    Falls back to inferred_label if the candidate is not BBox-anchored.
    """
    if not cand.tokens:
        return ""
    first = cand.tokens[0]
    anchor = first.anchor
    if not isinstance(anchor, BBox):
        return inferred_label(cand, all_tokens)
    same_line = inferred_label(cand, all_tokens)
    # Also collect tokens immediately above (within 25pt vertically) AND horizontally overlapping
    above: list = []
    cand_cy = (anchor.y0 + anchor.y1) / 2
    for t in all_tokens:
        if not isinstance(t.anchor, BBox) or t.anchor.page != anchor.page:
            continue
        t_cy = (t.anchor.y0 + t.anchor.y1) / 2
        # Token center must be above candidate center, and within 30pt
        if t_cy >= cand_cy:
            continue
        if cand_cy - t_cy > 30:
            continue
        # Horizontal overlap with candidate
        if t.anchor.x1 < anchor.x0 - 10 or t.anchor.x0 > anchor.x1 + 80:
            continue
        above.append(t)
    above.sort(key=lambda t: (t.anchor.y0, t.anchor.x0))
    above_text = " ".join(t.text for t in above[-5:])
    return (same_line + " " + above_text).strip()


def extract_ids(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    id_fields = [f for f in fields_for(doc_type) if type_of(doc_type, f) == FieldType.ID]
    cands = find_candidates(doc, FieldType.ID)
    for field in id_fields:
        field_key = field.replace("_id", "").replace("_", " ").lower()
        scored: list = []
        for c in cands:
            lbl = _nearby_label(c, doc.tokens).lower()
            overlap = sum(1 for w in field_key.split() if w in lbl)
            if overlap > 0:
                scored.append((overlap, c))
        if scored:
            scored.sort(key=lambda s: -s[0])
            best = scored[0][1]
            out[field] = ExtractedValue(
                value=best.text, provenance="extracted",
                anchor_text=best.text, anchor_ref=best.tokens[0].anchor,
                source="structural", confidence=1.0, attempt=1,
            )
    return out
