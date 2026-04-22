from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.proximity import inferred_label, arithmetic_fit
from src.services.structural_extractor.types import ExtractedValue

_SUBTOTAL_FIELD = {"Invoice": "invoice_amount", "Purchase_Order": "total_amount", "Quote": "total_amount"}
_TOTAL_FIELD    = {"Invoice": "invoice_total_incl_tax", "Purchase_Order": "total_amount_incl_tax", "Quote": "total_amount_incl_tax"}


def _mk(c):
    return ExtractedValue(
        value=c.parsed_value, provenance="extracted",
        anchor_text=c.text, anchor_ref=c.tokens[0].anchor,
        source="structural", confidence=1.0, attempt=1,
    )


def extract_amounts(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    money_cands = find_candidates(doc, FieldType.MONEY)
    if not money_cands:
        return out
    # Truncate to top-40
    money_cands = money_cands[:40]
    subtotal_field = _SUBTOTAL_FIELD.get(doc_type)
    total_field = _TOTAL_FIELD.get(doc_type)

    best_triple = None
    for i, sub_c in enumerate(money_cands):
        sub_v = sub_c.parsed_value
        for j, tax_c in enumerate(money_cands):
            if i == j:
                continue
            tax_v = tax_c.parsed_value
            for k, tot_c in enumerate(money_cands):
                if k in (i, j):
                    continue
                tot_v = tot_c.parsed_value
                if arithmetic_fit(sub_v, tax_v, tot_v) == 1.0 and tot_v > sub_v > 0:
                    sub_lbl = inferred_label(sub_c, doc.tokens).lower()
                    tax_lbl = inferred_label(tax_c, doc.tokens).lower()
                    tot_lbl = inferred_label(tot_c, doc.tokens).lower()
                    score = (
                        ("sub" in sub_lbl or "net" in sub_lbl)
                        + ("tax" in tax_lbl or "vat" in tax_lbl)
                        + ("total" in tot_lbl or "amount due" in tot_lbl or "balance" in tot_lbl)
                    )
                    if best_triple is None or score > best_triple[0]:
                        best_triple = (score, sub_c, tax_c, tot_c)

    if best_triple:
        _, sub_c, tax_c, tot_c = best_triple
        if subtotal_field:
            out[subtotal_field] = _mk(sub_c)
        out["tax_amount"] = _mk(tax_c)
        if total_field:
            out[total_field] = _mk(tot_c)
        if sub_c.parsed_value > 0:
            pct = round(tax_c.parsed_value / sub_c.parsed_value * 100, 2)
            out["tax_percent"] = ExtractedValue(
                value=pct, provenance="derived",
                derivation_trace={"rule_id": "tax_pct_from_amounts",
                                  "inputs": {"tax_amount": tax_c.parsed_value,
                                             "subtotal": sub_c.parsed_value}},
                source="derivation_registry", confidence=1.0, attempt=1,
            )

    curr_cands = find_candidates(doc, FieldType.CURRENCY_CODE)
    if curr_cands:
        by_count: dict[str, int] = {}
        for c in curr_cands:
            by_count[c.parsed_value] = by_count.get(c.parsed_value, 0) + 1
        best_curr = max(by_count.items(), key=lambda kv: kv[1])[0]
        out["currency"] = ExtractedValue(
            value=best_curr, provenance="extracted",
            anchor_text=curr_cands[0].text, anchor_ref=curr_cands[0].tokens[0].anchor,
            source="structural", confidence=1.0, attempt=1,
        )
    return out
