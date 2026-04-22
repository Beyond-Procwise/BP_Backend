import json
from dataclasses import asdict, is_dataclass

from src.services.structural_extractor.types import ExtractedValue


def _ref_to_json(ref):
    if ref is None:
        return None
    if is_dataclass(ref):
        d = asdict(ref)
        d["_type"] = type(ref).__name__
        return json.dumps(d)
    return json.dumps(ref)


def write_provenance(conn, parent_table: str, parent_pk: str,
                     fields: dict[str, ExtractedValue]) -> int:
    count = 0
    with conn.cursor() as cur:
        for field_name, ev in fields.items():
            if ev is None or ev.value is None:
                continue
            cur.execute(
                "INSERT INTO proc.bp_extraction_provenance "
                "(parent_table, parent_pk, field_name, source, anchor_ref, derivation_trace, confidence, attempt) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)",
                (
                    parent_table, str(parent_pk), field_name, ev.provenance,
                    _ref_to_json(ev.anchor_ref),
                    json.dumps(ev.derivation_trace) if ev.derivation_trace else None,
                    ev.confidence, ev.attempt,
                ),
            )
            count += 1
    return count
