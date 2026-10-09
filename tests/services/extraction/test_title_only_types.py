"""A title-only type is named by a document's own title, never by its clauses.

Measured 2026-10-09 with the four proposed contract types switched on: a contract
with no clean title ('THIS AGREEMENT is made on…') read as a DPA, a guaranty or a
side letter after just TWO body mentions of "DPA", "guarantee" or "side letter".
Ordinary clauses say those words all the time. The flag (proc.bp_document_type.
title_only) removes such a type from tier 2 (body volume) while leaving tier 1
(the title) alone.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_title_only_types.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import seed                                          # noqa: E402
from src.services.concepts.vocabulary import SEED_DOC_TYPE_ROWS, build_vocabulary  # noqa: E402
from src.services.extraction.type_resolver import resolve_document_type          # noqa: E402

FOUR = ("doctype.dpa", "doctype.side_letter", "doctype.renewal", "doctype.guaranty")

_CONCEPT_ROWS = [
    {"concept_code": c.concept_code, "domain": c.domain, "definition": c.definition,
     "not_to_be_confused_with": list(c.not_to_be_confused_with), "status": c.status,
     "rejection_reason": c.rejection_reason}
    for c in seed.CONCEPTS.values()
]


def _vocab(*, activate: bool, title_only: bool | None = None):
    """The seed vocabulary, optionally with the four proposed types switched on.

    ``title_only=None`` keeps the seed's own flag; True/False overrides it, so a
    test can show what the flag is holding back.
    """
    def doc(row):
        if row["concept_code"] not in FOUR or not activate:
            return row
        out = dict(row, status="active")
        if title_only is not None:
            out["title_only"] = title_only
        return out

    def con(row):
        return dict(row, status="active") if activate and row["concept_code"] in FOUR else row

    return build_vocabulary([con(c) for c in _CONCEPT_ROWS],
                            [doc(d) for d in SEED_DOC_TYPE_ROWS], source="test")


_UNTITLED = (
    "THIS AGREEMENT is made on 1 May 2026 between Acme Ltd and Beta plc.\n\n"
    "1. Services\n\nThe Supplier shall clean the offices.\n\n"
)
_BODY = {
    "doctype.dpa": "Processing follows the data processing agreement and the DPA.",
    "doctype.guaranty": "The Supplier does not guarantee results. A parent company guarantee is given.",
    "doctype.side_letter": "No side letter applies.",
    "doctype.renewal": "The renewal agreement terms apply on renewal.",
}


def _outcome(r):
    return (r.status, r.evidence_concept, r.agreement, r.candidates)


@pytest.mark.parametrize("code", FOUR)
@pytest.mark.parametrize("declared", [None, "doctype.contract_unspecified"])
def test_body_mentions_never_name_a_title_only_type(code, declared):
    text = _UNTITLED + "\n".join([_BODY[code]] * 5)
    without = resolve_document_type(declared_concept=declared, full_text=text,
                                    vocabulary=_vocab(activate=False))
    flagged = resolve_document_type(declared_concept=declared, full_text=text,
                                    vocabulary=_vocab(activate=True))
    assert _outcome(flagged) == _outcome(without)
    assert not any(e.concept_code in FOUR for e in flagged.evidence)


@pytest.mark.parametrize("code", ["doctype.dpa", "doctype.guaranty", "doctype.side_letter"])
def test_without_the_flag_the_same_page_is_misread(code):
    """The guard is load-bearing: unflagged, the body mentions win the page."""
    text = _UNTITLED + "\n".join([_BODY[code]] * 5)
    got = resolve_document_type(declared_concept=None, full_text=text,
                                vocabulary=_vocab(activate=True, title_only=False))
    assert code in got.candidates


@pytest.mark.parametrize("title, code", [
    ("DATA PROCESSING AGREEMENT", "doctype.dpa"),
    ("PARENT COMPANY GUARANTEE", "doctype.guaranty"),
    ("SIDE LETTER", "doctype.side_letter"),
    ("RENEWAL AGREEMENT", "doctype.renewal"),
])
def test_a_title_still_names_a_title_only_type(title, code):
    text = f"{title}\n\nIn relation to the Master Services Agreement MSA-2026-01."
    got = resolve_document_type(declared_concept=None, full_text=text,
                                vocabulary=_vocab(activate=True))
    assert (got.status, got.evidence_concept) == ("matched", code)
    assert any(e.concept_code == code and e.kind == "title_alias" for e in got.evidence)


def test_a_schedule_heading_after_the_real_title_does_not_name_it():
    text = ("MASTER SERVICES AGREEMENT\n\nBetween Acme Ltd and Beta plc.\n\n"
            "Schedule 3\n\nDATA PROCESSING AGREEMENT\n\nThe DPA applies.")
    got = resolve_document_type(declared_concept=None, full_text=text,
                                vocabulary=_vocab(activate=True))
    assert got.evidence_concept == "doctype.master_agreement"


def test_the_four_proposed_types_are_seeded_title_only():
    for code in FOUR:
        assert seed.DOCUMENT_TYPES[code].title_only is True, code


def test_no_active_type_is_title_only_yet():
    """Behaviour of every live type is unchanged by this flag's arrival."""
    assert [d.concept_code for d in seed.DOCUMENT_TYPES.values()
            if d.title_only and d.status == "active"] == []


def test_the_loader_reads_the_flag_and_a_missing_column_means_false():
    vocab = _vocab(activate=True)
    assert all(vocab.document_types[c].title_only for c in FOUR)
    rows = [dict(d) for d in SEED_DOC_TYPE_ROWS]
    for r in rows:
        r.pop("title_only", None)
    plain = build_vocabulary(_CONCEPT_ROWS, rows, source="test")
    assert not any(dt.title_only for dt in plain.document_types.values())
