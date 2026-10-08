"""Four types the vocabulary knows about but nothing may resolve to yet.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/concepts/test_contract_link_types.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import seed                                          # noqa: E402

NEW = {"doctype.dpa": "role.attachment", "doctype.side_letter": "role.variation",
       "doctype.renewal": "role.variation", "doctype.guaranty": "role.supporting"}


def test_the_four_types_are_seeded_proposed_with_no_pipeline():
    types = {t.concept_code: t for t in seed.DOCUMENT_TYPES.values()}
    for code, role in NEW.items():
        assert code in types and code in seed.CONCEPTS, code
        assert types[code].status == "proposed" and seed.CONCEPTS[code].status == "proposed"
        assert types[code].role == role and types[code].pipeline_doc_type is None


def test_a_proposed_type_never_routes_an_upload():
    """Review Focus 5: only status='active' rows route, whichever vocabulary is loaded."""
    import pytest
    from src.services.concepts import routing as R
    types = {t.concept_code: t for t in seed.DOCUMENT_TYPES.values()}
    for code in NEW:
        for alias in types[code].aliases:
            with pytest.raises(R.UnknownDocumentCategory):
                R.pipeline_for_category(alias)
