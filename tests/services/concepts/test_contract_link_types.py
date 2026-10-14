"""Four contract types, switched on 2026-10-11 and read through the contract pipeline.

They were seeded 'proposed' with no pipeline on 2026-10-08 and made title-only on
2026-10-09 (tests/services/extraction/test_title_only_types.py). With the title
guard in place they are active: a DPA, side letter, renewal agreement or guarantee
is a KIND of contract, so it ingests where a contract does.

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


def test_the_four_types_are_seeded_active_on_the_contract_pipeline():
    types = {t.concept_code: t for t in seed.DOCUMENT_TYPES.values()}
    for code, role in NEW.items():
        assert code in types and code in seed.CONCEPTS, code
        # Both halves: `status` lives on the concept AND the type, and a type whose
        # concept is still proposed is dropped by build_vocabulary.
        assert types[code].status == "active" and seed.CONCEPTS[code].status == "active"
        assert types[code].role == role and types[code].pipeline_doc_type == "contract"


def test_the_four_types_load_into_the_seed_vocabulary():
    from src.services.concepts.vocabulary import SEED_VOCABULARY
    for code in NEW:
        assert code in SEED_VOCABULARY.document_types, code
        assert code in SEED_VOCABULARY.concepts, code


def test_each_alias_routes_an_upload_to_the_contract_pipeline():
    from src.services.concepts import routing as R
    from src.services.concepts.vocabulary import SEED_VOCABULARY
    types = {t.concept_code: t for t in seed.DOCUMENT_TYPES.values()}
    for code in NEW:
        for alias in types[code].aliases:
            assert R.pipeline_for_category(alias, vocabulary=SEED_VOCABULARY) == \
                ("contract", code), alias


def test_activating_them_breaks_no_vocabulary_rule():
    from src.services.concepts import validate
    from src.services.concepts.vocabulary import SEED_DOC_TYPE_ROWS, SEED_VOCABULARY
    assert validate.run_all(SEED_VOCABULARY, SEED_DOC_TYPE_ROWS) == []


def test_policy_document_is_the_only_type_still_proposed():
    """Activation was for these four alone; nothing else rode along."""
    assert sorted(t.concept_code for t in seed.DOCUMENT_TYPES.values()
                  if t.status != "active") == ["doctype.policy_document"]
