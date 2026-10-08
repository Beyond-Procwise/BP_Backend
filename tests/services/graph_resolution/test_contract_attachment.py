# tests/services/graph_resolution/test_contract_attachment.py
"""A schedule or SLA attaches to the agreement that cites it.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_attachment.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                    # noqa: E402
from src.services.graph_resolution import edge_writer                            # noqa: E402
from src.services.graph_resolution.profiles import contract_attachment as at     # noqa: E402


def _parent(**o):
    r = {"contract_id": "MSA-1", "contract_title": "Master Services Agreement Helix",
         "supplier_id": "S-1", "resolved_doc_type": "doctype.master_agreement",
         "contract_start_date": "2026-01-01", "contract_end_date": "2027-12-31"}
    r.update(o)
    return r


def _sla(**o):
    r = {"contract_id": "SLA-1", "contract_title": "Service Level Agreement Helix",
         "supplier_id": "S-1", "resolved_doc_type": "doctype.sla",
         "parent_agreement_ref": "MSA-1", "parent_contract_id": None, "framework_ref": None,
         "_ref_resolves": True,
         "contract_start_date": "2026-03-01", "contract_end_date": "2026-09-30"}
    r.update(o)
    return r


def test_registered_and_uncalibrated():
    assert at.PROFILE in le.PROFILES and at.PROFILE in edge_writer.UNCALIBRATED_PROFILES


def test_an_sla_naming_its_agreement_is_proposed_it():
    got = at.score(_sla(), _parent())
    assert got["F"] >= 80.0 and got["profile"] == "contract_attachment"


def test_an_sla_with_no_reference_and_unrelated_words_is_not():
    got = at.score(_sla(parent_agreement_ref=None, _ref_resolves=False,
                        contract_title="Service Levels"), _parent())
    assert got["F"] < 65.0


def test_the_title_signal_is_kept_for_attachments():
    ids = [s["id"] for s in at.score(_sla(), _parent())["signals"]]
    assert "title_overlap" in ids and "expected_structure" not in ids


def test_payment_terms_corroborate_an_attachment():
    plain = at.score(_sla(), _parent())["F"]
    both = at.score(_sla(payment_terms="Net 30"), _parent(payment_terms="Net 30"))["F"]
    assert both > plain
