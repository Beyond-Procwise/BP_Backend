"""The board paper against the real corpus: a deal with competing bids measures, draws and passes.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/rga/test_board_paper_live.py
"""
from __future__ import annotations

import os

import pytest

from src.services.rga import postcheck
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import MetricBlock, NarrativeBlock, ReportAST, Section, TableBlock
from src.services.rga.render import html as page_renderer
from src.services.rga.render import pptx as deck_renderer
from src.services.rga.style import resolve_style_brief

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

SCOPE = {"deal_id": "DEALV3-77", "period_label": "DEALV3-77", "category": "SaaS / IT"}


@pytest.fixture(scope="module")
def pack():
    import src.services.rga  # noqa: F401
    return build_fact_pack("board_paper", scope=SCOPE, as_of="2026-09-25", emit_audit=False)


def test_the_deal_is_measured(pack):
    by = {f.label: f for f in pack.facts}
    assert by["Quotes on file"].value == 3
    assert by["Suppliers who bid"].value >= 1
    assert any(f.label.startswith("Approval step — CFO sign-off") for f in pack.facts)
    # Every bidder is named, from the supplier register -- never shown as a raw id.
    assert not any(f.label.startswith("Bid from SUP-") for f in pack.facts)


def test_a_paper_naming_bidders_and_approvers_passes_its_checks(pack):
    steps = [f.fact_id for f in pack.facts if f.label.startswith("Approval step")]
    bids = [f.fact_id for f in pack.facts if f.label.startswith("Bid from ")]
    ast = ReportAST(sections=[
        Section(id="overview", title="Executive overview", blocks=[
            MetricBlock(fact_ref="F0001", emphasis="primary"),
            NarrativeBlock(text="The deal is worth {{F0001}}, carried by {{F0006}} documents.",
                           fact_refs=["F0001", "F0006"])]),
        Section(id="background", title="Background", blocks=[MetricBlock(fact_ref=b) for b in bids]),
        Section(id="approvals", title="Approvals", blocks=[
            TableBlock(columns=["Step", "Order"], rows=[["Approver", s] for s in steps]),
            MetricBlock(fact_ref="F0021")]),
    ])
    brief = resolve_style_brief("board_paper")
    for renderer in (deck_renderer, page_renderer):
        art = renderer.render(ast, pack, brief, title="Board paper")
        result = postcheck.run(art, pack, ast, brief, emit_audit=False)
        assert result.passed, (renderer.__name__, [f.detail for f in result.blocking])
