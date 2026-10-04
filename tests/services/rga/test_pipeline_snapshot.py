"""The snapshot is drawn beside the deck, post-checked with it, and released only with it."""
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.services.rga import postcheck
from src.services.rga.layout_fit import Catalogue, LayoutOption, NoPack, SlotOption
from src.services.rga.models import (Confidence, FactEntry, FactPack, Finding, FindingCode,
                                     FormatHint, NarrativeBlock, Origin, ReportAST, Section,
                                     Severity)
from src.services.rga.pipeline import generate_report
from src.services.rga.render import RenderedArtefact

_TYPE = 'exec_procurement_summary'

LAYOUTS = (LayoutOption(layout_key='t_prose', name='Title + panel', kind='template',
                        slots=(SlotOption(name='title', type='text', fill='agent', max_words=19),
                               SlotOption(name='prose1', type='text', fill='agent',
                                          max_chars=240)),
                        region_ids=('title', 'prose1')),)
CAT = Catalogue(pack_key='deck-x', pack_id='p-1', format_kind='deck', layouts=LAYOUTS)

AST = ReportAST(sections=[Section(id='s1', title='What we found',
                                  blocks=[NarrativeBlock(text='Spend is concentrated.')])])

PACK = FactPack(pack_id='pk-1', report_type_id=_TYPE, scope={}, as_of='2026-09-30',
                generated_at='2026-10-04T00:00:00Z', generated_by='test',
                facts=[FactEntry(fact_id='F0001', label='Total spend', value=Decimal('240000'),
                                 currency='GBP', format_hint=FormatHint.MONEY_EXACT,
                                 confidence=Confidence.CORROBORATED, origin=Origin.OBSERVED,
                                 provenance_id='prov-1', derivation='sum of invoice lines')])


class _Drawer:
    """A renderer that draws nothing in particular, so these tests are about the WIRING."""

    def __init__(self, name):
        self.name = name

    def render(self, ast, pack, brief, *, title, **kw):
        return RenderedArtefact(content=b'x', media_type='application/octet-stream',
                                renderer=self.name, renderer_version='1', pack_id=pack.pack_id,
                                pack_hash=pack.hash, style_version=brief.version(),
                                ast_hash='h', style_provenance=dict(brief.provenance))


def _passing(*a, **k):
    return postcheck.PostCheckResult(findings=[])


def _run(**kw):
    with patch('src.services.rga.pipeline.build_fact_pack', return_value=PACK), \
         patch('src.services.rga.pipeline.postcheck.run', side_effect=_passing), \
         patch('src.services.rga.pipeline.catalogue', return_value=CAT), \
         patch('src.services.rga.pipeline.get_conn'):
        return generate_report(_TYPE, scope={'period': 'FY26'}, as_of='2026-09-30', ast=AST,
                               renderer=_Drawer('deck'), page_renderer=_Drawer('page'),
                               emit_audit=False, **kw)


def test_no_pack_key_means_no_snapshot_and_nothing_else_changes():
    run = _run()
    assert run.released is True
    assert run.snapshot is None


def test_a_pack_key_draws_a_snapshot_beside_the_deck():
    run = _run(pack_key='deck-x')
    assert run.released is True
    assert run.snapshot is not None
    assert run.snapshot['style'] == {'key': 'deck-x'}
    assert run.snapshot['rows'] and run.snapshot['blocks']


def test_the_snapshot_is_post_checked_like_the_other_two_drawings():
    """The design's own promise. A drawing the post-check never reads is one nothing stops from
    claiming an untraced figure — and this is the drawing a person then edits."""
    seen = []

    def record(artefact, *a, **k):
        seen.append(artefact.renderer)
        return postcheck.PostCheckResult(findings=[])

    with patch('src.services.rga.pipeline.build_fact_pack', return_value=PACK), \
         patch('src.services.rga.pipeline.postcheck.run', side_effect=record), \
         patch('src.services.rga.pipeline.catalogue', return_value=CAT), \
         patch('src.services.rga.pipeline.get_conn'):
        generate_report(_TYPE, scope={}, as_of='2026-09-30', ast=AST, renderer=_Drawer('deck'),
                        page_renderer=_Drawer('page'), emit_audit=False, pack_key='deck-x')
    assert 'snapshot' in seen


def test_a_snapshot_that_fails_its_post_check_is_withheld_and_the_deck_still_releases():
    """The deck passed its own checks; denying a valid deck because the builder drawing failed
    would punish the report for the new drawing's fault."""
    def selective(artefact, *a, **k):
        if artefact.renderer == 'snapshot':
            return postcheck.PostCheckResult(findings=[
                Finding(finding_id='pk-1-PC001', code=FindingCode.REPORT_UNTRACED_FIGURE,
                        severity=Severity.HIGH, detail='a figure nothing traces',
                        blocks_release=True)])
        return postcheck.PostCheckResult(findings=[])

    with patch('src.services.rga.pipeline.build_fact_pack', return_value=PACK), \
         patch('src.services.rga.pipeline.postcheck.run', side_effect=selective), \
         patch('src.services.rga.pipeline.catalogue', return_value=CAT), \
         patch('src.services.rga.pipeline.get_conn'):
        run = generate_report(_TYPE, scope={}, as_of='2026-09-30', ast=AST,
                              renderer=_Drawer('deck'), page_renderer=_Drawer('page'),
                              emit_audit=False, pack_key='deck-x')
    assert run.released is True
    assert run.snapshot is None
    assert any('pages' in (f.detail or '') for f in run.findings)


def test_a_pack_nobody_approved_is_a_finding_not_an_exception():
    with patch('src.services.rga.pipeline.build_fact_pack', return_value=PACK), \
         patch('src.services.rga.pipeline.postcheck.run', side_effect=_passing), \
         patch('src.services.rga.pipeline.catalogue', side_effect=NoPack('no approved pack')), \
         patch('src.services.rga.pipeline.get_conn'):
        run = generate_report(_TYPE, scope={}, as_of='2026-09-30', ast=AST,
                              renderer=_Drawer('deck'), page_renderer=_Drawer('page'),
                              emit_audit=False, pack_key='deck-nope')
    assert run.snapshot is None
    assert run.released is True                    # the deck is unaffected
    assert any('pack' in (f.detail or '').lower() for f in run.findings)


def test_a_section_no_layout_can_draw_names_itself_in_a_finding():
    """Review focus 1, at the pipeline's edge: the requester reads WHICH section had nowhere
    to go, not 'generation failed'."""
    long_title = ReportAST(sections=[Section(id='s_long', title=' '.join(['word'] * 40),
                                             blocks=[NarrativeBlock(text='x')])])
    with patch('src.services.rga.pipeline.build_fact_pack', return_value=PACK), \
         patch('src.services.rga.pipeline.postcheck.run', side_effect=_passing), \
         patch('src.services.rga.pipeline.catalogue', return_value=CAT), \
         patch('src.services.rga.pipeline.get_conn'):
        run = generate_report(_TYPE, scope={}, as_of='2026-09-30', ast=long_title,
                              renderer=_Drawer('deck'), page_renderer=_Drawer('page'),
                              emit_audit=False, pack_key='deck-x')
    assert run.snapshot is None
    assert any('s_long' in (f.detail or '') for f in run.findings)


def test_a_blocked_report_carries_no_snapshot():
    def blocking(artefact, *a, **k):
        return postcheck.PostCheckResult(findings=[
            Finding(finding_id='pk-1-PC001', code=FindingCode.REPORT_UNTRACED_FIGURE,
                    severity=Severity.HIGH, detail='a figure nothing traces',
                    blocks_release=True)])

    with patch('src.services.rga.pipeline.build_fact_pack', return_value=PACK), \
         patch('src.services.rga.pipeline.postcheck.run', side_effect=blocking), \
         patch('src.services.rga.pipeline.catalogue', return_value=CAT), \
         patch('src.services.rga.pipeline.get_conn'):
        run = generate_report(_TYPE, scope={}, as_of='2026-09-30', ast=AST,
                              renderer=_Drawer('deck'), page_renderer=_Drawer('page'),
                              emit_audit=False, pack_key='deck-x')
    assert run.released is False
    assert run.snapshot is None


def test_losing_the_builder_pages_does_not_fail_a_released_report():
    """Found while wiring the runner: an unguarded store_snapshot turned a RELEASED report into a
    failed one and emitted report.run_failed after report.released. The deck and the page are
    stored; the third drawing is the least of the three, and its loss is not the run's failure.

    run_job takes its store and its generate function, so this needs no patching of module state.
    """
    from src.services.rga import job_runner

    seen = []

    class _Store:
        def claim(self, job_id):
            return True

        def get(self, job_id):
            return {'job_id': job_id, 'report_type': _TYPE, 'scope': {}, 'as_of': '2026-09-30',
                    'requested_by': 'sub-1', 'entitlement': None, 'pack_key': 'deck-x'}

        def finish_released(self, job_id, **kw):
            seen.append('released')

        def store_snapshot(self, job_id, snapshot):
            raise RuntimeError('the column is not there')

        def finish_failed(self, job_id, error=None, **kw):
            seen.append('failed')

        def ask_signoff(self, *a, **k):
            seen.append('signoff asked')

    run = _run(pack_key='deck-x')
    assert run.snapshot is not None          # there WAS a drawing to lose

    job_runner.run_job('j-1', store=_Store(), generate=lambda *a, **k: run)

    assert 'released' in seen
    assert 'failed' not in seen
