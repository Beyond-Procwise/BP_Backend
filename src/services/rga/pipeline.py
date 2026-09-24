"""The state machine. One way through, and it fails closed at the gate.

    SCOPE -> FACT_PACK -> STYLE_BRIEF -> COMPOSE -> RENDER -> POST_CHECK -> RELEASE
                                                                 |
                                                            fail -> FINDING + BLOCK

No agent loop, no tool use, no branching on a model's opinion. Each stage is a
function call in a fixed order, and the only stage a model participates in is
COMPOSE — which can also be skipped entirely by handing in a hand-written AST,
which is how Phase 1 ran and how the reproducibility path still runs.

WHY RELEASE IS SEPARATE FROM RENDER

Because rendering is cheap and reversible and releasing is neither. A blocked
report still produces an artefact — you need to be able to look at what failed —
but it is not released, and the run says so. Regeneration is a fresh run through
this same function; nothing here ever edits an artefact in place, and it could
not: ``FactPack`` and ``RenderedArtefact`` are both frozen.

WHAT IS NOT HERE YET

APPROVAL. §8 wants external-release report types gated on an approval matrix,
and no matrix exists (discovery §3.3). Rather than pretend, ``approval_required``
is reported as unknown and no approval event is emitted — an absent gate that
announces itself, instead of a gate that always says yes.

ENTITLEMENT is checked at the door, not here: ``report.generate`` is gated in
api/routers/reports.py when the job is filed, and the decision -- including
whether shadow mode suppressed a refusal (an audit row reading
``status=allowed`` can mean "denied, but shadowed") -- arrives through the
job's audit context and is recorded on report.scope_resolved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.services.rga import audit, postcheck
from src.services.rga.compose import CompositionError, compose_report
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import FactPack, Finding, FindingCode, ReportAST, Severity
from src.services.rga.render import RenderedArtefact
from src.services.rga.style import StyleBrief, resolve_style_brief

logger = logging.getLogger(__name__)

AGENT = "rga_pipeline"


@dataclass(frozen=True)
class ReportRun:
    """Everything one run produced, including the reasons it was stopped."""

    run_id: str
    report_type_id: str
    pack: Optional[FactPack] = None
    brief: Optional[StyleBrief] = None
    ast: Optional[ReportAST] = None
    artefact: Optional[RenderedArtefact] = None
    result: Optional[postcheck.PostCheckResult] = None
    released: bool = False
    stage_reached: str = "SCOPE"
    findings: List[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.released

    def blocking_reasons(self) -> List[str]:
        return [f"{f.code.value}: {f.detail}" for f in self.findings
                if f.blocks_release]


def generate_report(
    report_type_id: str,
    *,
    scope: Dict[str, Any],
    as_of: Optional[str] = None,
    ast: Optional[ReportAST] = None,
    generate: Optional[Callable[..., Optional[str]]] = None,
    template: Optional[str] = None,
    renderer: Any = None,
    title: str = "Executive procurement summary",
    emit_audit: bool = True,
    writer: Any = None,
) -> ReportRun:
    """Run one report end to end. Never raises for a report that merely failed.

    ``ast`` skips COMPOSE — the deterministic path. Passing one is how a stored
    run is regenerated without a model, and it is what makes the reproducibility
    guarantee testable.
    """
    if renderer is None:
        from src.services.rga.render import pptx as renderer

    def event(action_type: str, **kwargs) -> None:
        if emit_audit:
            audit.emit(action_type, agent=AGENT, writer=writer, **kwargs)

    findings: List[Finding] = []

    # -- SCOPE -------------------------------------------------------------
    # Entitlement was decided at the door; see the module docstring.
    pack = build_fact_pack(report_type_id, scope=scope, as_of=as_of,
                           emit_audit=emit_audit, writer=writer)
    run_id = pack.pack_id
    # The gate runs at the door (api/routers/reports.py) when the job is filed;
    # its decision reaches here through the job's audit context. A caller that
    # came in any other way passed no gate, and the event says so.
    entitlement = audit.current_context().get("entitlement")
    scope_details = {"report_type_id": report_type_id, "scope": scope,
                     "as_of": pack.as_of, "entitlement_checked": bool(entitlement)}
    if entitlement:
        scope_details["entitlement"] = entitlement
    event(audit.SCOPE_RESOLVED, run_id=run_id, pack_hash=pack.hash,
          summary=f"{report_type_id} · {scope}", details=scope_details)

    # FACT_PACK emits its own report.factpack_built inside build_fact_pack, so
    # it is not repeated here — one event per thing that happened.
    findings.extend(pack.findings)

    # -- STYLE_BRIEF -------------------------------------------------------
    brief = resolve_style_brief(report_type_id)
    event(audit.STYLEBRIEF_RESOLVED, run_id=run_id, pack_hash=pack.hash,
          summary=brief.disclosure(),
          details={"style_version": brief.version(), "resolver": brief.resolver,
                   "unresolved_scopes": list(brief.unresolved_scopes),
                   "provenance": dict(brief.provenance)})

    # -- COMPOSE -----------------------------------------------------------
    if ast is None:
        try:
            ast = compose_report(pack, brief, report_type_id, generate=generate,
                                 template=template, emit_audit=emit_audit,
                                 writer=writer)
        except CompositionError as exc:
            findings.append(Finding(
                finding_id=f"{run_id}-CMP001",
                code=FindingCode.REPORT_UNTRACED_FIGURE,
                severity=Severity.HIGH,
                detail=f"composition failed: {exc}",
                blocks_release=True))
            logger.warning("rga: %s composition failed — %s", run_id, exc)
            return ReportRun(run_id=run_id, report_type_id=report_type_id,
                             pack=pack, brief=brief, stage_reached="COMPOSE",
                             findings=findings)
    else:
        # The deterministic path. Recorded so a run that used no model is
        # distinguishable from one that did.
        event(audit.COMPOSED, run_id=run_id, pack_hash=pack.hash,
              summary="hand-written AST; no model involved",
              details={"model": None, "prompt_version": None,
                       "sections": len(ast.sections)})

    # -- RENDER ------------------------------------------------------------
    # A renderer fault is this function's to report, not to propagate. It
    # promises never to raise for a report that merely failed, and a drawing
    # library raising on a degenerate block is exactly that: live, a table with
    # no columns reached python-pptx and came back as ZeroDivisionError, which
    # took the whole call down instead of blocking one report.
    try:
        artefact = renderer.render(ast, pack, brief, title=title)
    except Exception as exc:
        findings.append(Finding(
            finding_id=f"{run_id}-RND001",
            code=FindingCode.RENDER_FAILED,
            severity=Severity.HIGH,
            detail=f"the renderer could not draw this report: "
                   f"{type(exc).__name__}: {exc}",
            blocks_release=True))
        logger.exception("rga: %s render failed", run_id)
        return ReportRun(run_id=run_id, report_type_id=report_type_id, pack=pack,
                         brief=brief, ast=ast, stage_reached="RENDER",
                         findings=findings)
    event(audit.RENDERED, run_id=run_id, pack_hash=pack.hash,
          summary=f"{artefact.renderer}/{artefact.renderer_version} · "
                  f"{len(artefact.content)} bytes",
          details={"renderer": artefact.renderer,
                   "renderer_version": artefact.renderer_version,
                   "media_type": artefact.media_type,
                   "ast_hash": artefact.ast_hash,
                   "style_version": artefact.style_version,
                   "bytes": len(artefact.content)})

    # -- POST_CHECK --------------------------------------------------------
    result = postcheck.run(artefact, pack, ast, brief, emit_audit=emit_audit,
                           writer=writer)
    findings.extend(result.findings)

    if not result.passed:
        logger.info("rga: %s blocked by %d finding(s)", run_id, len(result.blocking))
        return ReportRun(run_id=run_id, report_type_id=report_type_id, pack=pack,
                         brief=brief, ast=ast, artefact=artefact, result=result,
                         stage_reached="POST_CHECK", findings=findings)

    # -- APPROVAL ----------------------------------------------------------
    # Not built; no matrix exists to consult. Deliberately emits nothing rather
    # than an approval event nobody granted.

    # -- RELEASE -----------------------------------------------------------
    # record_action_or_fail underneath: a release whose audit cannot be written
    # does not happen.
    event(audit.RELEASED, run_id=run_id, pack_hash=pack.hash,
          summary=f"{report_type_id} released · {len(artefact.content)} bytes",
          details={"report_type_id": report_type_id,
                   "ast_hash": artefact.ast_hash,
                   "style_version": artefact.style_version,
                   "renderer_version": artefact.renderer_version,
                   "approval_required": "unknown — no approval matrix exists",
                   "external_release": False})

    return ReportRun(run_id=run_id, report_type_id=report_type_id, pack=pack,
                     brief=brief, ast=ast, artefact=artefact, result=result,
                     released=True, stage_reached="RELEASE", findings=findings)
