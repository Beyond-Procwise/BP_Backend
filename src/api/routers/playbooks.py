"""Playbooks and the proposals they raise.

Two different approvals live here and they are not the same act:

  * Approving a PLAYBOOK makes a strategy active. It changes what the system
    will recommend for every matching finding from now on, so it is
    configuration and gates on ``playbook.approve``.
  * Approving a PROPOSAL runs a workflow, once, for one finding. It gates on
    the existing ``workflow.run`` and executes through
    ``agent_workflows.start_run`` -- the one run path -- rather than a copy.

Nothing here runs a workflow without a person having called an approve
endpoint. There is deliberately no autorun flag to add later without a
conversation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.routers.agent_workflows import start_run
from repositories import playbook_repo as repo
from repositories.playbook_repo import LifecycleError
from services import agent_actions
from services.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playbooks", tags=["Playbooks"])


class PlaybookBody(BaseModel):
    playbook_name: str
    trigger_source: str
    trigger_match: Dict[str, Any] = Field(default_factory=dict)
    agent_workflow_id: int
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class RejectBody(BaseModel):
    reason: str = ""


def _subject(principal: Any) -> str:
    return getattr(principal, "subject", None) or ""


_OPEN_SQL = {
    "detection_finding": (
        "SELECT 1 FROM proc.bp_detection_finding "
        " WHERE finding_id = %s::bigint AND status = 'open'"
    ),
    "opportunity": (
        "SELECT 1 FROM proc.bp_opportunity "
        " WHERE opportunity_id = %s AND retired_at IS NULL"
    ),
}


def finding_is_open(source: str, finding_id: str) -> bool:
    """Is the finding this proposal was raised for still open?

    Asked at approval time, not at proposal time. A queue that sat for a week
    must not act on work somebody has since closed.
    """

    sql = _OPEN_SQL.get(source)
    if sql is None:
        return False
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, (finding_id,))
            return cur.fetchone() is not None
        finally:
            cur.close()


# -- playbooks -----------------------------------------------------------

@router.get("")
def list_playbooks(status: Optional[str] = None) -> Dict[str, Any]:
    return {"playbooks": repo.list_playbooks(status)}


@router.post("")
def create_playbook(body: PlaybookBody, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter")
    try:
        playbook_id = repo.create(
            name=body.playbook_name,
            trigger_source=body.trigger_source,
            trigger_match=body.trigger_match,
            agent_workflow_id=body.agent_workflow_id,
            params=body.params,
            description=body.description,
            authored_by=_subject(principal),
        )
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        # An unknown or null match key. The author is told which one, because a
        # playbook that never fires is indistinguishable from one that has not
        # matched yet.
        raise HTTPException(status_code=400, detail=str(exc))
    return {"playbook_id": playbook_id, "playbook": repo.get(playbook_id)}


@router.get("/proposals")
def list_proposals(status: Optional[str] = "proposed", limit: int = 100) -> Dict[str, Any]:
    return {"proposals": repo.list_proposals(status=status, limit=limit)}


@router.get("/{playbook_id}")
def get_playbook(playbook_id: int) -> Dict[str, Any]:
    found = repo.get(playbook_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such playbook")
    return found


@router.put("/{playbook_id}")
def update_playbook(
    playbook_id: int, body: PlaybookBody, principal=Depends(require_user)
) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id})
    try:
        return repo.update(
            playbook_id,
            name=body.playbook_name,
            trigger_source=body.trigger_source,
            trigger_match=body.trigger_match,
            agent_workflow_id=body.agent_workflow_id,
            params=body.params,
            description=body.description,
            modified_by=_subject(principal),
        )
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{playbook_id}/submit")
def submit_playbook(playbook_id: int, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id})
    try:
        repo.submit(playbook_id, modified_by=_subject(principal))
    except LifecycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return repo.get(playbook_id)


@router.post("/{playbook_id}/approve")
def approve_playbook(playbook_id: int, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.approve", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id})
    try:
        return repo.approve(playbook_id, approver=_subject(principal))
    except LifecycleError as exc:
        # Self-approval and "this workflow is inactive" are refusals, not
        # malformed requests, so they are 403 rather than 400.
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/{playbook_id}/retire")
def retire_playbook(playbook_id: int, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("playbook.write", principal, agent="PlaybooksRouter",
         context={"playbook_id": playbook_id, "retiring": True})
    repo.retire(playbook_id, modified_by=_subject(principal))
    return repo.get(playbook_id)


# -- proposals -----------------------------------------------------------

def _audit(
    action_type: str,
    proposal: Dict[str, Any],
    subject: str,
    *,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record what was decided about a proposal.

    gate() already audits playbook.write, playbook.approve and workflow.run.
    It does not cover these: rejecting is ungated (refusing to act is not an
    act) and superseding happens after the gate has passed. Without this, the
    two most interesting outcomes -- a person said no, and the work had already
    been closed -- would leave no trail at all.
    """

    agent_actions.record_action(
        phase="playbook",
        action_type=action_type,
        agent="PlaybooksRouter",
        deal_id=proposal.get("deal_id"),
        status=status,
        summary=(
            f"proposal {proposal['proposal_id']} ({proposal.get('playbook_name')}) "
            f"for {proposal['finding_source']} {proposal['finding_id']}: {status}"
        ),
        details={
            "proposal_id": proposal["proposal_id"],
            "playbook_id": proposal["playbook_id"],
            "finding_source": proposal["finding_source"],
            "finding_id": proposal["finding_id"],
            "decided_by": subject,
            **(extra or {}),
        },
    )


def _decidable(proposal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not proposal:
        raise HTTPException(status_code=404, detail="No such proposal")
    status = proposal.get("proposal_status")
    if status != "proposed":
        raise HTTPException(
            status_code=409,
            detail=f"proposal {proposal['proposal_id']} is already {status}",
        )
    return proposal


@router.post("/proposals/{proposal_id}/approve")
def approve_proposal(
    proposal_id: int, request: Request, principal=Depends(require_user)
) -> Dict[str, Any]:
    """Accept the recommendation and run the strategy. Once."""

    gate("workflow.run", principal, agent="PlaybooksRouter",
         context={"proposal_id": proposal_id})
    proposal = _decidable(repo.get_proposal(proposal_id))
    subject = _subject(principal)

    if not finding_is_open(proposal["finding_source"], proposal["finding_id"]):
        repo.mark_proposal_superseded(proposal_id, subject)
        _audit("proposal.superseded", proposal, subject, status="superseded")
        return {"proposal_id": proposal_id, "proposal_status": "superseded",
                "detail": "the finding this was raised for is no longer open"}

    # Retiring is how a strategy is withdrawn. It stops the playbook proposing;
    # it has to stop the proposals it already raised from running the graph too,
    # or "retired" means the queue keeps executing it for as long as the queue
    # is long. Rejecting one stays available either way, which is what the spec
    # means by an existing proposal staying decidable.
    playbook = repo.get(proposal["playbook_id"]) or {}
    if playbook.get("playbook_status") != "active":
        repo.mark_proposal_superseded(proposal_id, subject)
        _audit("proposal.superseded", proposal, subject, status="superseded")
        return {
            "proposal_id": proposal_id, "proposal_status": "superseded",
            "detail": (
                f"the playbook that raised this is "
                f"{playbook.get('playbook_status') or 'gone'}, so its strategy "
                "is no longer in service"
            ),
        }

    # THE CLAIM, and it happens before anything runs. get_conn() is autocommit,
    # so the status read above took no lock: two approvals arriving together --
    # a double-clicked button, a gateway retry -- both pass it. Whoever wins
    # this UPDATE runs the graph; the loser is told it is already decided. With
    # the claim after the run instead, both would already have sent the email.
    if not repo.claim_proposal(proposal_id, subject):
        raise HTTPException(
            status_code=409,
            detail=f"proposal {proposal_id} has already been decided",
        )

    payload = dict(proposal.get("params") or {})
    # The strategy is told which finding it is answering. A graph that does not
    # know that can only act on the whole corpus.
    payload.update({
        "finding_source": proposal["finding_source"],
        "finding_id": proposal["finding_id"],
        "deal_id": proposal.get("deal_id"),
        "playbook_id": proposal["playbook_id"],
        "proposal_id": proposal_id,
    })
    try:
        result = start_run(request, proposal["agent_workflow_id"], payload, principal)
    except Exception:
        # The claim was taken before the run, which is what stops a double
        # execution. If the run could not be started at all, that claim has to
        # go back -- otherwise one 503 from an orchestrator still loading leaves
        # the proposal undecidable for ever, with the unique index preventing a
        # replacement. Released only while run_id IS NULL, so a run that did
        # start is never un-decided.
        repo.release_proposal_claim(proposal_id)
        raise

    if result.get("status") == "awaiting_input":
        # It stopped to ask a person a question, which is not executing.
        # Calling it executed would close the proposal for ever while nothing
        # had happened, and the unique index would block a fresh one for that
        # finding. It stays claimed, with its run recorded, until the run ends.
        repo.note_proposal_run(proposal_id, result.get("run_id"))
        _audit("proposal.approved", proposal, subject, status="awaiting_input",
               extra={"run_id": result.get("run_id")})
        return {"proposal_id": proposal_id, "proposal_status": "approved", **result}

    repo.mark_proposal_executed(proposal_id, result.get("run_id"), subject)
    _audit("proposal.executed", proposal, subject, status="executed",
           extra={"run_id": result.get("run_id")})
    return {"proposal_id": proposal_id, "proposal_status": "executed", **result}


@router.post("/proposals/{proposal_id}/reject")
def reject_proposal(
    proposal_id: int, body: RejectBody, principal=Depends(require_user)
) -> Dict[str, Any]:
    """Decline the recommendation. Runs nothing, and needs no gate: refusing to
    act is not an act. The reason is required, because a rejected
    recommendation with no reason teaches nobody anything."""

    proposal = _decidable(repo.get_proposal(proposal_id))
    reason = (body.reason or "").strip()
    if not reason:
        raise HTTPException(
            status_code=400,
            detail="a rejection must say why -- it is the only signal an author gets",
        )
    repo.mark_proposal_rejected(proposal_id, _subject(principal), reason)
    _audit("proposal.rejected", proposal, _subject(principal),
           status="rejected", extra={"reason": reason})
    return {"proposal_id": proposal_id, "proposal_status": "rejected"}
