"""The light report editor: words and layout, never a number (ruled 2026-09-24).

A person may retitle a released report, rewrite its paragraphs, reorder or remove its
sections and blocks, and place any of the report's own verified figures -- as ``{{F0042}}``,
the same way the model does. Nobody types a figure: the report tree's own validators refuse
one, as they refuse the model's.

WHY AN EDIT IS RE-DRAWN FROM THE STORED PACK

The figures a report may state are the ones measured when it was made. Re-querying on save
would quietly swap in today's numbers under an edit that was only meant to change words, so
the edit is drawn against the Fact Pack stored with the job (``job_store.draft``). Both files
then face the same post-check the agent's draft faced, and a blocked edit saves nothing.

WHY EVERY SAVE IS A VERSION

A sign-off counts only for the version it saw (``signoff.state``), so a saved edit puts the
report back to awaiting sign-off, and the person who made it cannot sign it off. The versions
are kept (proc.bp_report_version) and each save is on the record as report.edited -- written
before the save commits, so an edit that cannot be audited does not happen.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.services.rga import audit, job_store, postcheck
from src.services.rga.models import (ChartBlock, FactPack, Finding, FindingCode, ReportAST,
                                     TableBlock)
from src.services.rga.style import resolve_style_brief

AGENT = "rga_editor"

#: Bounds on what one save may carry. Far above any report the agent writes (a handful of
#: sections, a few blocks each); here so a pasted blob cannot tie the renderer up.
MAX_SECTIONS = 30
MAX_BLOCKS = 60
MAX_TEXT = 4000
MAX_TITLE = 200

_BEFORE_EDITING = "this report was made before editing existed; run it again to edit it"


class NotEditable(Exception):
    """The job cannot be edited at all -- not a problem with the edit."""


class EditRefused(Exception):
    """The edit breaks a rule. ``reasons`` are sentences for the person who made it."""

    def __init__(self, reasons: List[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


#: Someone else saved a version first (``current`` is theirs). Raised by the store, under the
#: row lock that decides who won.
StaleVersion = job_store.StaleVersion


# -- reading ------------------------------------------------------------------------------


def facts_for_editor(pack: FactPack) -> List[Dict[str, Any]]:
    """Each figure the editor may place, with its standing -- never its workings: the
    derivation and provenance stay on the printed page's footnotes."""
    return [{"fact_id": f.fact_id, "label": f.label, "display": f.display,
             "confidence": f.confidence.value, "origin": f.origin.value} for f in pack.facts]


def _job(job_id: str) -> Dict[str, Any]:
    job = job_store.get(job_id)
    if job is None:
        raise LookupError(job_id)
    return job


def _refusal(job: Dict[str, Any]) -> Optional[str]:
    if job.get("status") != "released":
        return f"only a released report can be edited; this one is {job.get('status')}"
    if not job.get("editable"):
        return _BEFORE_EDITING
    return None


def _load(job_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], FactPack]:
    job = _job(job_id)
    reason = _refusal(job)
    if reason:
        raise NotEditable(reason)
    stored = job_store.draft(job_id)
    if stored is None:
        raise NotEditable(_BEFORE_EDITING)
    return job, stored, FactPack.model_validate(stored["fact_pack"])


def draft(job_id: str) -> Dict[str, Any]:
    """What the editor opens: the current version, and the figures it may place."""
    job = _job(job_id)
    reason = _refusal(job)
    if reason:
        return {"version": job.get("current_version"), "title": job.get("title"), "ast": None,
                "facts": [], "editable": False, "reason": reason}
    _, stored, pack = _load(job_id)
    return {"version": stored["version"], "title": stored["title"], "ast": stored["ast"],
            "facts": facts_for_editor(pack), "editable": True, "reason": None}


# -- checking -----------------------------------------------------------------------------


def _where(ast_json: Any, loc: Tuple[Any, ...]) -> str:
    """'In "Executive summary", paragraph 2' -- where a person would look."""
    try:
        section = ast_json["sections"][loc[1]]
        name = f'In "{section.get("title") or "an untitled section"}"'
    except (KeyError, IndexError, TypeError):
        return "In the report"
    if len(loc) > 3 and loc[2] == "blocks":
        return f"{name}, block {int(loc[3]) + 1}"
    return name


def _validation_words(exc: ValidationError, ast_json: Any) -> List[str]:
    reasons: List[str] = []
    for err in exc.errors():
        loc, msg = tuple(err.get("loc") or ()), str(err.get("msg") or "")
        where = _where(ast_json, loc)
        if "literal number" in msg:
            reasons.append(f"{where}: Sentences can't contain typed numbers — "
                           "insert a figure instead.")
        elif "literal figure" in msg:
            reasons.append(f"{where}: Table cells can't contain typed numbers — "
                           "choose a figure instead.")
        elif "at least one column" in msg:
            reasons.append(f"{where}: a table needs at least one column.")
        else:
            reasons.append(f"{where}: this part of the report is not laid out correctly.")
    return list(dict.fromkeys(reasons))


# Titles are printed in places a stylesheet reads (the page footer), so the characters that
# could end or extend a style rule are refused outright rather than trusted to escaping alone.
_MARKUP = re.compile(r"[<>{}]")
_NO_MARKUP = "{what} can't contain any of < > {{ }} -- use words instead."


def _labels(title: str, ast: ReportAST) -> List[Tuple[str, str]]:
    """Every heading and label a person can type into, with where it is: the report title,
    section titles, table column headings and chart series labels. Paragraphs and table cells
    have their own rule in the report tree's validators."""
    out = [("The report title", title)]
    for s in ast.sections:
        out.append((f'The section title "{s.title}"', s.title))
        for b in s.blocks:
            if isinstance(b, TableBlock):
                out += [(f'A column heading in "{s.title}"', c) for c in b.columns]
            elif isinstance(b, ChartBlock):
                out += [(f'A chart label in "{s.title}"', x.label) for x in b.series]
    return out


def _new_numbers(title: str, ast: ReportAST, base_title: str, base: ReportAST) -> List[str]:
    """Headings and labels may not gain a number (final review): the post-check only asks
    whether a number is one of the report's figures somewhere, so a typed '376' that happens
    to equal a deal count passed as a section title. A heading the agent itself wrote with a
    number in it ('2026 Q1') may stay as it is."""
    kept = {text for _, text in _labels(base_title, base)}
    return list(dict.fromkeys(
        f"{where}: Titles and labels can't contain typed numbers — say it in a paragraph "
        "with a figure instead." for where, text in _labels(title, ast)
        if text not in kept and any(ch.isnumeric() for ch in text)))


def _limits(title: str, ast_json: Any) -> List[str]:
    reasons: List[str] = []
    if not title.strip():
        reasons.append("The report needs a title.")
    elif len(title) > MAX_TITLE:
        reasons.append(f"The title is too long; keep it under {MAX_TITLE} characters.")
    if _MARKUP.search(title):
        reasons.append(_NO_MARKUP.format(what="The title"))
    sections = ast_json.get("sections") if isinstance(ast_json, dict) else None
    if not isinstance(sections, list):
        return reasons + ["The report's layout could not be read."]
    if len(sections) > MAX_SECTIONS:
        reasons.append(f"The report has too many sections; keep it to {MAX_SECTIONS}.")
        return reasons
    for s in sections:
        if not isinstance(s, dict):
            continue
        name = s.get("title") or "an untitled section"
        if not str(s.get("title") or "").strip():
            reasons.append("Every section needs a title.")
        elif _MARKUP.search(str(s.get("title"))):
            reasons.append(_NO_MARKUP.format(what=f'The section title "{name}"'))
        blocks = s.get("blocks") or []
        if len(blocks) > MAX_BLOCKS:
            reasons.append(f'"{name}" has too many blocks; keep it to {MAX_BLOCKS}.')
        if any(isinstance(b, dict) and len(str(b.get("text") or "")) > MAX_TEXT for b in blocks):
            reasons.append(f'A paragraph in "{name}" is too long; keep it under '
                           f"{MAX_TEXT} characters.")
    return list(dict.fromkeys(reasons))


def _finding_words(f: Finding, pack: FactPack) -> str:
    fact = pack.fact(f.fact_id or f.location or "")
    label = f'"{fact.label}"' if fact else "a figure"
    if f.code is FindingCode.REPORT_UNTRACED_FIGURE:
        token = f.detail.split("'")[1] if f.detail.count("'") >= 2 else "A number"
        near = f' (in "{f.location}")' if f.location else ""
        return (f"{token}{near} isn't one of the report's verified figures — remove it, "
                "or insert a figure instead.")
    if f.code is FindingCode.REPORT_STATES_NOTHING:
        return "The report must show at least one of its verified figures."
    if f.code is FindingCode.UNASSESSED_IN_RECOMMENDATION:
        entry = next((p for p in pack.facts if f"{p.fact_id} (" in f.detail), None)
        name = f'"{entry.label}"' if entry else "a figure"
        return (f"A recommendation can't rest on {name}: it was not measured. "
                "Make it a statement, or rest it on a measured figure.")
    if f.code in (FindingCode.MISSING_ORIGIN_BADGE, FindingCode.MISSING_PROVENANCE_FOOTNOTE):
        return f"{label} would be printed without its evidence marks; place it again."
    return "The edited report failed a check the agent's own draft must pass."


def _parse(pack: FactPack, title: str, ast_json: Any,
           stored: Optional[Dict[str, Any]] = None) -> ReportAST:
    """The edit as a report tree, or every reason it is not one -- all at once, so a person
    fixes them in one go rather than meeting them one per save."""
    reasons = _limits(title, ast_json)
    if any("too many sections" in r or "could not be read" in r for r in reasons):
        raise EditRefused(reasons)
    try:
        ast = ReportAST.model_validate(ast_json)
    except ValidationError as exc:
        raise EditRefused(reasons + _validation_words(exc, ast_json)) from None
    reasons += [f"{r} isn't one of this report's figures — choose one from the figure list."
                for r in sorted(r for r in ast.fact_refs() if pack.fact(r) is None)]
    if stored is not None:
        reasons += _new_numbers(title, ast, stored.get("title") or "",
                                ReportAST.model_validate(stored["ast"]))
    if reasons:
        raise EditRefused(reasons)
    return ast


def _draw(job: Dict[str, Any], pack: FactPack, ast: ReportAST, title: str, *,
          draft: bool = False):
    """Both files, from the stored pack, with the report type's style; and the post-check's
    blocking reasons in words (both files, each reason once)."""
    from src.services.rga.render import html as page_renderer
    from src.services.rga.render import pptx as deck_renderer

    brief = resolve_style_brief(job["report_type"])
    try:
        deck = deck_renderer.render(ast, pack, brief, title=title)
        page = page_renderer.render(ast, pack, brief, title=title, draft=draft)
    except Exception:  # noqa: BLE001 - a drawing fault is a refusal, not a crash
        raise EditRefused(["The edited report could not be drawn; undo the last change "
                           "and try again."]) from None
    blocking: List[str] = []
    for art in (deck, page):
        result = postcheck.run(art, pack, ast, brief, emit_audit=False)
        blocking.extend(_finding_words(f, pack) for f in result.blocking)
    return deck, page, list(dict.fromkeys(blocking))


# -- the two actions ----------------------------------------------------------------------


def preview(job_id: str, *, title: str, ast_json: Any) -> Tuple[Optional[bytes], List[str]]:
    """The printable page for an unsaved edit, and what would stop it saving. Nothing is
    written. An edit that cannot even be read gives reasons and no page."""
    job, stored, pack = _load(job_id)
    try:
        ast = _parse(pack, title.strip(), ast_json, stored)
        # Marked as a draft on every page: a preview is the whole report, and must never pass
        # for one that has been signed off (final review).
        _, page, blocking = _draw(job, pack, ast, title.strip(), draft=True)
    except EditRefused as exc:
        return None, exc.reasons
    return page.content, blocking


def save(job_id: str, *, base_version: int, title: str, ast_json: Any, by: str,
         summary: Optional[str] = None) -> int:
    """Save an edit as the next version. Raises NotEditable, EditRefused or StaleVersion;
    on any of them nothing is saved."""
    job, stored, pack = _load(job_id)
    title = title.strip()
    ast = _parse(pack, title, ast_json, stored)
    deck, page, blocking = _draw(job, pack, ast, title)
    if blocking:
        raise EditRefused(blocking)

    deck_sha = hashlib.sha256(deck.content).hexdigest()
    page_sha = hashlib.sha256(page.content).hexdigest()

    def on_the_record(version: int) -> None:
        # Inside the save's transaction: a raising writer rolls the save back.
        with audit.run_context(job_id=job_id, requested_by=job.get("requested_by")):
            audit.emit(audit.EDITED, run_id=job.get("run_id") or job_id, agent=AGENT,
                       pack_hash=pack.hash,
                       summary=f"report edited · version {version}",
                       details={"version": version, "base_version": base_version,
                                "edited_by": by, "summary": summary, "title": title,
                                "ast_hash": page.ast_hash, "deck_sha256": deck_sha,
                                "page_sha256": page_sha,
                                "style_version": page.style_version})

    return job_store.save_version(
        job_id, base_version=base_version, title=title, ast=ast.model_dump(mode="json"),
        deck=deck.content, page=page.content, by=by, summary=summary,
        before_commit=on_the_record)
