"""The two places a language model is allowed to act, and the gate on each.

WHAT THE MODEL IS AND IS NOT DOING HERE

It is choosing *structure and words*: which facts lead, how they are grouped
into sections, what the prose around them says. It is not producing a single
figure. Every number in the output arrives as a ``{{F0042}}`` reference that the
renderer substitutes from the pack, and the type system refuses prose containing
a digit — so "the model must not invent numbers" is not an instruction in a
prompt, it is a shape the output cannot take.

WHY THE LOCAL MODEL AND NOT AN API

Ruled 2026-09-12. The integration the brief named (D13) does not exist in this
codebase, and the local path is not a compromise: ``format=<json schema>`` gives
Ollama grammar-guided decoding, so invalid tokens are *masked during decoding*
and the model cannot emit a shape the schema forbids. That is strictly stronger
than asking an API politely for JSON and parsing what comes back. The two narrow
LLM functions already in this codebase — ``analytics/insight`` and
``style/compiler`` — both work this way, so this is a third instance of an
established pattern rather than a new one. Swapping providers later is a change
to ``_default_generate`` alone.

WHAT THE GRAMMAR CANNOT DO, AND WHAT CATCHES IT

A JSON schema constrains shape, never meaning. It can say ``text`` is a string;
it cannot say the string may not contain a digit, and it cannot say every
``fact_ref`` must exist in this particular pack. Those rules live in the Pydantic
validators and in ``_semantic_faults`` below, and the decoder cannot see either.
So a rejection is treated as a correctable mistake: the fault is handed back once,
naming what was wrong, and a second failure fails closed with no AST at all.

The retry runs at a non-zero temperature, for the reason ``style/compiler``
records from observation: at temperature 0 the decoder is deterministic, so a
retry reproduces the rejected answer byte for byte and the second attempt is
pure waste.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.services.analytics.models import Confidence
from src.services.rga import audit
from src.services.rga.models import (
    FactPack,
    NarrativeBlock,
    ReportAST,
    canonical_hash,
)
from src.services.rga.style import StyleBrief

logger = logging.getLogger(__name__)

AGENT = "rga_composer"

# Bump when the instructions below change. Recorded on report.composed so a
# report can be tied to the wording that produced it.
PROMPT_VERSION = "compose/1.0.0"
PROMPT_TYPE = "report_composition"
SELECT_PROMPT_TYPE = "report_type_selection"

# One corrective retry, then closed. Matches style/compiler and the brief.
MAX_ATTEMPTS = 2
RETRY_TEMPERATURE = 0.3

# A composition is not on a user's request path — a report is a deliberate act —
# but it is also not an overnight batch. Ten minutes covers a cold model load on
# this box without letting a hung call sit forever.
COMPOSE_TIMEOUT_SECONDS = 600
COMPOSE_NUM_PREDICT = 4096


class CompositionError(RuntimeError):
    """The model could not produce a usable report. No AST is returned.

    Deliberately carries no fallback. A hand-written AST substituted here would
    make a failed composition indistinguishable from a successful one, and the
    caller would release a report nobody composed.
    """


# --------------------------------------------------------------------------
# The wire format, and why it is not the AST
# --------------------------------------------------------------------------
#
# ``ReportAST`` models a section's blocks as a tagged union, which is the right
# shape for the renderer and the post-check. It is the wrong shape to hand to a
# grammar, because **Ollama's JSON-schema converter does not honour ``oneOf``
# with a ``discriminator``**. Measured on 2026-09-12 against AgentNick:unified,
# same prompt, three schemas:
#
#   full tagged union        -> 21 blocks, every one a finding_list (the only
#                               member with no required fields)
#   union minus finding_list -> good prose, but every object emitted with no
#                               "type" field at all
#   no union whatsoever      -> 17 correct metric blocks, right fact_refs,
#                               right emphasis
#
# The model was never the problem. So composition happens against a union-free
# draft — one typed array per block kind — which the grammar can express
# exactly, and ``to_ast`` converts it into the real AST afterwards. The strict
# type keeps its job: every rule (no digits in prose, no literal figure in a
# cell) is enforced on the converted result, not on the draft.


# Every closed vocabulary below is a Literal, not a str.
#
# They were bare strings first, and the model duly invented a role the AST then
# refused at conversion — a whole composition thrown away over a word the
# grammar could have forbidden outright. A JSON schema cannot express "no digit
# in this sentence", but it expresses an enum perfectly well, so anything that
# CAN be constrained in the grammar is constrained there and never left to be
# caught downstream.


class DraftMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_ref: str
    emphasis: Literal["primary", "secondary"] = "secondary"


class DraftNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    fact_refs: List[str] = Field(default_factory=list)
    role: Literal["statement", "recommendation"] = "statement"


class DraftTable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)


class DraftSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    fact_refs: List[str] = Field(default_factory=list)


class DraftChart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chart_type: Literal["bar", "waterfall", "line", "donut"] = "bar"
    series: List[DraftSeries] = Field(default_factory=list)


class DraftSection(BaseModel):
    """One section, with each kind of content in its own array.

    Separate arrays rather than one polymorphic list: that is the whole point —
    there is no union here for a grammar to mishandle.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    metrics: List[DraftMetric] = Field(default_factory=list)
    narratives: List[DraftNarrative] = Field(default_factory=list)
    tables: List[DraftTable] = Field(default_factory=list)
    charts: List[DraftChart] = Field(default_factory=list)
    finding_refs: List[str] = Field(default_factory=list)


class CompositionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: List[DraftSection] = Field(default_factory=list)


#: Block order within a rendered section. Fixed rather than chosen by the model:
#: figures first, then what they mean, then the detail, then what is wrong.
_BLOCK_ORDER = ("metrics", "narratives", "tables", "charts", "findings")


def to_ast(draft: CompositionDraft) -> ReportAST:
    """Convert a draft into the strict AST. Validation happens here, not before.

    Every rule the draft cannot express — no digit in prose, no literal figure
    in a table cell, a known block tag — is enforced by ``ReportAST`` as this
    builds it, so a draft that describes an illegal report raises exactly the
    ``ValidationError`` the caller turns into a corrective retry.
    """
    from src.services.rga.models import (
        ChartBlock, ChartSeries, FindingListBlock, MetricBlock, NarrativeBlock,
        Section, TableBlock,
    )

    sections = []
    for drafted in draft.sections:
        blocks: List[Any] = []
        blocks.extend(
            MetricBlock(fact_ref=m.fact_ref, emphasis=m.emphasis)
            for m in drafted.metrics)
        blocks.extend(
            NarrativeBlock(text=n.text, fact_refs=n.fact_refs, role=n.role)
            for n in drafted.narratives)
        blocks.extend(
            TableBlock(columns=t.columns, rows=t.rows) for t in drafted.tables)
        blocks.extend(
            ChartBlock(chart_type=c.chart_type,
                       series=[ChartSeries(label=s.label, fact_refs=s.fact_refs)
                               for s in c.series])
            for c in drafted.charts)
        if drafted.finding_refs:
            blocks.append(FindingListBlock(finding_refs=drafted.finding_refs))
        sections.append(Section(id=drafted.id, title=drafted.title, blocks=blocks))
    return ReportAST(sections=sections)


def composition_schema() -> Dict[str, Any]:
    """The schema handed to the grammar. Must contain no union, ever.

    ``tests/services/rga/test_compose.py`` asserts that, because reintroducing
    one would not fail loudly — it would quietly produce empty reports again.
    """
    return CompositionDraft.model_json_schema()


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

_SYSTEM = """You lay out a procurement report. You are given a fixed set of measured \
facts and a house style. You decide the structure and write the prose.

THE ONE RULE THAT MATTERS: you must never type a digit. Not one, anywhere, in any \
string. Every figure is referenced by its fact id and substituted later from the \
record.

  WRONG:  "Invoiced spend reached £5.8M across 376 deals."
  RIGHT:  "Invoiced spend reached {{F0003}} across {{F0001}} deals."

  WRONG:  "only 29.5% achieved three-way match"
  RIGHT:  "only {{F0004}} achieved three-way match"

This applies to every number without exception — amounts, counts, percentages, \
days, years, quarters. "the top 3 suppliers", "up 12%", "16 days", "in 2026 Q1": \
all rejected. If a fact says it, reference the fact. If no fact says it, do not \
say it at all.

Do NOT name the period, quarter, month or year anywhere in your prose. The report \
prints its own scope on the front page, so "during the quarter" or "in the period" \
is what you write — never "in 2026 Q1", which is a digit and will be rejected.

Units are safe to write as words: "{{F0005}} days", "{{F0002}} suppliers".

Never write an identifier of any kind into prose — not a fact id, not a finding \
id, not a pack id. Identifiers contain digits and will be rejected. The ONLY \
thing that may appear in prose is a {{F0042}} placeholder. To draw attention to \
a finding, put its id in that section's finding_refs array and say nothing about \
it in the text.

Where each kind of content goes:
  - metrics:    a headline figure. Name the fact id in fact_ref.
  - narratives: prose. Use {{F0003}} placeholders for every figure, and list \
those same ids in fact_refs.
  - tables:     each cell is either a fact id — F0003 or {{F0003}}, both \
accepted — or a text label with no digits in it.
  - charts:     name the fact ids in each series.
  - finding_refs: ids of findings to list. These are FINDING ids, not fact ids.

Further rules:
  - Every narrative block must list, in fact_refs, the facts it relies on.
  - A fact marked UNASSESSED has no measured value. You may state that it is \
unavailable. You must NOT use it as the basis of a recommendation block \
(role="recommendation"); recommending on the strength of a figure nobody measured \
is the failure this rule exists to prevent.
  - Use only the fact ids listed below. An id that is not in the list does not exist.
  - Do not editorialise beyond what the facts state. No "significant", "concerning", \
"strong" unless a fact or finding says so.
"""

_USER = """REPORT TYPE: {report_type}
SCOPE: {scope}

HOUSE STYLE
  tone: {tone}
  lead with: {lead_with}
  section order: {section_order}
  executive summary: at most {max_bullets} bullets
  preferred charts: {charts}

FACTS AVAILABLE ({n_facts})
{facts}

FINDINGS RAISED ({n_findings})
{findings}

Produce the report as JSON matching the schema. Order the sections as the house \
style asks. Lead with what the style says to lead with.
"""


def _render_facts(pack: FactPack) -> str:
    """The menu the model chooses from.

    It is shown each fact's *rendered* form, never a raw amount — the same rule
    ``analytics/insight`` holds: a model shown raw amounts is a model formatting
    money again, and this one is not allowed to type digits at all.
    """
    lines = []
    for entry in pack.facts:
        bits = [f"  {entry.fact_id}  {entry.label}", f"= {entry.display}"]
        if entry.unit:
            bits.append(f"({entry.unit})")
        bits.append(f"[{entry.confidence.value}]")
        if entry.confidence is Confidence.UNASSESSED:
            bits.append("— no measured value; may be stated, never recommended on")
        lines.append(" ".join(bits))
    return "\n".join(lines) or "  (none)"


def _render_findings(pack: FactPack) -> str:
    if not pack.findings:
        return "  (none)"
    return "\n".join(
        f"  {f.finding_id}  {f.severity.value}  {f.code.value} — {f.detail}"
        for f in pack.findings
    )


def build_prompt(pack: FactPack, brief: StyleBrief, report_type: str,
                 template: Optional[str] = None) -> str:
    """The exact text handed to the model. ``template`` overrides the built-in."""
    body = (template or _SYSTEM) + "\n\n" + _USER.format(
        report_type=report_type,
        scope=" · ".join(f"{k}={v}" for k, v in sorted(pack.scope.items())),
        tone=brief.get("report.style.tone"),
        lead_with=brief.get("report.style.lead_with"),
        section_order=", ".join(brief.get("report.style.section_order")),
        max_bullets=brief.get("report.style.exec_summary.max_bullets"),
        charts=", ".join(brief.get("report.style.chart.preferred")),
        n_facts=len(pack.facts),
        facts=_render_facts(pack),
        n_findings=len(pack.findings),
        findings=_render_findings(pack),
    )
    return body


def load_governed_instructions(conn: Any, prompt_type: str = PROMPT_TYPE) -> Optional[str]:
    """The instruction text from ``proc.bp_prompt``, if governance set one.

    The DB row wins over the built-in — that is the standing rule in this
    codebase. Read by the caller and passed in, so an unreadable governance row
    costs a report nothing but its own wording.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prompts_desc FROM proc.bp_prompt "
                "WHERE prompt_type = %s AND COALESCE(prompts_status, 1) = 1 "
                "ORDER BY prompt_id DESC LIMIT 1", (prompt_type,))
            row = cur.fetchone()
    except Exception:
        logger.debug("rga: prompt lookup failed; using the built-in", exc_info=True)
        return None
    if not row or not row[0]:
        return None
    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    if isinstance(payload, dict):
        text = payload.get("prompt_template") or payload.get("template")
        if text and str(text).strip():
            return str(text)
    return None


# --------------------------------------------------------------------------
# The model seam
# --------------------------------------------------------------------------


def _default_generate(prompt: str, *, schema: Any, temperature: float) -> Optional[str]:
    """The platform's local model, grammar-constrained to ``schema``.

    ``think=False`` because AgentNick is a hybrid reasoning model: left alone it
    answers in a ``thinking`` field this path does not read, and ``response``
    comes back empty. The model is left to ``ollama_client``'s default, which is
    the resident one — pinning a different model here would evict it.
    """
    from src.services.ollama_client import ollama_generate

    return ollama_generate(
        prompt, format=schema, temperature=temperature,
        num_predict=COMPOSE_NUM_PREDICT, think=False, retries=1,
        timeout=COMPOSE_TIMEOUT_SECONDS,
    )


Generator = Callable[..., Optional[str]]


# --------------------------------------------------------------------------
# compose_report
# --------------------------------------------------------------------------


def _semantic_faults(ast: ReportAST, pack: FactPack) -> List[str]:
    """The rules a JSON schema cannot state. Empty means the tree is usable.

    These are checked here as well as in the post-check on purpose. The
    post-check is the gate on the *artefact* and must stay; catching the same
    fault at composition means the model gets told what it did wrong while it
    can still fix it, instead of a report being built and then blocked.
    """
    faults: List[str] = []

    # Caught here as well as at the gate so the model is told while it can still
    # fix it. Live, an untagged union let twenty-one empty blocks through and the
    # report released stating nothing at all.
    if pack.facts and not ast.fact_refs():
        faults.append(
            f"the report references none of the {len(pack.facts)} available facts; "
            f"every section must present measured figures by their fact ids")
    if not any(section.blocks for section in ast.sections):
        faults.append("every section is empty; a report needs content blocks")

    known = {f.fact_id for f in pack.facts}
    for ref in sorted(ast.fact_refs()):
        if ref not in known:
            faults.append(
                f"{ref} is not a fact in this pack; available ids are "
                f"{', '.join(sorted(known)) or '(none)'}")

    for section in ast.sections:
        for block in section.blocks:
            if not isinstance(block, NarrativeBlock):
                continue
            refs = set(block.fact_refs) | set(block.placeholders())
            if block.role == "recommendation":
                for ref in sorted(refs):
                    entry = pack.fact(ref)
                    if entry is not None and entry.confidence is Confidence.UNASSESSED:
                        faults.append(
                            f"section {section.id!r} makes a recommendation resting on "
                            f"{ref}, which is UNASSESSED — state it, do not recommend on it")
            for ref in sorted(set(block.placeholders()) - set(block.fact_refs)):
                faults.append(
                    f"section {section.id!r} quotes {{{{{ref}}}}} but does not list "
                    f"{ref} in fact_refs")
    return faults


def _attempt(raw: Optional[str], pack: FactPack) -> Tuple[Optional[ReportAST], Optional[str]]:
    """Parse and validate one response. Returns (ast, fault)."""
    if raw is None or not str(raw).strip():
        return None, "the model returned nothing"
    text = str(raw).strip()
    # Grammar-guided decoding should make fences impossible, but a model behind
    # a different provider may add them and the cost of tolerating that is one
    # strip.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        return None, f"the response was not JSON: {exc}"
    try:
        draft = CompositionDraft.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        return None, f"the draft did not validate at {where}: {first.get('msg')}"
    try:
        # The strict rules live here: converting is what enforces them.
        ast = to_ast(draft)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        return None, f"the report did not validate at {where}: {first.get('msg')}"
    faults = _semantic_faults(ast, pack)
    if faults:
        return None, "; ".join(faults[:4])
    return ast, None


def compose_report(
    pack: FactPack,
    brief: StyleBrief,
    report_type: str,
    *,
    generate: Optional[Generator] = None,
    template: Optional[str] = None,
    emit_audit: bool = True,
    writer: Any = None,
) -> ReportAST:
    """Lay out a report from a Fact Pack. Raises ``CompositionError`` if it cannot.

    One corrective retry, then closed. There is no fallback AST, by design.
    """
    schema = composition_schema()
    prompt = build_prompt(pack, brief, report_type, template)
    call = generate or _default_generate

    faults: List[str] = []
    ast: Optional[ReportAST] = None
    attempts = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        temperature = 0.0 if attempt == 1 else RETRY_TEMPERATURE
        ask = prompt if attempt == 1 else (
            f"{prompt}\n\nYour previous answer was rejected: {faults[-1]}\n"
            f"Return a corrected report. Change only what was wrong.")
        try:
            raw = call(ask, schema=schema, temperature=temperature)
        except Exception as exc:
            faults.append(f"the model was unreachable: {exc}")
            logger.warning("rga: compose attempt %d failed: %s", attempt, exc)
            break
        ast, fault = _attempt(raw, pack)
        if ast is not None:
            break
        faults.append(fault or "unknown fault")
        logger.info("rga: compose attempt %d rejected — %s", attempt, fault)

    if emit_audit:
        audit.emit(
            audit.COMPOSED, run_id=pack.pack_id, agent=AGENT,
            pack_hash=pack.hash,
            status="ok" if ast is not None else "rejected",
            summary=(f"composed {len(ast.sections)} section(s)" if ast is not None
                     else f"failed after {attempts} attempt(s)"),
            details={
                "report_type_id": report_type,
                "model": _model_name(generate),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": audit.prompt_hash(prompt),
                "governed_prompt": template is not None,
                "attempts": attempts,
                "faults": faults,
                # See services/rga/audit: the shared client discards them.
                "token_counts": "unavailable via services.ollama_client",
                "ast_hash": canonical_hash(ast.model_dump(mode="json")) if ast else None,
            },
            writer=writer,
        )

    if ast is None:
        raise CompositionError(
            f"the model did not produce a usable report after {attempts} attempt(s): "
            + "; ".join(faults))
    return ast


def _model_name(generate: Optional[Generator]) -> str:
    if generate is not None:
        return getattr(generate, "__name__", "injected")
    try:
        from src.services.ollama_client import DEFAULT_MODEL

        return DEFAULT_MODEL
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# select_report_type
# --------------------------------------------------------------------------


class TypeSelection(BaseModel):
    """Which report the request is asking for, and how sure the model is."""

    model_config = ConfigDict(extra="forbid")

    report_type_id: str
    confidence: float = Field(ge=0.0, le=1.0)


_SELECT_SYSTEM = """You match a request to one of the report types listed. Answer \
with the id of the closest type and your confidence between 0 and 1.

If the request does not clearly match any of them, return the closest id with a \
low confidence. Do not invent an id that is not listed.

REPORT TYPES
{types}

REQUEST
{request}
"""


def select_report_type(
    user_request: str,
    available_types: List[str],
    *,
    generate: Optional[Generator] = None,
    emit_audit: bool = False,
    writer: Any = None,
) -> Optional[TypeSelection]:
    """Pick a report type from a natural-language request, or ``None``.

    Only used when the user did not choose from the selector. Returns ``None``
    rather than guessing when the answer is unusable or names a type that does
    not exist — the caller then asks the person, which is the correct outcome
    and costs one question.
    """
    if not available_types:
        return None

    schema = TypeSelection.model_json_schema()
    # Constrain the id to the ones that exist, so the grammar itself forbids a
    # hallucinated type rather than the validator catching it afterwards.
    schema["properties"]["report_type_id"] = {"enum": sorted(available_types)}
    prompt = _SELECT_SYSTEM.format(
        types="\n".join(f"  {t}" for t in sorted(available_types)),
        request=user_request.strip())

    call = generate or _default_generate
    selection: Optional[TypeSelection] = None
    fault: Optional[str] = None
    try:
        raw = call(prompt, schema=schema, temperature=0.0)
    except Exception as exc:
        fault = f"the model was unreachable: {exc}"
        raw = None

    if fault is None:
        if raw is None or not str(raw).strip():
            fault = "the model returned nothing"
        else:
            try:
                selection = TypeSelection.model_validate(json.loads(str(raw)))
            except (TypeError, ValueError, ValidationError) as exc:
                fault = f"unusable answer: {exc}"

    if selection is not None and selection.report_type_id not in available_types:
        fault = f"{selection.report_type_id!r} is not an available report type"
        selection = None

    if emit_audit:
        audit.emit(
            "report.type_selected", run_id=audit.prompt_hash(prompt)[:12],
            agent=AGENT, status="ok" if selection else "rejected",
            summary=(f"{selection.report_type_id} @ {selection.confidence:.2f}"
                     if selection else (fault or "no selection")),
            details={"request": user_request[:500], "fault": fault,
                     "available": sorted(available_types)},
            writer=writer,
        )
    return selection
