"""What a report is made of: facts, findings, and the tree a renderer walks.

The one rule this module exists to enforce is that **a language model never
originates a number**. That is not a prompt instruction here; it is a property
of the types:

  * A ``FactEntry`` cannot be built without a ``provenance_id``. A figure with
    no evidence behind it is not a fact that failed a check later — it is not
    constructible.
  * A ``NarrativeBlock`` cannot hold a digit. The schema has no numeric field,
    and the validator rejects a bare numeral in the prose. Numbers reach a
    sentence only as ``{{F0042}}`` placeholders that the renderer substitutes
    from the pack.

``Confidence`` is imported rather than redefined. ``services/analytics/models``
already carries the exact tri-state, and a second enum with the same members is
the drift ``services/actions`` was written to prevent.

``Origin`` is new, and reluctantly. The discovery pass found no origin tagging
anywhere in this codebase — ``LEGACY_UNVERIFIED`` has no referent at all — so
there is nothing to import. It is defined here rather than in a shared module
so that whoever eventually owns the platform-wide origin dimension is not
pre-empted by a definition RGA happened to need first. See docs/rga/discovery.md §3.1.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from src.services.analytics.formatting import (
    EMPTY_AMOUNT,
    format_delta,
    format_int,
    format_money,
    format_pct,
)
from src.services.analytics.models import Confidence

# A fact is addressed as F0042 everywhere: in the pack, in a narrative
# placeholder, in a chart series and in a post-check failure message. One
# spelling, so a reference that does not resolve is a typo rather than a
# convention nobody wrote down.
FACT_ID = re.compile(r"^F\d{4}$")

# The placeholder a narrative may carry in place of a figure.
PLACEHOLDER = re.compile(r"\{\{(F\d{4})\}\}")

# Any run of digits. Used two ways: to refuse a literal number in prose, and to
# collect the tokens a rendered artefact is allowed to contain. Deliberately the
# same expression as ``analytics/models._NUMBER`` so the two agree on what
# counts as a figure.
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class Origin(str, Enum):
    """Where a fact's underlying record came from.

    Kept apart from ``Confidence`` and never merged with it. They answer
    different questions: confidence is how well the figure is corroborated,
    origin is whether anybody ever verified the record it was computed from.
    A CORROBORATED aggregate over LEGACY_UNVERIFIED rows is a real combination,
    and collapsing the two fields would render it as something else.
    """

    OBSERVED = "OBSERVED"
    LEGACY_UNVERIFIED = "LEGACY_UNVERIFIED"


class FormatHint(str, Enum):
    """Which shared formatter renders this fact. Not a style choice.

    The post-check compares the numeric tokens in the artefact against the
    tokens in each fact's ``display``. That comparison is only sound if the
    renderer and the fact agree on the rendering, so the hint lives on the fact
    and the renderer obeys it rather than choosing.
    """

    MONEY = "money"
    PCT = "pct"
    INT = "int"
    DELTA = "delta"
    TEXT = "text"


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FindingCode(str, Enum):
    """Why a report was blocked, or what about it the reader must be told.

    These are RGA's own; they are not rows in ``proc.bp_detection_finding``
    (which is the extraction/compliance spine and has a different vocabulary).
    Surfacing them there is a separate piece of work — see discovery §3.2/D7.
    """

    FACT_WITHOUT_PROVENANCE = "FACT_WITHOUT_PROVENANCE"
    REPORT_UNTRACED_FIGURE = "REPORT_UNTRACED_FIGURE"
    # A report that states nothing passes every other check in §6 vacuously:
    # each of them is "if there is a figure, it must trace", and there are no
    # figures. Found live — the model returned twenty-one empty blocks and the
    # gate released it.
    REPORT_STATES_NOTHING = "REPORT_STATES_NOTHING"
    UNKNOWN_FACT_REF = "UNKNOWN_FACT_REF"
    UNASSESSED_IN_RECOMMENDATION = "UNASSESSED_IN_RECOMMENDATION"
    MISSING_ORIGIN_BADGE = "MISSING_ORIGIN_BADGE"
    MISSING_PROVENANCE_FOOTNOTE = "MISSING_PROVENANCE_FOOTNOTE"
    MISSING_STYLE_PROVENANCE = "MISSING_STYLE_PROVENANCE"
    MISSING_HASH_RECORD = "MISSING_HASH_RECORD"
    RENDER_FAILED = "RENDER_FAILED"
    MEASURE_UNAVAILABLE = "MEASURE_UNAVAILABLE"
    PARTIAL_CURRENCY_CONVERSION = "PARTIAL_CURRENCY_CONVERSION"


class Finding(BaseModel):
    """Something the reader, or the release gate, has to know.

    A failed post-check is itself a Finding — §6 of the brief — so this type is
    used both for "the pack could not measure this" and for "the artefact does
    not trace". ``blocks_release`` is what separates them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: str
    code: FindingCode
    severity: Severity
    detail: str
    blocks_release: bool = True
    fact_id: Optional[str] = None
    location: Optional[str] = None


class FactEntry(BaseModel):
    """One number the report is allowed to state, and the evidence for it.

    ``value`` may be ``None``. That is not a zero and must never render as one:
    an absent measure renders as ``—`` and carries ``UNASSESSED``. The
    distinction is the whole reason this codebase has an UNASSESSED singleton
    (``services/formulas/unassessed``) rather than a default of 0.0.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    label: str
    value: Optional[Decimal] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    format_hint: FormatHint = FormatHint.TEXT
    confidence: Confidence
    origin: Origin
    provenance_id: str = Field(min_length=1)
    derivation: str = Field(min_length=1)

    @field_validator("fact_id")
    @classmethod
    def _well_formed_id(cls, value: str) -> str:
        if not FACT_ID.match(value):
            raise ValueError(f"fact_id must look like F0042, got {value!r}")
        return value

    @field_validator("provenance_id")
    @classmethod
    def _provenance_is_real(cls, value: str) -> str:
        """A blank-but-present provenance is the failure this check exists for.

        ``min_length=1`` already refuses the empty string; this refuses the
        whitespace that would satisfy it.
        """
        if not value.strip():
            raise ValueError(
                "a fact must carry a provenance_id — a figure with no evidence "
                "behind it is not added to a pack, it raises a Finding")
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display(self) -> str:
        """The exact text a renderer may print and a sentence may quote.

        Produced by the shared formatter, not by a local one. Two formatters in
        one product diverge, and here divergence would break the post-check:
        a figure printed one way and checked against another spelling traces to
        nothing.
        """
        if self.value is None:
            return EMPTY_AMOUNT
        if self.format_hint is FormatHint.MONEY:
            return format_money(self.value, self.currency)
        if self.format_hint is FormatHint.PCT:
            return format_pct(self.value)
        if self.format_hint is FormatHint.DELTA:
            return format_delta(self.value)
        if self.format_hint is FormatHint.INT:
            return format_int(self.value)
        return str(self.value)

    def tokens(self) -> set[str]:
        """Every numeric token this fact licenses in the artefact."""
        found = {m.group(0) for m in NUMBER.finditer(self.display)}
        if self.unit:
            found.update(m.group(0) for m in NUMBER.finditer(self.unit))
        return found


# --------------------------------------------------------------------------
# The report tree.
# --------------------------------------------------------------------------


class NarrativeBlock(BaseModel):
    """Prose. The only block a model writes, and it may not contain a figure.

    ``role`` separates a statement from a recommendation because they are
    governed differently: §6 forbids an UNASSESSED fact being the basis of a
    recommendation, while stating that a measure is unavailable is exactly what
    an honest report should do.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["narrative"] = "narrative"
    text: str
    fact_refs: List[str] = Field(default_factory=list)
    role: Literal["statement", "recommendation"] = "statement"

    @field_validator("text")
    @classmethod
    def _no_literal_numbers(cls, value: str) -> str:
        """The schema has no numeric field; this closes the prose loophole.

        A digit reaches a sentence only through a ``{{F0042}}`` placeholder, so
        the placeholders are removed and anything numeric left over is a figure
        the model invented.
        """
        stripped = PLACEHOLDER.sub(" ", value)
        leaked = {m.group(0) for m in NUMBER.finditer(stripped)}
        if leaked:
            # The offending sentence is quoted back, not just the digits it
            # yielded. A composer told only that it emitted "001, 03, 87" cannot
            # act on that — those are fragments of an identifier it wrote, and
            # it needs to see which words to remove.
            raise ValueError(
                "a narrative block may not contain a literal number "
                f"({', '.join(sorted(leaked))}) — found in {value[:120]!r}. "
                "Reference a fact as {{F0042}}; never write an identifier of "
                "any kind (a fact id, a finding id) into prose")
        return value

    def placeholders(self) -> List[str]:
        return [m.group(1) for m in PLACEHOLDER.finditer(self.text)]


class MetricBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["metric"] = "metric"
    fact_ref: str
    emphasis: Literal["primary", "secondary"] = "secondary"


class ChartSeries(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    fact_refs: List[str]


class ChartBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["chart"] = "chart"
    chart_type: Literal["bar", "waterfall", "line", "donut"]
    series: List[ChartSeries]


class TableBlock(BaseModel):
    """Cells are fact ids or literal labels — never literal figures.

    A cell that parses as a number is refused for the same reason a narrative
    digit is: it would be a figure nothing traces.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["table"] = "table"
    columns: List[str]
    rows: List[List[str]] = Field(default_factory=list)

    @field_validator("columns")
    @classmethod
    def _a_table_needs_columns(cls, columns: List[str]) -> List[str]:
        """Zero columns is not an empty table, it is not a table.

        The renderer divides the available width by the column count, so a
        column-less table reached python-pptx and came back as
        ``ZeroDivisionError`` — a composition that had passed every semantic
        check bringing down the render stage. Refused here so it becomes a
        corrective retry instead of a traceback.
        """
        if not columns:
            raise ValueError(
                "a table needs at least one column; give it column headings or "
                "use a different block")
        return columns

    @model_validator(mode="after")
    def _rows_match_the_columns(self) -> "TableBlock":
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"row {index} has {len(row)} cell(s) but the table declares "
                    f"{width} column(s); a ragged row renders against the wrong "
                    f"headings")
        return self

    @field_validator("rows")
    @classmethod
    def _cells_are_refs_or_labels(cls, rows: List[List[str]]) -> List[List[str]]:
        """Normalise ``{{F0003}}`` to ``F0003``, then refuse anything numeric.

        Accepting the placeholder spelling as well as the bare id is not
        laxity — it is one syntax instead of two. A composer told to write
        ``{{F0003}}`` in prose will write it in a cell too, and rejecting that
        threw away an otherwise correct report with the actively misleading
        message "carries a literal figure": the cell traced perfectly well, it
        was simply spelled the other way. Observed live.
        """
        normalised: List[List[str]] = []
        for row in rows:
            cells: List[str] = []
            for cell in row:
                placeholder = PLACEHOLDER.fullmatch(cell.strip())
                if placeholder:
                    cells.append(placeholder.group(1))
                    continue
                if FACT_ID.match(cell):
                    cells.append(cell)
                    continue
                if NUMBER.search(cell):
                    raise ValueError(
                        f"table cell {cell!r} carries a literal figure; use a "
                        "fact id (F0003) so the value traces to the pack")
                cells.append(cell)
            normalised.append(cells)
        return normalised


class FindingListBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["finding_list"] = "finding_list"
    finding_refs: List[str] = Field(default_factory=list)


# Tagged on ``type``, and that tag is load-bearing rather than tidy.
#
# As an untagged union this silently accepted ``{}`` as a FindingListBlock —
# every field on that block has a default, so an empty object is a valid
# instance of it, and an untagged union falls through to whichever member
# matches. Live, the model emitted twenty-one empty objects and got back a
# report of twenty-one empty finding lists, which the post-check then passed
# because a report containing no figures has nothing untraced in it.
#
# Tagging makes the absent ``type`` an error instead of a silent reroute, and it
# makes the JSON schema handed to the grammar say so too, so the decoder cannot
# emit an untagged object in the first place. The Python constructors are
# unaffected: ``MetricBlock(fact_ref="F0001")`` still fills the tag from its
# default.
Block = Annotated[
    Union[NarrativeBlock, MetricBlock, ChartBlock, TableBlock, FindingListBlock],
    Field(discriminator="type"),
]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    blocks: List[Block] = Field(default_factory=list)


class ReportAST(BaseModel):
    """The composed report, before it is any particular file format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections: List[Section] = Field(default_factory=list)

    def fact_refs(self) -> set[str]:
        """Every fact this tree claims to rely on, from every block kind."""
        refs: set[str] = set()
        for section in self.sections:
            for block in section.blocks:
                if isinstance(block, NarrativeBlock):
                    refs.update(block.fact_refs)
                    refs.update(block.placeholders())
                elif isinstance(block, MetricBlock):
                    refs.add(block.fact_ref)
                elif isinstance(block, ChartBlock):
                    for series in block.series:
                        refs.update(series.fact_refs)
                elif isinstance(block, TableBlock):
                    for row in block.rows:
                        refs.update(cell for cell in row if FACT_ID.match(cell))
        return refs


# --------------------------------------------------------------------------
# The pack.
# --------------------------------------------------------------------------


def canonical_hash(payload: Any) -> str:
    """A stable sha256 over a JSON-shaped value.

    Sorted keys and no incidental whitespace, so the same content hashes the
    same regardless of dict ordering or how it was built. ``default=str`` so a
    Decimal or datetime hashes by its exact text rather than a float that lost
    the last digit.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class FactPack(BaseModel):
    """An immutable, addressable snapshot of everything a report may state.

    ``hash`` covers the facts, the findings and the scope — everything that
    would change the report — but not ``generated_at``, which changes on every
    build and would make two identical packs look different. That is what makes
    DoD3's "same scope + as-of → identical hash" a real check rather than a
    tautology about clocks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pack_id: str
    tenant_id: str = "default"
    report_type_id: str
    scope: Dict[str, Any]
    as_of: str
    generated_at: datetime
    generated_by: str
    facts: List[FactEntry] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hash(self) -> str:
        return canonical_hash({
            "report_type_id": self.report_type_id,
            "tenant_id": self.tenant_id,
            "scope": self.scope,
            "as_of": self.as_of,
            "facts": [
                f.model_dump(mode="json", exclude={"display"}) for f in self.facts
            ],
            "findings": [f.model_dump(mode="json") for f in self.findings],
        })

    def fact(self, fact_id: str) -> Optional[FactEntry]:
        for entry in self.facts:
            if entry.fact_id == fact_id:
                return entry
        return None

    def fact_by_label(self, label: str) -> Optional[FactEntry]:
        """Look a fact up by what it is called.

        For tests and inspection only. Nothing in the rendering path may address
        a fact this way: labels are prose and change, ids are the contract.
        """
        for entry in self.facts:
            if entry.label == label:
                return entry
        return None

    def quotable_tokens(self) -> set[str]:
        """Every numeric token the rendered artefact is allowed to contain.

        The facts' rendered forms, plus the scope line — the period and the
        population are legitimately printable, and refusing them would fail a
        report for saying which quarter it covers. Lifted from
        ``AnalyticAnswer.quotable_numbers``, which solved this first.
        """
        tokens: set[str] = set()
        for entry in self.facts:
            tokens |= entry.tokens()
        for value in self.scope.values():
            tokens.update(m.group(0) for m in NUMBER.finditer(str(value)))
        # The as-of date is part of the scope a reader checks, even though it is
        # carried beside the scope dict rather than inside it. A report that
        # states the date it was measured to is not quoting an unsourced figure.
        tokens.update(m.group(0) for m in NUMBER.finditer(self.as_of))
        return tokens

    def blocking_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.blocks_release]

    # -- persistence -------------------------------------------------------
    #
    # ``display`` and ``hash`` are derived, and a derived field written back as
    # an input is how a stored pack comes to disagree with itself: someone edits
    # the stored ``display``, the value stays put, and the artefact quotes a
    # figure the fact does not hold. So they are excluded from what is stored,
    # and the hash is kept ALONGSIDE the pack rather than inside it — which
    # turns reloading into a tamper check for free.

    def stored(self) -> Dict[str, Any]:
        """The pack as it should be persisted: inputs, plus the hash as a seal."""
        return {
            "pack_hash": self.hash,
            "pack": self.model_dump(
                mode="json",
                exclude={"hash": True, "facts": {"__all__": {"display"}}},
            ),
        }

    @classmethod
    def from_stored(cls, payload: Dict[str, Any]) -> "FactPack":
        """Rebuild a pack and verify it is the one that was stored.

        A pack whose recomputed hash disagrees with its seal is not loaded. It
        is the same figure-level guarantee the artefact carries, applied one
        layer earlier: a report regenerated from an edited pack would be
        reproducible and wrong, which is worse than not reproducible.
        """
        pack = cls.model_validate(payload["pack"])
        sealed = payload.get("pack_hash")
        if sealed and sealed != pack.hash:
            raise ValueError(
                f"stored pack {pack.pack_id} does not match its recorded hash "
                f"(sealed {sealed[:12]}, recomputed {pack.hash[:12]}) — it has "
                f"been altered since it was written")
        return pack
