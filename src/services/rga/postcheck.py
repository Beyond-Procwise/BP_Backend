"""The gate. Runs on the rendered artefact, and fails closed.

WHY THE ARTEFACT AND NOT THE AST

Because the AST is what was intended and the artefact is what the reader gets,
and every interesting failure lives between the two: a renderer that formats a
figure differently from the fact it came from, a substitution that silently
dropped a badge, a template that prints something nobody put in the tree. §6 of
the brief is explicit, and it is right — checking the AST would be checking the
model's homework against the model's own answer sheet.

THE TOKEN CHECK, AND WHY IT STRIPS BEFORE IT SCANS

Every numeric token in the deck must be one the pack licenses. Three kinds of
number legitimately appear that are not measurements, and each is removed
before the scan rather than added to the allowed set:

  * **Provenance chrome.** A sha256 is mostly digits. Counting a hash as a
    fabricated figure would fail every report for disclosing its own identity.
  * **Fact labels.** "Q1 2026 invoiced spend" carries 1 and 2026. A label is
    part of the question, not the answer.
  * **Footnote markers.** ``[F0003]`` is a reference to a figure, not a figure.

This is the technique ``AnalyticAnswer.unquoted_numbers`` established — strip
the names, then compare what is left — and it is used here for the same reason:
adding those tokens to the allowed set would license them everywhere, so
naming a supplier "Kestrel Supplies 8" would license "8% of spend".
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from src.services.analytics.models import Confidence
from src.services.rga.models import (
    NUMBER,
    FactPack,
    Finding,
    FindingCode,
    NarrativeBlock,
    Origin,
    ReportAST,
    Severity,
)
from src.services.rga.render import RenderedArtefact
from src.services.rga.style import StyleBrief

logger = logging.getLogger(__name__)

AGENT = "rga_postcheck"
PHASE = "reporting"

_MARKER = re.compile(r"\[F\d{4}\]")


class PostCheckResult:
    """What the gate decided, and everything it objected to."""

    def __init__(self, findings: List[Finding]) -> None:
        self.findings = findings

    @property
    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.blocks_release]

    @property
    def passed(self) -> bool:
        return not self.blocking

    def __repr__(self) -> str:
        return (f"PostCheckResult(passed={self.passed}, "
                f"findings={[f.code.value for f in self.findings]})")


def _strip(text: str, artefact: RenderedArtefact, pack: FactPack) -> str:
    """Remove everything that is a reference rather than a measurement."""
    out = _MARKER.sub(" ", text)
    # A footnote is made entirely of references: the fact's label, the query
    # that produced it and the audit row it points at. Every one of those
    # carries digits — a pack id, a rate timestamp, a fact number — and none of
    # them is a measurement. They are removed for the same reason the hash is.
    removable = list(artefact.chrome_strings())
    for entry in pack.facts:
        removable.extend((entry.label, entry.derivation, entry.provenance_id))
    # Longest first, so a string that contains another is removed whole.
    for item in sorted((s for s in removable if s), key=len, reverse=True):
        out = out.replace(item, " ")
    return out


def run(
    artefact: RenderedArtefact,
    pack: FactPack,
    ast: ReportAST,
    brief: StyleBrief,
    *,
    emit_audit: bool = True,
    writer: Any = None,
) -> PostCheckResult:
    """Every check in §6, on the bytes that would be released."""

    findings: List[Finding] = []
    seq = [0]

    def raise_finding(code: FindingCode, detail: str, *,
                      severity: Severity = Severity.HIGH,
                      location: Optional[str] = None,
                      blocks: bool = True) -> None:
        seq[0] += 1
        findings.append(Finding(
            finding_id=f"{pack.pack_id}-PC{seq[0]:03d}", code=code,
            severity=severity, detail=detail, location=location,
            blocks_release=blocks,
        ))

    # -- 0. the report must actually say something --------------------------
    #
    # Every other check here is conditional on a figure being present: "if there
    # is a number, it must trace". A report with no numbers therefore satisfies
    # all of them and releases clean. That is not a hypothetical — a live
    # composition returned a deck of empty blocks and this gate passed it, which
    # is the whole reason this check exists and why it runs first.
    referenced = ast.fact_refs()
    if pack.facts and not referenced:
        raise_finding(
            FindingCode.REPORT_STATES_NOTHING,
            f"the report references none of the {len(pack.facts)} measured "
            f"fact(s) in pack {pack.pack_id}; a report that states no figure is "
            f"not a report that passed, it is a report that said nothing")

    # -- 1. every fact_ref in the AST exists in the pack --------------------
    # Ordered before the token scan: an unresolved reference renders as a
    # literal "{{F0099}}", which would otherwise also trip the scan and report
    # the same defect twice under the wrong name.
    for ref in sorted(referenced):
        if pack.fact(ref) is None:
            raise_finding(FindingCode.UNKNOWN_FACT_REF,
                          f"the report references {ref}, which is not in pack "
                          f"{pack.pack_id}")

    # -- 2. no untraced figure in the rendered artefact ---------------------
    from src.services.rga.render import extract_text_for

    chunks = extract_text_for(artefact)
    allowed = pack.quotable_tokens()
    for chunk in chunks:
        stripped = _strip(chunk, artefact, pack)
        for match in NUMBER.finditer(stripped):
            token = match.group(0)
            if token in allowed or token.replace(",", "") in {
                    t.replace(",", "") for t in allowed}:
                continue
            raise_finding(
                FindingCode.REPORT_UNTRACED_FIGURE,
                f"{token!r} appears in the artefact but is not any fact's "
                f"rendered value",
                location=chunk.strip()[:120].replace("\n", " "),
            )

    # -- 3. an UNASSESSED fact may not carry a recommendation ---------------
    for section in ast.sections:
        for block in section.blocks:
            if not isinstance(block, NarrativeBlock):
                continue
            if block.role != "recommendation":
                continue
            for ref in set(block.fact_refs) | set(block.placeholders()):
                entry = pack.fact(ref)
                if entry is not None and entry.confidence is Confidence.UNASSESSED:
                    raise_finding(
                        FindingCode.UNASSESSED_IN_RECOMMENDATION,
                        f"section {section.id!r} recommends on the basis of "
                        f"{ref} ({entry.label}), which is UNASSESSED",
                        location=section.id)

    # -- 4. every drawing of a fact carries that fact's badge ---------------
    #
    # Checked PER FACT and PER FRAME, not per deck. The first version of this
    # searched the whole artefact for the badge string, which meant one fact's
    # disclosure satisfied the check for every other fact — a guard that was
    # green while checking nothing. The renderer's contract is that a fact's
    # marker and its badge appear in the same text frame, so that pairing is
    # what is verified here.
    show_confidence = bool(brief.get("report.style.show_confidence_badges"))

    for entry in pack.facts:
        if entry.fact_id not in referenced:
            continue
        required = []
        if entry.origin is Origin.LEGACY_UNVERIFIED:
            required.append(Origin.LEGACY_UNVERIFIED.value)
        if show_confidence:
            required.append(entry.confidence.value)
        if not required:
            continue

        marker = f"[{entry.fact_id}]"
        drawn = [c for c in chunks if marker in c]
        if not drawn:
            raise_finding(
                FindingCode.MISSING_ORIGIN_BADGE,
                f"{entry.fact_id} ({entry.label}) is referenced by the report "
                f"but is drawn nowhere in the artefact",
                location=entry.fact_id)
            continue
        for badge in required:
            if not any(badge in c for c in drawn):
                raise_finding(
                    FindingCode.MISSING_ORIGIN_BADGE,
                    f"{entry.fact_id} ({entry.label}) is drawn without its "
                    f"{badge} badge; a figure whose standing is not stated "
                    f"beside it reads as one that was measured",
                    location=entry.fact_id)

    # -- 5. every figure on the page can be traced back off the page --------
    #
    # The badge says how well a figure is known; the footnote says where it came
    # from. They are different disclosures and a report can lose one while
    # keeping the other, so the footnote is checked on its own terms: the frame
    # carrying it names the audit row, which is the thing a reader would follow.
    if brief.get("report.style.show_provenance_footnotes"):
        for entry in pack.facts:
            if entry.fact_id not in referenced:
                continue
            marker = f"[{entry.fact_id}]"
            if not any(marker in c and entry.provenance_id in c for c in chunks):
                raise_finding(
                    FindingCode.MISSING_PROVENANCE_FOOTNOTE,
                    f"{entry.fact_id} ({entry.label}) is printed without its "
                    f"provenance footnote; the reader cannot get from the "
                    f"figure back to the record it came from",
                    location=entry.fact_id)

    # -- 6. the style provenance chain is attached --------------------------
    if not artefact.style_provenance:
        raise_finding(FindingCode.MISSING_STYLE_PROVENANCE,
                      "the artefact record carries no style provenance chain, "
                      "so the question 'why did this report look like this' "
                      "has no answer")
    else:
        missing = [k for k in brief.values if k not in artefact.style_provenance]
        if missing:
            raise_finding(
                FindingCode.MISSING_STYLE_PROVENANCE,
                f"{len(missing)} style key(s) resolved without a recorded "
                f"provenance: {', '.join(sorted(missing)[:3])}")

    # -- 7. the reproducibility record is complete --------------------------
    if not artefact.recorded():
        raise_finding(FindingCode.MISSING_HASH_RECORD,
                      "the artefact record is incomplete: it must carry the "
                      "pack hash, the style version, the AST hash and the "
                      "renderer version or it cannot be regenerated")
    elif artefact.pack_hash != pack.hash:
        raise_finding(FindingCode.MISSING_HASH_RECORD,
                      f"the artefact records pack hash {artefact.pack_hash[:12]} "
                      f"but the pack hashes to {pack.hash[:12]}")

    result = PostCheckResult(findings)
    if emit_audit:
        _audit(result, artefact, pack, writer=writer)
    return result


def _audit(result: PostCheckResult, artefact: RenderedArtefact,
           pack: FactPack, *, writer: Any = None) -> None:
    """Record the verdict either way.

    A pass is recorded as well as a failure, for the reason
    ``policy_observation`` gives about its own allows: "we observed no failures"
    and "we were not checking" are otherwise the same observation.
    """
    from src.services.rga import audit

    audit.emit(
        audit.POSTCHECK_PASSED if result.passed else audit.POSTCHECK_FAILED,
        run_id=pack.pack_id,
        agent=AGENT,
        pack_hash=pack.hash,
        status="ok" if result.passed else "blocked",
        summary=f"{len(result.blocking)} blocking, {len(result.findings)} total",
        details={
            "pack_id": pack.pack_id,
            "ast_hash": artefact.ast_hash,
            "style_version": artefact.style_version,
            "renderer": f"{artefact.renderer}/{artefact.renderer_version}",
            "findings": [f.model_dump(mode="json") for f in result.findings],
        },
        writer=writer,
    )
