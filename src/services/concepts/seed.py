"""The seeded vocabulary, matching deploy/sql/2026-10-01_concept_vocabulary.sql.

Why the duplication: the same reason uom.py keeps _CANONICAL alongside
proc.bp_uom_canonical. Code needs a vocabulary to fall back on when the table
cannot be read, because a resolver that recognises nothing marks every document
unknown — and those absences then look like the documents said nothing. A test
(tests/services/concepts/test_concept_table.py) fails the moment the two diverge.

ASSUMPTION, unconfirmed: the eight role names come from the build spec's
default set (§3.1), which the spec itself marks [CONFIRM]. They are data, so
renaming one is an UPDATE and a seed edit, not a rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class Concept:
    concept_code: str
    domain: str
    definition: str
    not_to_be_confused_with: Tuple[str, ...] = ()
    status: str = "active"
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class DocumentType:
    concept_code: str
    role: str
    default_parent_type: Optional[str]
    execution_mode: Optional[str]
    aliases: Tuple[str, ...]
    identifiers: Tuple[Mapping[str, Optional[str]], ...]
    structural_signals: Tuple[str, ...]
    #: Which of the four physical pipelines ingests this type. None means the
    #: type is recognised but nothing can ingest it yet — an honest state.
    pipeline_doc_type: Optional[str]
    status: str = "active"
    #: A structure whose purpose is to sit beneath a parent only claims a page
    #: that names that parent. Data, so flagging another structure later is an
    #: UPDATE rather than a code edit.
    requires_parent_evidence: bool = False
    #: Matchable phrases that count as naming a parent, folded like aliases.
    parent_evidence_phrases: Tuple[str, ...] = ()


_ROLES = (
    ("role.framework", "An umbrella agreement under which later contracts are called off."),
    ("role.master", "Governs a relationship and is itself the contract for the work it covers."),
    ("role.transaction", "Orders, confirms or bills a specific quantity or amount."),
    ("role.variation", "Changes the terms of a document that already exists."),
    ("role.attachment", "Has no force alone; takes effect by being incorporated into another document."),
    ("role.notice", "Communicates a fact or an intention; creates no new obligation by itself."),
    ("role.termination", "Ends a document that is already in force."),
    ("role.supporting", "Evidence or working papers around a relationship; not itself binding."),
)

_LINK_TYPES = (
    ("link.calls_off", "The source orders work under the target framework."),
    ("link.governed_by", "The source's terms are set by the target."),
    ("link.incorporates", "The source pulls the target's terms into itself by reference."),
    ("link.varies", "The source changes the target's terms."),
    ("link.supersedes", "The source replaces the target in full."),
    ("link.terminates", "The source ends the target."),
    ("link.attaches_to", "The source is an attachment of the target."),
    ("link.references", "The source mentions the target without changing it."),
)

_EXECUTION_MODES = (
    ("exec.bilateral", "Signed by two parties."),
    ("exec.unilateral", "Issued and signed by one party."),
    ("exec.multilateral", "Signed by three or more parties."),
    ("exec.incorporated", "Not separately executed; takes effect through the document that incorporates it."),
)

_EVENT_KINDS = (
    ("event.signed", "The parties executed the document."),
    ("event.effective", "The document's terms began to apply."),
    ("event.varied", "The document's terms were changed."),
    ("event.renewed", "The document's term was extended on its own renewal terms."),
    ("event.extended", "The document's term was lengthened other than by renewal."),
    ("event.expired", "The document's term ran out."),
    ("event.terminated", "The document was ended before its term ran out."),
    ("event.superseded", "The document was replaced by another."),
)


#: (concept_code, definition, not_to_be_confused_with)
_DOCUMENT_TYPE_CONCEPTS = (
    ("doctype.framework_agreement",
     "Sets terms for future call-offs but orders nothing itself.",
     ("doctype.master_agreement",)),
    ("doctype.master_agreement",
     "Governs a supplier relationship and is the contract for work done under it.",
     ("doctype.framework_agreement", "doctype.call_off_contract")),
    ("doctype.sow",
     "Defines the deliverables, timescale and price of a specific piece of work under a master agreement.",
     ("doctype.call_off_contract", "doctype.order")),
    ("doctype.call_off_contract",
     "The contract formed when work is ordered under a framework; incorporates the framework's terms and sets their precedence.",
     ("doctype.order", "doctype.sow", "doctype.framework_agreement",
      "doctype.order_form")),
    ("doctype.order",
     "Instructs a supplier to deliver a stated quantity at a stated price.",
     ("doctype.call_off_contract", "doctype.sales_order")),
    ("doctype.invoice", "Demands payment for goods or services supplied.", ()),
    ("doctype.goods_receipt", "Records what was physically delivered and accepted.", ()),
    ("doctype.quote", "Offers a price before any order exists.", ()),
    ("doctype.variation",
     "Changes the terms of an existing contract.",
     ("doctype.addendum", "doctype.ccn")),
    ("doctype.schedule",
     "A numbered part of a contract that has no force on its own.",
     ("doctype.addendum", "doctype.sla")),
    ("doctype.addendum",
     "Adds to a contract after signature without replacing it.",
     ("doctype.variation", "doctype.schedule")),
    ("doctype.ccn",
     "Change control note: records an agreed change under a contract's own change procedure.",
     ("doctype.variation",)),
    ("doctype.termination_notice",
     "Ends a contract that is in force.",
     ("doctype.notice_general",)),
    ("doctype.notice_general",
     "Communicates a fact or intention under a contract without ending or changing it.",
     ("doctype.termination_notice",)),
    ("doctype.nda", "Binds the parties to keep information confidential.", ()),
    ("doctype.sla",
     "States service levels and remedies; normally a schedule to an agreement rather than a contract alone.",
     ("doctype.schedule",)),
    ("doctype.service_agreement",
     "Contracts for the supply of a service on stated terms.",
     ("doctype.master_agreement", "doctype.consulting_agreement")),
    ("doctype.consulting_agreement",
     "Contracts for advisory work, usually against time and materials or a retainer.",
     ("doctype.service_agreement",)),
    ("doctype.contract_unspecified",
     "A contract whose kind the document does not state. Recorded as itself rather than guessed at.",
     ()),
    # Observed in proc.bp_contract_master and not yet understood. Proposed, so
    # it is counted and visible but never resolves.
    ("doctype.policy_document",
     "Observed as a contract_type value; what it denotes here is not yet established.",
     ()),
    ("doctype.order_form",
     "Orders specific goods or services on the terms of an agreement it names.",
     ("doctype.call_off_contract", "doctype.quote", "doctype.order")),
    ("doctype.sales_order",
     "The supplier's own confirmation of an order it has received.",
     ("doctype.order", "doctype.invoice")),
)


CONCEPTS: Mapping[str, Concept] = {
    c.concept_code: c
    for c in (
        *(Concept(code, "RELATIONSHIP_ROLE", defn) for code, defn in _ROLES),
        *(Concept(code, "LINK_TYPE", defn) for code, defn in _LINK_TYPES),
        *(Concept(code, "EXECUTION_MODE", defn) for code, defn in _EXECUTION_MODES),
        *(Concept(code, "EVENT_KIND", defn) for code, defn in _EVENT_KINDS),
        *(
            Concept(code, "DOCUMENT_TYPE", defn, confused,
                    status="proposed" if code == "doctype.policy_document" else "active")
            for code, defn, confused in _DOCUMENT_TYPE_CONCEPTS
        ),
    )
}

DOCUMENT_TYPES: Mapping[str, DocumentType] = {
    dt.concept_code: dt for dt in (
        DocumentType(
            "doctype.framework_agreement", "role.framework", None, "exec.bilateral",
            # "framework" stays last: matches the migration's append order, and
            # the seed-vs-table drift test compares aliases as ordered lists.
            # It was dropped in Task 1 to satisfy a guard that compared aliases
            # against EVERY concept's local name, including role.framework —
            # but the alias index is built from bp_document_type rows alone, so
            # a role code can never contest an alias. The guard is now scoped to
            # DOCUMENT_TYPE codes and the alias is back.
            ("framework agreement", "framework contract", "framework"),
            ({"field": "framework_ref", "pattern": r"^[A-Z]{2}\d{4,6}$", "parent_type": None},),
            ("sets terms without ordering", "names a call-off procedure"),
            "contract",
        ),
        DocumentType(
            "doctype.master_agreement", "role.master", None, "exec.bilateral",
            ("master agreement", "msa", "master service agreement",
             "master services agreement"),
            ({"field": "contract_id", "pattern": None, "parent_type": None},),
            ("recites the parties and the relationship", "numbered clauses"),
            "contract",
        ),
        DocumentType(
            "doctype.sow", "role.master", "doctype.master_agreement", "exec.bilateral",
            ("sow", "statement of work", "work order", "task order"),
            ({"field": "contract_id", "pattern": None, "parent_type": "doctype.master_agreement"},),
            ("lists deliverables and milestones", "names the agreement it sits under"),
            "contract",
        ),
        DocumentType(
            "doctype.call_off_contract", "role.master", "doctype.framework_agreement",
            "exec.bilateral",
            # NOT "order form": it is a real name for a call-off, but on this corpus
            # it is also the title cell every quote-template workbook carries, and it
            # produced 12 false disagreements out of 12 uses — no true positive. See
            # specs/2026-10-01-document-relationship-layer-rulings.md. Re-adding it
            # needs a way to tell a quote template's heading from a real call-off's,
            # which needs the golden-set documents. A real call-off still matches on
            # "call-off contract" / "call off contract" / "call-off".
            ("call-off contract", "call off contract", "call-off"),
            ({"field": "framework_ref", "pattern": None, "parent_type": "doctype.framework_agreement"},),
            ("lists incorporated documents", "states an order of precedence"),
            "contract",
        ),
        DocumentType(
            "doctype.order", "role.transaction", "doctype.call_off_contract",
            "exec.unilateral",
            ("purchase order", "purchase_order", "purchaseorder", "po", "order"),
            ({"field": "po_id", "pattern": r"^(?:PO)?\d{4,10}$", "parent_type": None},),
            ("line items with quantities and a total", "ship-to address"),
            "purchase_order",
        ),
        DocumentType(
            "doctype.invoice", "role.transaction", "doctype.order", "exec.unilateral",
            ("invoice", "tax invoice", "bill"),
            ({"field": "invoice_id", "pattern": None, "parent_type": None},
             {"field": "po_id", "pattern": None, "parent_type": "doctype.order"}),
            ("amount due and payment terms", "bill-to address"),
            "invoice",
        ),
        DocumentType(
            # Alias order is part of the data: the migration lists them
            # identically, and test_document_type_rows_equal_the_seed_column_for_column
            # compares aliases as an ORDERED list.
            "doctype.goods_receipt", "role.transaction", "doctype.order",
            "exec.unilateral",
            ("goods receipt", "goods received note", "grn", "delivery note",
             "despatch note", "dispatch note", "advice note", "packing list",
             "packing slip", "proof of delivery", "pod"),
            ({"field": "grn_id", "pattern": None, "parent_type": None},
             {"field": "po_id", "pattern": None, "parent_type": "doctype.order"}),
            ("quantities with no prices", "signed for on receipt",
             "a carrier, vehicle or consignment reference"),
            "goods_receipt",
        ),
        DocumentType(
            "doctype.quote", "role.supporting", None, "exec.unilateral",
            # "quotes" stays last: matches the migration's append order, and the
            # seed-vs-table drift test compares aliases as ordered lists.
            ("quote", "quotation", "estimate", "price quotation", "quotes"),
            ({"field": "quote_id", "pattern": None, "parent_type": None},),
            ("validity or expiry date", "prices with no order reference"),
            "quote",
        ),
        DocumentType(
            "doctype.variation", "role.variation", None, "exec.bilateral",
            ("variation", "variation form", "amendment", "avenant", "deed of variation"),
            ({"field": "amendment_ref", "pattern": None, "parent_type": None},),
            ("names the document it changes", "states what the change is"),
            "contract",
        ),
        DocumentType(
            "doctype.schedule", "role.attachment", None, "exec.incorporated",
            ("schedule", "annex", "appendix", "exhibit"),
            (),
            ("numbered as part of another document", "no signature block"),
            "contract",
        ),
        DocumentType(
            "doctype.addendum", "role.variation", None, "exec.bilateral",
            ("addendum", "supplemental agreement"),
            (),
            ("adds terms after signature", "names the document it supplements"),
            "contract",
        ),
        DocumentType(
            "doctype.ccn", "role.variation", None, "exec.bilateral",
            ("ccn", "change control note", "change note", "change request"),
            (),
            ("cites the contract's change procedure", "states cost and time impact"),
            "contract",
        ),
        DocumentType(
            "doctype.termination_notice", "role.termination", None, "exec.unilateral",
            ("termination notice", "notice of termination"),
            (),
            ("states a termination date", "cites a termination clause"),
            "contract",
        ),
        DocumentType(
            "doctype.notice_general", "role.notice", None, "exec.unilateral",
            # "notice" stays last: matches the migration's append order, and the
            # seed-vs-table drift test compares aliases as ordered lists. Bare
            # "notice" was renamed to "general notice" in Task 1 against the
            # same over-wide guard (role.notice); restored for the same reason,
            # and "general notice" is kept beside it. doctype.termination_notice
            # claims "termination notice"/"notice of termination", NOT bare
            # "notice", so there is no two-owner collision.
            ("general notice", "notice"),
            (),
            ("cites a notice clause", "creates no new obligation"),
            None,
        ),
        DocumentType(
            "doctype.nda", "role.master", None, "exec.bilateral",
            ("nda", "non-disclosure agreement", "confidentiality agreement"),
            (),
            ("defines confidential information", "states a confidentiality period"),
            "contract",
        ),
        DocumentType(
            "doctype.sla", "role.attachment", None, "exec.incorporated",
            ("sla", "service level agreement"),
            (),
            ("service levels with targets", "remedies or service credits"),
            "contract",
        ),
        DocumentType(
            "doctype.service_agreement", "role.master", None, "exec.bilateral",
            ("service agreement", "service contract", "services agreement"),
            (),
            ("describes a service and its term", "numbered clauses"),
            "contract",
        ),
        DocumentType(
            "doctype.consulting_agreement", "role.master", None, "exec.bilateral",
            ("consulting", "consulting agreement", "consultancy agreement"),
            (),
            ("rates or a retainer", "named consultants or roles"),
            "contract",
        ),
        DocumentType(
            "doctype.contract_unspecified", "role.master", None, None,
            # "contracts" stays last: matches the migration's append order, and
            # the seed-vs-table drift test compares aliases as ordered lists.
            ("contract", "agreement", "contracts"),
            ({"field": "contract_id", "pattern": None, "parent_type": None},),
            ("numbered clauses", "a signature block"),
            "contract",
        ),
        DocumentType(
            "doctype.policy_document", "role.supporting", None, None,
            ("policy",),
            (), (), None,
            status="proposed",
        ),
        DocumentType(
            # NOT an alias of doctype.call_off_contract -- that alias titled every
            # quote-template workbook and gave 12 false disagreements out of 12
            # uses. As its own structure with requires_parent_evidence it claims
            # only a document that names the agreement it sits under, which is the
            # build spec's own distinction: "'order form' means one thing under a
            # framework and another on its own".
            "doctype.order_form", "role.master", "doctype.framework_agreement",
            "exec.bilateral",
            ("order form",),
            ({"field": "framework_ref", "pattern": None,
              "parent_type": "doctype.framework_agreement"},),
            ("lists incorporated documents", "states an order of precedence"),
            "contract",
            requires_parent_evidence=True,
            # NOT bare "incorporated": phrases match whole-word, and it would
            # still match the supplier name "Acme Incorporated". See the
            # migration's comment for the measurement.
            parent_evidence_phrases=(
                "framework", "order of precedence",
                "incorporated into", "incorporated by reference", "call off",
                "framework agreement no", "framework agreement number",
                "framework agreement ref", "master agreement no",
                "master agreement number", "master agreement ref",
                "parent agreement no", "parent contract no", "principal agreement no",
            ),
        ),
        DocumentType(
            # The supplier's mirror of a purchase order, so it extracts with the
            # PO schema (Nick's ruling, 2026-10-02): lines, quantities, a total.
            "doctype.sales_order", "role.transaction", "doctype.order",
            "exec.unilateral",
            ("sales order", "sales order acknowledgement", "order acknowledgement"),
            ({"field": "po_id", "pattern": None, "parent_type": "doctype.order"},),
            ("line items with quantities and a total",),
            "purchase_order",
        ),
    )
}
