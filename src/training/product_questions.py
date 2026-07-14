"""The product-understanding question set.

`eval_gate` measures whether AgentNick reads a DOCUMENT correctly. That is a different axis
from the one we care about here: whether it understands the PRODUCT — how a document travels,
why a thing is or is not visible, what an entity means, where a user goes to see it. A model
can score 0.9 on extraction and still be unable to tell a buyer why their quote never showed
up. Nothing in this repo measured that, so this is the set that does.

Every expected fact is taken from a real source of truth, never invented:

  * `resources/knowledge/platform_ontology.yaml` — the platform's model of itself
  * the live `bp_sqldb` — the deals, discrepancies and rankings that actually exist

Answers are scored on FACT COVERAGE, not on a model's opinion of another model. AgentNick is
the only base model here, so an LLM judge would be AgentNick grading its own homework. Each
fact carries the surface forms a correct answer could reasonably use; the fact is covered if
any of them appears. That is blunt, and it is reproducible, and it cannot flatter itself.

The facts are stated in PRODUCT language on purpose. "Two documents with the same invoice
number collapse into one record" is the fact. "_stg is keyed on the business id" is the same
fact said as architecture — and under the output-safety rule saying it that way is a failure,
not a pass. So an answer that leaks internals scores ZERO for that question no matter how
correct it is. Knowing without saying is the whole point, and this is where it is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Fact:
    """One thing a correct answer has to convey, and the ways it might say it."""

    name: str
    aliases: List[str]


@dataclass(frozen=True)
class Question:
    id: str
    category: str  # flow | gate | entity | screen | why | gap
    question: str
    facts: List[Fact] = field(default_factory=list)
    source: str = ""  # where the gold came from — so it can be re-checked


QUESTIONS: List[Question] = [
    # ---------------------------------------------------------------- flow
    Question(
        id="flow_upload_to_visible",
        category="flow",
        question=(
            "I uploaded an invoice a few minutes ago and I still can't see it anywhere in "
            "the product. Walk me through what actually happens to a document after I upload "
            "it, and why it might not be showing yet."
        ),
        facts=[
            Fact("read_automatically", ["read automatically", "automatically read",
                                        "processed automatically", "we read it",
                                        "extract", "reads the document", "picked up"]),
            Fact("checked_before_visible", ["check", "checked", "review", "confidence",
                                            "quality", "verified", "validated"]),
            Fact("not_visible_until_passes", ["not visible until", "only appears once",
                                              "won't appear", "invisible", "does not appear",
                                              "hidden until", "until it passes",
                                              "held", "on hold"]),
        ],
        source="platform_ontology.yaml processes.document_ingest (+ promotion)",
    ),
    Question(
        id="flow_upload_where",
        category="screen",
        question="Where do I upload a purchase order? Give me the steps.",
        facts=[
            Fact("analyse_or_opportunity", ["analyse", "find an opportunity",
                                            "analyse documents"]),
            Fact("drop_or_attach", ["drop", "drag", "attach", "paperclip", "upload button",
                                    "choose files", "dropzone"]),
        ],
        source="platform_ontology.yaml screens.analyse.how_to",
    ),

    # ---------------------------------------------------------------- gate
    Question(
        id="gate_invoice_needs_po",
        category="gate",
        question=(
            "My invoice extracted fine — the numbers all look right — but it never showed up "
            "in the product. Why would that happen?"
        ),
        facts=[
            Fact("needs_matching_po", ["purchase order", "po ", "matching order",
                                       "parent order", "its po", "a po"]),
            Fact("held_for_human", ["held", "hold", "review", "someone needs to",
                                    "manual", "flagged", "waiting"]),
        ],
        source="platform_ontology.yaml processes.promotion.gate_invoice",
    ),
    Question(
        id="gate_quote_no_po",
        category="gate",
        question=(
            "Does a quote need a matching purchase order before it becomes visible, the same "
            "way an invoice does?"
        ),
        facts=[
            Fact("no", ["no", "does not", "doesn't", "not require", "no purchase order",
                        "without a purchase order"]),
            Fact("quote_comes_first", ["before", "raised first", "earlier", "precedes",
                                       "comes first", "prior to", "not yet exist"]),
        ],
        source="platform_ontology.yaml processes.promotion.gate_quote",
    ),

    # ---------------------------------------------------------------- why
    Question(
        id="why_no_supplier_ranking",
        category="why",
        question=(
            "I opened one of my deals and there is no supplier ranking on it at all. Why "
            "would a deal have no ranking?"
        ),
        facts=[
            Fact("needs_two_bidders", ["two", "2", "more than one", "multiple suppliers",
                                       "competing", "at least two", "several suppliers",
                                       "only one supplier", "single supplier"]),
            Fact("ranks_quotes", ["quote", "bid", "offer"]),
        ],
        source="platform_ontology.yaml agents.supplier_ranking ('Needs 2+ bidders')",
    ),
    Question(
        id="why_duplicate_invoice_number",
        category="why",
        question=(
            "Two different people uploaded two different documents that happen to carry the "
            "same invoice number. What does the product do with them?"
        ),
        facts=[
            Fact("collapse_to_one", ["one", "single", "same record", "merge", "collapse",
                                     "combined", "overwrite", "replaces", "one record"]),
            Fact("later_wins", ["later", "latest", "most recent", "newer", "last one",
                                "overwrites the earlier", "replaces"]),
        ],
        source="platform_ontology.yaml known_gaps.doc_identity",
    ),
    Question(
        id="why_money_null",
        category="why",
        question=(
            "One of my invoices came through with a blank total. Why wouldn't the system "
            "just work the total out, or estimate it?"
        ),
        facts=[
            # Contractions matter: the model writes "doesn't guess", not "does not guess".
            # The alias list missed that and scored a correct answer at zero.
            Fact("never_invented", ["never", "not invent", "won't guess", "does not guess",
                                    "doesn't guess", "don't guess", "no guess",
                                    "not estimate", "doesn't estimate", "or estimate",
                                    "never synthesise", "never synthesize", "not make up",
                                    "not fabricate"]),
            Fact("left_blank_and_flagged", ["blank", "empty", "null", "left as is",
                                            "flag", "raise", "discrepancy", "review",
                                            "for you to check"]),
        ],
        source="platform_ontology.yaml processes.money_reading",
    ),

    # ---------------------------------------------------------------- entity
    Question(
        id="entity_deal",
        category="entity",
        question="What is a 'deal' in this product, in plain terms?",
        facts=[
            Fact("groups_documents", ["group", "groups", "collection", "brings together",
                                      "ties together", "links", "related documents",
                                      "set of documents"]),
            Fact("across_doc_types", ["quote", "purchase order", "invoice", "po"]),
        ],
        source="live bp_sqldb: deal DEALV2-5206556 spans quote/PO/invoice for one supplier",
    ),
    Question(
        id="entity_discrepancy",
        category="entity",
        question="What is a discrepancy, and what am I supposed to do about one?",
        facts=[
            Fact("mismatch", ["mismatch", "disagree", "contradict", "does not match",
                              "doesn't match", "inconsistent", "conflict", "differs"]),
            Fact("human_reviews", ["review", "check", "look at", "action", "resolve",
                                   "confirm", "you decide", "raised for you"]),
        ],
        source="platform_ontology.yaml agents.discrepancy_detection (+ 688 live rows)",
    ),

    # ---------------------------------------------------------------- gap
    Question(
        id="gap_no_upload_progress",
        category="gap",
        question=(
            "Is there a screen where I can watch my upload's progress and see whether it "
            "succeeded or failed?"
        ),
        facts=[
            Fact("honest_no", ["no", "not", "cannot", "can't", "there isn't", "no screen",
                               "unable"]),
        ],
        source="platform_ontology.yaml known_gaps.no_ingest_status_ui",
    ),
    Question(
        id="gap_same_filename",
        category="gap",
        question=(
            "If my colleague and I both upload a file called 'quote.pdf', is there any "
            "problem with that?"
        ),
        facts=[
            Fact("they_collide", ["overwrite", "overwrites", "replace", "replaces", "clash",
                                  "collide", "conflict", "same name", "lost", "each other"]),
        ],
        source="platform_ontology.yaml known_gaps.s3_key_collision",
    ),
]


def by_category() -> dict:
    out: dict = {}
    for q in QUESTIONS:
        out.setdefault(q.category, []).append(q)
    return out
