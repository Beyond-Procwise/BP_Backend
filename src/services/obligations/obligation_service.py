"""Read a contract's prose into grounded, queryable obligations.

Flow:  contract text -> AutoHypergraph (AgentNick) -> grounding guard -> Postgres

Two independent safety nets, in order:
  1. Hyper-Extract prunes hyperedges naming an entity it never extracted (structural).
  2. Our grounding guard drops obligations whose source_quote is not in the document
     (semantic). See grounding.py for why the existing field guard cannot be reused.

Nothing here writes to the field-extraction tables. This layer is additive and cannot
regress field accuracy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from src.services.db import get_conn
from src.services.obligations.agentnick import (
    MAX_WORKERS,
    AgentNickChat,
    AgentNickEmbeddings,
)
from src.services.obligations.grounding import is_quote_grounded
from src.services.obligations.schema import ContractEntity, ContractObligation

log = logging.getLogger(__name__)

STATUS_EXTRACTED = "extracted"
STATUS_NO_GROUNDED = "no_grounded_obligations"
STATUS_FAILED = "failed"

# The library's default edge prompt says nothing about quoting, and the model then writes
# quotes the grounding guard must reject: it stitches several clauses into one "quote"
# (clause_ref "27.5, 27.6, 27.7"), or abbreviates with "...". Both are true obligations lost
# to a bad quote. The fix belongs here, not in a weaker guard.
EDGE_PROMPT = """You are a procurement contract lawyer extracting obligations.

An obligation binds SEVERAL entities at once — the party who must act, the party entitled,
the goods, the trigger, the deadline, the penalty. List ALL of them in `participants`.

RULES FOR source_quote — these are absolute:
1. Copy ONE sentence from the contract, EXACTLY, character for character.
2. NEVER abbreviate. Do not write "..." or "[...]" anywhere in the quote.
3. NEVER join two clauses into one quote. One obligation = one clause = one sentence.
4. `clause_ref` is a SINGLE clause number (e.g. "27.4"), never a list.
5. If you cannot copy an exact sentence for an obligation, DO NOT emit that obligation.

A quote that is not a verbatim sentence from the text below will be discarded, and the
obligation lost. Copy, do not paraphrase.

CHOOSE `type` BY WHAT THE CLAUSE DOES — do not default to one value:
- must_perform  : a party MUST do something ("the Contractor shall deliver")
- entitled_to   : a party MAY do something, or is under no obligation ("the Authority may reject")
- triggered_by  : the duty only arises on some event ("on dispatch, the Contractor shall...")
- penalised_by  : a consequence for failure ONLY (a credit, refund, or termination right)
- risk_transfer : where risk or ownership in the Goods sits

Only use entities from the Known Entities list.

# Known Entities
{known_nodes}

# Contract Text
{source_text}
"""


def _key(name: str) -> str:
    """Normalise an entity name so 'the Contractor' and 'Contractor' are one node."""
    return re.sub(r"\s+", " ", (name or "").strip().lower()).removeprefix("the ").strip(" .,;")


@dataclass
class ObligationRun:
    document_id: str
    entities: List[ContractEntity] = field(default_factory=list)
    obligations: List[ContractObligation] = field(default_factory=list)
    dropped: List[Tuple[ContractObligation, str]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.obligations:
            return STATUS_EXTRACTED
        # Distinct from "extracted 0": we DID find obligations, none survived grounding.
        return STATUS_NO_GROUNDED


def extract_obligations(
    document_id: str, contract_text: str, embedding_model: Any
) -> ObligationRun:
    """Extract grounded obligations from contract prose. Raises if AgentNick is unreachable."""
    if not contract_text or not contract_text.strip():
        raise ValueError(f"{document_id}: no contract text to read")

    from hyperextract.types.hypergraph import AutoHypergraph
    from ontomem import MergeStrategy

    graph = AutoHypergraph(
        node_schema=ContractEntity,
        edge_schema=ContractObligation,
        node_key_extractor=lambda n: _key(n.name),
        # Hyperedges are unordered sets, so participants must be sorted or dedup breaks.
        # clause_ref is in the key because the same clause can yield two relation types.
        edge_key_extractor=lambda e: f"{e.clause_ref}|{e.type}|{sorted(_key(p) for p in e.participants)}",
        nodes_in_edge_extractor=lambda e: tuple(_key(p) for p in e.participants),
        llm_client=AgentNickChat(),
        embedder=AgentNickEmbeddings(embedding_model),
        # The default (MergeStrategy.LLM.BALANCED) has an LLM *rewrite* merged field values,
        # which would rewrite source_quote and destroy verbatim grounding. First-seen wins.
        node_strategy_or_merger=MergeStrategy.KEEP_EXISTING,
        edge_strategy_or_merger=MergeStrategy.KEEP_EXISTING,
        extraction_mode="two_stage",
        prompt_for_edge_extraction=EDGE_PROMPT,
        max_workers=MAX_WORKERS,
    )

    # feed_text() mutates; parse() returns a NEW instance and leaves this one empty,
    # which yields zero obligations with no error at all.
    graph = graph.feed_text(contract_text)

    run = ObligationRun(document_id=document_id, entities=list(graph.nodes))
    for edge in graph.edges:
        if is_quote_grounded(edge.source_quote, contract_text):
            run.obligations.append(edge)
        else:
            run.dropped.append((edge, "source_quote not found verbatim in document"))

    for edge, why in run.dropped:
        log.warning(
            "%s: DROPPED ungrounded obligation %r (clause %s) — %s | quote=%r",
            document_id, edge.name, edge.clause_ref, why, edge.source_quote[:120],
        )
    log.info(
        "%s: %d obligations grounded, %d dropped, status=%s",
        document_id, len(run.obligations), len(run.dropped), run.status,
    )
    return run


def persist_obligations(run: ObligationRun, contract_id: str | None = None) -> int:
    """Write grounded obligations. Returns the number of obligations written."""
    types = {_key(e.name): e.type.value for e in run.entities}

    with get_conn() as conn:
        cur = conn.cursor()
        # Re-extraction of the same document replaces its obligations rather than duplicating.
        cur.execute(
            "DELETE FROM proc.bp_contract_obligation WHERE document_id = %s", (run.document_id,)
        )
        for ob in run.obligations:
            cur.execute(
                """
                INSERT INTO proc.bp_contract_obligation
                    (document_id, contract_id, name, obligation_type, clause_ref,
                     source_quote, grounded)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                RETURNING obligation_id
                """,
                (
                    run.document_id, contract_id, ob.name, ob.type.value,
                    ob.clause_ref, ob.source_quote,
                ),
            )
            obligation_id = cur.fetchone()[0]
            for name in {_key(p) for p in ob.participants}:
                cur.execute(
                    """
                    INSERT INTO proc.bp_contract_obligation_party
                        (obligation_id, entity_name, entity_type)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (obligation_id, entity_name) DO NOTHING
                    """,
                    (obligation_id, name, types.get(name, "party")),
                )
        _record_run(cur, run.document_id, run.status, len(run.obligations), len(run.dropped))
        conn.commit()

    return len(run.obligations)


def _record_run(cur, document_id: str, status: str, n_grounded: int, n_dropped: int,
                error: str | None = None) -> None:
    cur.execute(
        """
        INSERT INTO proc.bp_contract_obligation_run
            (document_id, status, n_grounded, n_dropped, error)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            status = EXCLUDED.status, n_grounded = EXCLUDED.n_grounded,
            n_dropped = EXCLUDED.n_dropped, error = EXCLUDED.error,
            created_date = now()
        """,
        (document_id, status, n_grounded, n_dropped, error),
    )


def run_for_document(document_id: str, file_path: str, agent_nick: Any) -> None:
    """Pipeline entry point: read a contract file into grounded obligations.

    Called from the watcher after a contract extracts. Any failure is recorded as
    ``failed`` — never as a contract with zero obligations, which is what an unrecorded
    failure would look like to every reader downstream.
    """
    try:
        from src.services.extraction.dispatch import parse_document

        # Re-parse rather than reuse dispatch's `_source_text`: that is truncated to 6000
        # chars for the training collector, and a quote past that cap would fail grounding.
        text = parse_document(file_path).full_text
        embedding_model = getattr(agent_nick, "embedding_model", None)

        run = extract_obligations(document_id, text, embedding_model)
        persist_obligations(run)
    except Exception as exc:  # noqa: BLE001
        log.exception("%s: obligation extraction FAILED", document_id)
        with get_conn() as conn:
            cur = conn.cursor()
            _record_run(cur, document_id, STATUS_FAILED, 0, 0, str(exc)[:500])
            conn.commit()
