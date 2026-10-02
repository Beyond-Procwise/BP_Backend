"""Propose which contract a contract document sits under. Never link it.

The failure this module exists not to repeat: contract_succession.py has a full
scoring function and unit tests, and pass_runner.run_all() has never called it.
A profile nothing calls is indistinguishable from a profile that found nothing.

NOTHING RUNS THIS MODULE. READ THIS BEFORE ASSUMING OTHERWISE.
    * No scheduler, API route or watcher invokes propose_parent_links(). It is
      called explicitly: by tests and by the deployment verification. A passing
      test_the_runner_is_actually_called proves the function works when called,
      NOT that anything calls it.
    * Calling it automatically (a scheduler job or an API route) is an OPEN
      DECISION, not an oversight: it would add rows to a queue a person works, and
      whether buyers want proposals appearing unprompted has not been asked.
    * confirm() is likewise uncalled. It expects a UI or API caller.
deal_link_proposals.propose has the same shape (a writer with no caller).

WHY A PROPOSAL AND NOT A LINK. proc.bp_contract_master.parent_contract_id is
populated on 1,561 bp_contract_master rows (the table that supplies
candidate PARENTS, not the children this module scores) and resolves to a real contract on ZERO of them.
Writing a parent on a score would be writing the same kind of value that is
already wrong 1,561 times over on the parent side. A person confirms, through the queue they
already work.

WHERE IT LANDS. proc.bp_extraction_discrepancy, the Action Centre findings
surface -- the same place deal_link_proposals.py puts its parent proposals. A new
table would be a second queue nobody opens.

THE KEY. The open-row key is
(doc_type, doc_pk_candidate, coalesce(source_file,''), issue_type, field_name),
enforced by the partial unique index ix_bp_extraction_discrepancy_open_key.
Three things follow, and all three have bitten this product:
  * source_file goes through normalise_source_file on write AND in the
    idempotency SELECT. One spelling written and another read re-proposes on
    every scheduler tick.
  * doc_pk_candidate is a contract_id read OUT of a document, so two documents
    can carry the same one. source_file separates them. That collision cost 65
    days of findings on bp_sqldb.
  * ONE open proposal per document. Two candidates for one child produce the
    same five key columns and the index rejects the second, so the best
    candidate goes in expected_value and the rest in computed_value.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.services.concepts.contract_type_map import structure_for_contract_type
from src.services.concepts.vocabulary import ensure_vocabulary
from src.services.db import get_conn
from src.services.extraction.persistence import normalise_source_file
from src.services.graph_resolution.profiles import contract_hierarchy as _ch

log = logging.getLogger(__name__)

ISSUE_TYPE = "contract_parent_proposed"
FIELD_NAME = "parent_contract_id"

#: Below this the evidence is too thin to be worth a person's attention. The
#: linking engine's own review band -- not a number invented here.
MIN_SCORE = 65.0

#: How far apart the best and second-best must be for the proposal to read as
#: "confirm this" rather than "choose between these".
SEPARATION = 8.0

# "Unparented" means the pointer does not RESOLVE, not that it is absent.
#
# This filter used to be `AND parent_contract_id IS NULL`, meaning "skip contracts
# that already have a parent". A dangling pointer is not a parent, so the SEMANTICS
# were wrong: a contract whose pointer resolves to nothing is as parentless as one
# with no pointer, and it is exactly the case contract_hierarchy._cmp_reference was
# changed to rescue (a dangling reference reads MISSING, 75.59, not CONFLICT, 45.00).
#
# SCALE, honestly. These figures are PARENT-SIDE counts from bp_contract_master, the
# table that supplies candidate PARENTS, not the table children come from:
#     parent_contract_id IS NULL ........ 1,490
#     pointer set and RESOLVES .......... 0
#     pointer set but DANGLING .......... 1,561
# Children come from proc.bp_contracts, which has 0 rows. Of the 3,051
# bp_contract_master rows only 7 have a structure that could ever be a child
# (4 Invoice, 2 Amendment, 1 Purchase Order) and none carries a parent pointer; the
# 1,561 dangling pointers sit entirely on parent-type rows. So the correction is
# right, and its impact TODAY is ZERO: no uploaded contract documents exist yet.
#
# The test happens in Python (_is_unparented), not SQL, so the child filter and the
# scorer's reference_resolves use ONE normalisation (_ch._norm_ref) on BOTH sides:
# a pointer differing only in case, spacing or punctuation resolves for both.
# Direction chosen: normalise both sides, i.e. 'msa 4417' resolves to 'MSA-4417'.
_CHILD_SQL = """
    SELECT c.contract_id, c.contract_title, c.supplier_id, c.resolved_doc_type,
           c.resolved_role, c.framework_ref, c.parent_agreement_ref, c.parent_contract_id,
           c.contract_start_date, c.contract_end_date, c.total_contract_value, c.currency
      FROM proc.bp_contracts c
     WHERE c.resolved_doc_type IS NOT NULL
"""


def _is_unparented(child: dict, known: set[str]) -> bool:
    """No pointer, or a pointer that resolves to no contract other than itself."""
    ptr = _ch._norm_ref(child.get("parent_contract_id"))
    if ptr in _ch._PLACEHOLDERS:
        return True
    return ptr not in known or ptr == _ch._norm_ref(child.get("contract_id"))


def _is_variation(child: dict) -> bool:
    """A document that changes another one: variation, addendum, CCN."""
    dt = ensure_vocabulary().document_types.get(child.get("resolved_doc_type"))
    return (dt.role if dt else child.get("resolved_role")) == "role.variation"


def _wanted_parent_types(child: dict) -> set[str]:
    """The structures this child may sit under. Two paths, on purpose.

    A SOW's parent is identifiable BY TYPE: the vocabulary says it sits under a
    master agreement, so only master agreements are considered. A variation's
    parent is NOT identifiable by type -- it can amend any contract, which is why
    default_parent_type is NULL for it and correctly so. Its parent is
    identifiable only by the reference it carries. So the structure signal steps
    aside (it reads MISSING, which scores 0.5 and penalises nothing) and the
    reference, supplier, term and title signals do the work. The candidate set is
    every contract-family structure, except other variations: a variation
    amends a contract, not another variation.
    """
    if _is_variation(child):
        return {code for code, dt in ensure_vocabulary().document_types.items()
                if dt.pipeline_doc_type == "contract" and dt.role != "role.variation"}
    want = _ch.expected_parent_type(child.get("resolved_doc_type"))
    return {want} if want else set()


def is_child(child: dict) -> bool:
    """Does this document sit under something at all?"""
    return bool(_wanted_parent_types(child))


def candidate_parents(cur, child: dict) -> list[dict]:
    """Contracts that could be this child's parent.

    Two sources, because one alone would be empty for a long time:
      * proc.bp_contracts -- uploaded documents with a recognised structure;
      * proc.bp_contract_master -- the 3,051-row corpus, whose free-text
        contract_type reads as a structure through contract_type_map (3,016 of
        them do).

    Narrowed by supplier in SQL rather than in Python: without it this is 3,051
    score_link calls per child, and the deal-assignment service has already
    taught this product what an unnarrowed per-document query costs.
    """
    wanted = _wanted_parent_types(child)
    if not wanted:
        return []
    supplier = child.get("supplier_id")
    if not supplier:
        return []

    out: list[dict] = []
    cur.execute(
        """SELECT contract_id, contract_title, supplier_id, resolved_doc_type,
                  contract_start_date, contract_end_date
             FROM proc.bp_contracts
            WHERE supplier_id = %s AND resolved_doc_type = ANY(%s)
              AND contract_id <> %s""",
        (supplier, sorted(wanted), child.get("contract_id")),
    )
    for r in cur.fetchall():
        out.append(dict(zip(
            ("contract_id", "contract_title", "supplier_id", "resolved_doc_type",
             "contract_start_date", "contract_end_date"), r)))

    cur.execute(
        """SELECT contract_id, contract_title, supplier_id, contract_type,
                  contract_start_date, contract_end_date
             FROM proc.bp_contract_master
            WHERE supplier_id = %s AND contract_id <> %s""",
        (supplier, child.get("contract_id")),
    )
    for r in cur.fetchall():
        row = dict(zip(
            ("contract_id", "contract_title", "supplier_id", "contract_type",
             "contract_start_date", "contract_end_date"), r))
        # The corpus's free-text type, read as a structure. Not written back:
        # contract_type is source data.
        row["resolved_doc_type"] = structure_for_contract_type(row.pop("contract_type"))
        if row["resolved_doc_type"] in wanted:
            out.append(row)
    return out


def _known_contract_ids(cur) -> set[str]:
    """Every contract_id that really exists, normalised the way references are.

    Both tables, because a reference may name an uploaded contract or a corpus
    one. Read once per pass: the alternative is a scan per child.
    """
    ids: set[str] = set()
    for table in ("proc.bp_contracts", "proc.bp_contract_master"):
        cur.execute(f"SELECT contract_id FROM {table} WHERE contract_id IS NOT NULL")
        ids.update(_ch._norm_ref(r[0]) for r in cur.fetchall())
    ids.discard("")
    return ids


def _claimed_references(child: dict) -> list[str]:
    """The child's raw, non-placeholder reference values, in field order."""
    out = []
    for field in _ch._REFERENCE_FIELDS:
        raw = child.get(field)
        if _ch._norm_ref(raw) not in _ch._PLACEHOLDERS:
            out.append(str(raw).strip())
    return out


def reference_resolves(child: dict, known: set[str]) -> bool:
    """True when at least one claimed reference names a contract that exists.

    This is the flag contract_hierarchy._cmp_reference reads as
    ``_ref_resolves``. True makes a non-matching reference a genuine CONFLICT;
    False makes it a dangling pointer (MISSING). Left unset, the CONFLICT path is
    dead. A child's own id is not evidence of a parent and is not counted.
    """
    own = _ch._norm_ref(child.get("contract_id"))
    return any(
        _ch._norm_ref(ref) in known and _ch._norm_ref(ref) != own
        for ref in _claimed_references(child)
    )


def _source_file_for(cur, contract_id: str) -> str:
    """The document this contract came from, normalised.

    Falls back to 'contract:<id>' for a corpus row that never arrived as an
    upload -- a stable key, and never a basename.
    """
    cur.execute(
        "SELECT source_file FROM proc.bp_contract_raw WHERE contract_id = %s "
        "ORDER BY raw_id DESC LIMIT 1",
        (contract_id,),
    )
    row = cur.fetchone()
    return normalise_source_file(row[0]) if row and row[0] else f"contract:{contract_id}"


def propose_parent_links(limit: Optional[int] = None) -> dict:
    """Score every parentless contract document and propose its best parent.

    The ``considered`` counts are not decoration. 'proposed: 0' reads as "every
    contract has a parent" when the truth may be "no contract resembled a parent
    its supplier holds", and those are different problems with different fixes.
    """
    proposed = contested = no_candidate = 0
    considered = {"children": 0, "with_structure": 0, "with_candidates": 0}
    details: list[dict] = []

    # One connection for the pass, exactly as deal_link_proposals.propose does.
    # get_conn() is AUTOCOMMIT, so each INSERT lands on execute and there is no
    # transaction to commit or roll back.
    with get_conn() as conn:
        cur = conn.cursor()
        known_ids = _known_contract_ids(cur)
        cur.execute(_CHILD_SQL)
        cols = [d[0] for d in cur.description]
        children = [d for d in (dict(zip(cols, r)) for r in cur.fetchall())
                    if _is_unparented(d, known_ids)]
        if limit:
            children = children[:limit]

        for child in children:
            considered["children"] += 1
            if not is_child(child):
                continue              # sits under nothing: not a child at all
            considered["with_structure"] += 1

            candidates = candidate_parents(cur, child)
            if not candidates:
                no_candidate += 1
                continue
            considered["with_candidates"] += 1

            # Resolve the child's claimed references against real contracts ONCE
            # and hand the answer to the scorer. Without it contract_hierarchy
            # reads every non-match as dangling and never reports a CONFLICT.
            scoring_child = dict(child)
            scoring_child["_ref_resolves"] = reference_resolves(child, known_ids)
            scored = sorted(
                ((_ch.score(scoring_child, parent), parent)
                 for parent in candidates),
                key=lambda pair: -pair[0]["F"],
            )
            best, best_parent = scored[0]
            if best["F"] < MIN_SCORE:
                no_candidate += 1
                continue

            runner_up = scored[1][0]["F"] if len(scored) > 1 else None
            separated = runner_up is None or (best["F"] - runner_up) >= SEPARATION
            routing = "suggested" if separated else "contested"
            alternatives = [p["contract_id"] for _s, p in scored[1:4]]

            source_file = _source_file_for(cur, child["contract_id"])

            # Idempotent: the same five key columns the unique index enforces, so
            # the SELECT and the index agree. A scheduler tick must not stack.
            # The five key columns AND the partial clause, copied from the index
            # verbatim. The index is
            #   UNIQUE (doc_type, coalesce(doc_pk_candidate,''),
            #           coalesce(source_file,''), issue_type, coalesce(field_name,''))
            #   WHERE coalesce(status,'open') <> 'resolved'
            # so ANY non-resolved row occupies the slot -- 'dismiss' and 'ignored'
            # included. Testing `status = 'open'` instead would miss a dismissed
            # row and the INSERT below would then hit a unique violation.
            # bp_testdb has only open/resolved today, but bp_sqldb carries
            # dismissed and ignored rows, and that is a deployment target.
            cur.execute(
                """SELECT 1 FROM proc.bp_extraction_discrepancy
                    WHERE doc_type = 'contract'
                      AND coalesce(doc_pk_candidate,'') = %s
                      AND coalesce(source_file,'') = %s
                      AND issue_type = %s
                      AND coalesce(field_name,'') = %s
                      AND coalesce(status,'open') <> 'resolved' LIMIT 1""",
                (child["contract_id"], source_file, ISSUE_TYPE, FIELD_NAME),
            )
            if cur.fetchone():
                continue

            why = {d["id"]: d["status"] for d in best["signals"]}
            # The scored output reads MISSING for a dangling reference and for an
            # absent one alike. We still hold the raw field, so say which it was.
            claimed = _claimed_references(child)
            dangling_note = ""
            if claimed and not scoring_child["_ref_resolves"]:
                dangling_note = (
                    f"It names {', '.join(claimed)}, which no contract matches, so "
                    f"that reference was ignored rather than counted against it. ")
            cur.execute(
                """INSERT INTO proc.bp_extraction_discrepancy
                       (doc_type, source_file, doc_pk_candidate, field_name,
                        issue_type, severity, raw_value, expected_value,
                        computed_value, blocks_promotion, notes)
                   VALUES ('contract', %s, %s, %s, %s, 'info', NULL, %s, %s,
                           false, %s)""",
                (
                    source_file, child["contract_id"], FIELD_NAME, ISSUE_TYPE,
                    best_parent["contract_id"],
                    ", ".join(alternatives) or None,
                    (
                        f"this {child['resolved_doc_type'].split('.')[-1]} appears to "
                        f"sit under contract {best_parent['contract_id']} "
                        f"(score {best['F']:.1f}, {routing}). "
                        f"reference: {why.get('declared_reference')}; "
                        f"structure: {why.get('expected_structure')}; "
                        f"supplier: {why.get('supplier')}; "
                        f"term: {why.get('term_containment')}; "
                        f"title: {why.get('title_overlap')}. "
                        + dangling_note
                        + (f"Other candidates: {', '.join(alternatives)}. "
                           if alternatives else "")
                        + f"Nothing has been linked. Confirm to set "
                          f"parent_contract_id = {best_parent['contract_id']}."
                    ),
                ),
            )
            proposed += 1 if routing == "suggested" else 0
            contested += 1 if routing == "contested" else 0
            details.append({"contract_id": child["contract_id"],
                            "parent": best_parent["contract_id"],
                            "F": best["F"], "routing": routing})

    result = {"proposed": proposed, "contested": contested,
              "no_candidate": no_candidate, "considered": considered,
              "details": details}
    log.info("contract parent proposals: %s",
             {k: v for k, v in result.items() if k != "details"})
    return result


def confirm(contract_id: str, parent_contract_id: str, source_file: str,
            reviewer: Optional[str] = None) -> bool:
    """A person accepted the proposal: set the parent and close the finding.

    Links ONLY a parent that was actually proposed: there must be an OPEN proposal
    for this (contract_id, source_file) whose expected_value is parent_contract_id.
    Otherwise nothing is written and False is returned. Without the check this
    would set any parent it was handed, and a dismissed proposal (status ignored)
    could still be turned into a link.
    """
    key = (contract_id, normalise_source_file(source_file) or "", ISSUE_TYPE, FIELD_NAME)
    with get_conn() as conn:
        cur = conn.cursor()
        # 1. Cheap existence check, so the common failure is caught before anything
        #    mutates.
        cur.execute("SELECT 1 FROM proc.bp_contracts WHERE contract_id = %s", (contract_id,))
        if not cur.fetchone():
            return False
        # 2. CLAIM the proposal in ONE conditional statement, before linking.
        #    get_conn() is AUTOCOMMIT: there is no transaction to wrap this in, and a
        #    FOR UPDATE lock would end with its own statement. A single conditional
        #    UPDATE is the lock that works here. rowcount 0 means a dismissal or
        #    another confirm got there first, so nothing is written. The ordering
        #    decides the failure direction: if anything goes wrong after the claim
        #    the proposal is closed WITHOUT a link (a person sees it disappear), never
        #    a link with no valid proposal (silent and wrong).
        #    `status = 'open'` is deliberate and NOT the index's clause: a dismissed
        #    proposal must not be resurrected, and bp_lifecycle_guard blocks
        #    ignored -> resolved outright.
        cur.execute(
            """UPDATE proc.bp_extraction_discrepancy
                  SET status = 'resolved', resolved_by = %s, resolved_at = now()
                WHERE doc_type = 'contract' AND doc_pk_candidate = %s
                  AND coalesce(source_file,'') = %s AND issue_type = %s
                  AND field_name = %s AND expected_value = %s AND status = 'open'""",
            (reviewer or "contract-parent-confirm", *key, parent_contract_id),
        )
        if cur.rowcount == 0:
            return False
        # 3. Only now link.
        cur.execute(
            "UPDATE proc.bp_contracts SET parent_contract_id = %s WHERE contract_id = %s",
            (parent_contract_id, contract_id),
        )
        if cur.rowcount == 0:
            log.error("confirm(%s -> %s): the contract vanished after its proposal was "
                      "claimed; the proposal is closed WITHOUT a link", contract_id,
                      parent_contract_id)
            return False
    return True


__all__ = ["candidate_parents", "is_child", "reference_resolves", "propose_parent_links", "confirm",
           "ISSUE_TYPE", "FIELD_NAME", "MIN_SCORE", "SEPARATION"]
