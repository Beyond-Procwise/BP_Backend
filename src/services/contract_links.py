"""Propose which contract a contract document sits under. Never link it.

The failure this module exists not to repeat: contract_succession.py has a full
scoring function and unit tests, and pass_runner.run_all() has never called it.
A profile nothing calls is indistinguishable from a profile that found nothing.

WHAT RUNS THIS MODULE, as of 2026-10-04 (it ran NOTHING for two days, and the
docstring said so; Nick settled the cadence question that kept it unwired).
    * extraction.promotion.promote() -- the single funnel every promotion path
      goes through -- calls propose_contract_parent() on a promoted contract,
      SCOPED to that one document (contract_id=). So a contract gets its parent
      proposed within seconds of being promoted, which is the whole point: the
      maths starts making the connections without anyone asking it to.
    * backend_scheduler's `contract-parent-links` job runs the UNSCOPED pass as a
      backstop, for documents promoted while the flag was off, re-read since, or
      arrived by a path that did not promote.
    * Both are governed: autonomous_operation.contract_parent_proposals_enabled.
      An unreadable policy proposes NOTHING -- this writes to a queue a person
      works, so it fails closed, unlike a read path.
    * confirm() is called by DecisionEngine.execute when a person accepts the
      proposal in the Action Centre. It is the only code allowed to write
      parent_contract_id.
deal_link_proposals.propose still has the old shape (a writer with no caller).

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
from src.services.governed_limits import limit as _governed_limit
from src.services.graph_resolution.profiles import contract_amendment as _am
from src.services.graph_resolution.profiles import contract_attachment as _at
from src.services.graph_resolution.profiles import contract_hierarchy as _ch

log = logging.getLogger(__name__)

ISSUE_TYPE = "contract_parent_proposed"
FIELD_NAME = "parent_contract_id"

#: A re-read of the same document REFRESHES its open proposal rather than
#: leaving the first read's answer in the queue. This is the pattern the table
#: already has at promotion._DISCREPANCY_UPSERT and persistence.write_discrepancies,
#: and it must stay identical to both: the conflict target and the partial clause
#: are matched to ix_bp_extraction_discrepancy_open_key BY SHAPE, so a site that
#: disagrees with the index is rejected outright by Postgres ("there is no unique
#: or exclusion constraint matching the ON CONFLICT specification") rather than
#: warning. The index is
#:   UNIQUE (doc_type, coalesce(doc_pk_candidate,''), coalesce(source_file,''),
#:           issue_type, coalesce(field_name,''))
#:   WHERE coalesce(status,'open') <> 'resolved'
#: so ANY non-resolved row occupies the slot -- 'dismiss' and 'ignored' included.
#: `status` is deliberately NOT in the SET list: refreshing the evidence must not
#: resurrect a proposal a person dismissed.
#: Replaced a SELECT-then-INSERT, which (a) left a superseded parent in the queue
#: for confirm() to validate against, (b) could never close the losing row when
#: two documents share one contract_id, and (c) was a TOCTOU -- two passes both
#: saw nothing and the second INSERT raised UniqueViolation out of the middle of a
#: run, with the rows already written committed (get_conn is AUTOCOMMIT).
_PROPOSAL_UPSERT = """
    INSERT INTO proc.bp_extraction_discrepancy
        (doc_type, source_file, doc_pk_candidate, field_name,
         issue_type, severity, raw_value, expected_value,
         computed_value, blocks_promotion, notes)
    VALUES ('contract', %s, %s, %s, %s, 'info', NULL, %s, %s, false, %s)
    ON CONFLICT (doc_type, coalesce(doc_pk_candidate, ''),
                 coalesce(source_file, ''),
                 issue_type, coalesce(field_name, ''))
    WHERE coalesce(status, 'open') <> 'resolved'
    DO UPDATE SET
      expected_value = EXCLUDED.expected_value,
      computed_value = EXCLUDED.computed_value,
      severity = EXCLUDED.severity,
      notes = EXCLUDED.notes,
      blocks_promotion = EXCLUDED.blocks_promotion
"""

def MIN_SCORE() -> float:
    """Below this F the evidence is too thin for a person's attention.

    promotion_thresholds.contract_parent_min_score (65: the linking engine's own
    review band). Read when used, never at import, and RAISES if absent.
    """
    return _governed_limit("promotion_thresholds", "contract_parent_min_score")


def SEPARATION() -> float:
    """How far apart best and runner-up must be to read 'confirm this'.

    promotion_thresholds.contract_parent_separation (8).
    """
    return _governed_limit("promotion_thresholds", "contract_parent_separation")


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
_CORROBORATING = """c.buyer_org_id, c.currency, c.payment_terms, c.governing_law, c.jurisdiction,
           c.contract_signatory_name, c.buyer_signatory_name, c.cost_centre_id,
           c.business_unit_id, c.spend_category"""
_CHILD_SQL = f"""
    SELECT c.contract_id, c.contract_title, c.supplier_id, c.resolved_doc_type,
           c.resolved_role, c.framework_ref, c.parent_agreement_ref, c.parent_contract_id,
           c.contract_start_date, c.contract_end_date, c.total_contract_value,
           {_CORROBORATING}
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
    return _role_of(child) == "role.variation"


def _role_of(child: dict):
    dt = ensure_vocabulary().document_types.get(child.get("resolved_doc_type"))
    return dt.role if dt else child.get("resolved_role")


#: Short labels the notes have always used for the five base signals; every other
#: signal is printed under its own id.
_NOTE_LABELS = {"declared_reference": "reference", "expected_structure": "structure",
                "supplier": "supplier", "term_containment": "term",
                "title_overlap": "title"}


def _evidence_clause(signals: list[dict]) -> str:
    """Every signal in the pair's profile, in profile order, as ``label: status``."""
    return "; ".join(f"{_NOTE_LABELS.get(d['id'], d['id'])}: {d['status']}"
                     for d in signals) + ". "


def _reference_status(scored: dict):
    return next((d["status"] for d in scored["signals"]
                 if d["id"] == "declared_reference"), None)


def _is_attachment(child: dict) -> bool:
    """A schedule or SLA: cited by an agreement, with no force of its own."""
    return _role_of(child) == "role.attachment"


def _profile_module(child: dict):
    """The scoring profile for this child, chosen by its role in the vocabulary."""
    if _is_variation(child):
        return _am
    if _is_attachment(child):
        return _at
    return _ch


def _link_type(child: dict) -> str:
    if _is_variation(child):
        return "amends"
    if _is_attachment(child):
        return "attaches_to"
    return "child_of"


def link_type_of(contract_id: str) -> Optional[str]:
    """child_of / amends / attaches_to for a stored contract, or None if unknown."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT resolved_doc_type, resolved_role FROM proc.bp_contracts "
                    "WHERE contract_id = %s", (contract_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    return _link_type({"resolved_doc_type": row[0], "resolved_role": row[1]})


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
    if _is_attachment(child):
        # A schedule sits under SEVERAL kinds of agreement, and default_parent_type
        # holds one value (also read by the upload gate), so the set is chosen here.
        return {code for code, dt in ensure_vocabulary().document_types.items()
                if dt.pipeline_doc_type == "contract"
                and dt.role in ("role.master", "role.framework")}
    want = _ch.expected_parent_type(child.get("resolved_doc_type"))
    return {want} if want else set()


def is_child(child: dict) -> bool:
    """Does this document sit under something at all?"""
    return bool(_wanted_parent_types(child))


_COMMON_COLS = ("contract_id", "contract_title", "supplier_id", "contract_start_date",
                "contract_end_date", "buyer_org_id", "currency", "total_contract_value",
                "payment_terms", "governing_law", "jurisdiction", "contract_signatory_name",
                "cost_centre_id", "business_unit_id", "spend_category")


def _fetch(cur, table: str, where: str, params: tuple) -> list[dict]:
    """Candidate rows from either table, with the same keys.

    bp_contract_master holds the free-text contract_type (read as a structure, never
    written back) and has no buyer_signatory_name; bp_contracts holds the resolved
    structure and both. The result has resolved_doc_type and buyer_signatory_name
    either way, so a profile never has to know which table a candidate came from.
    """
    master = table.endswith("bp_contract_master")
    cols = list(_COMMON_COLS) + (["contract_type"] if master
                                 else ["resolved_doc_type", "buyer_signatory_name"])
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE {where}", params)
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if master:
        for row in rows:
            row["resolved_doc_type"] = structure_for_contract_type(row.pop("contract_type"))
            row["buyer_signatory_name"] = None
    return rows


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

    THIRD source, and it does not depend on the supplier: the contracts this
    document NAMES. Measured on a real order form 2026-10-03 -- it printed
    "Framework Agreement No. FA-2026-0042", its framework was sitting in
    proc.bp_contracts under exactly that id, and it got no proposal at all,
    because the supplier extractor had read the words "Framework Agreement No"
    as the supplier name and the supplier gate returned [] before anything
    looked. The supplier narrows a SEARCH; it is not a precondition for one, and
    a reference that resolves to a real contract is the strongest signal this
    layer has. An id lookup is bounded by definition, so nothing is paid for it.

    A named contract still has to be a structure this child may sit under: a
    reference is evidence, not an override, and a SOW does not sit under an
    order form however explicitly it names one.
    """
    wanted = _wanted_parent_types(child)
    if not wanted:
        return []
    supplier = child.get("supplier_id")

    out: list[dict] = []
    # One contract, one candidate. A registered contract whose PDF is later
    # uploaded sits in BOTH tables under the same contract_id, and counted twice
    # it ties with itself: best - runner_up = 0, so routing reads 'contested' and
    # the proposed parent appears in its own alternatives list. The document row
    # is kept over the register row -- it carries resolved_doc_type read from the
    # page rather than mapped from free text. Keyed on the same normalisation
    # references use, so 'c02397' and 'C02397' are one contract here too.
    seen: set[str] = set()

    # 1. The contracts this document names. No supplier needed.
    own = _ch._norm_ref(child.get("contract_id"))
    norm = "upper(regexp_replace(contract_id, '[^A-Za-z0-9]', '', 'g'))"
    for ref in _claimed_references(child):
        if _ch._norm_ref(ref) == own:
            continue                     # itself; _drop_self_parent_reference's case
        where = f"{norm} = upper(regexp_replace(%s, '[^A-Za-z0-9]', '', 'g'))"
        rows = (_fetch(cur, "proc.bp_contracts", where, (ref,))
                or _fetch(cur, "proc.bp_contract_master", where, (ref,)))
        for row in rows:
            key = _ch._norm_ref(row["contract_id"])
            if key in seen or key == own:
                continue
            if row["resolved_doc_type"] not in wanted:
                continue                 # evidence, not an override
            seen.add(key)
            out.append(row)

    # 2. Everything this supplier holds that the child could sit under.
    if not supplier:
        return out
    for row in _fetch(cur, "proc.bp_contracts",
                      "supplier_id = %s AND resolved_doc_type = ANY(%s) AND contract_id <> %s",
                      (supplier, sorted(wanted), child.get("contract_id"))):
        key = _ch._norm_ref(row["contract_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    for row in _fetch(cur, "proc.bp_contract_master",
                      "supplier_id = %s AND contract_id <> %s",
                      (supplier, child.get("contract_id"))):
        key = _ch._norm_ref(row["contract_id"])
        if key in seen:
            continue
        if row["resolved_doc_type"] in wanted:
            seen.add(key)
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


def propose_parent_links(limit: Optional[int] = None,
                         contract_id: Optional[str] = None) -> dict:
    """Score every parentless contract document and propose its best parent.

    The ``considered`` counts are not decoration. 'proposed: 0' reads as "every
    contract has a parent" when the truth may be "no contract resembled a parent
    its supplier holds", and those are different problems with different fixes.

    ``contract_id`` narrows the pass to ONE child, which is how the promotion
    hook calls it: a contract was just promoted, so score that document and
    leave every other contract alone. It narrows the CHILDREN only -- the
    candidate parents are still the whole corpus, because a child's parent is
    almost never the document beside it. Without this the hook would re-score
    the entire parentless corpus on every single upload, which is both wasteful
    and (worse) would refresh other documents' proposals under a person who was
    reading them.

    A `contract_id` that names nothing returns the honest empty answer --
    considered.children == 0 -- rather than falling back to the full pass. A
    scoped run that silently became a corpus run is the kind of fallback that
    makes a flag useless.
    """
    proposed = contested = no_candidate = below_threshold = 0
    considered = {"children": 0, "with_structure": 0, "with_candidates": 0}
    details: list[dict] = []

    # One connection for the pass, exactly as deal_link_proposals.propose does.
    # get_conn() is AUTOCOMMIT, so each INSERT lands on execute and there is no
    # transaction to commit or roll back.
    with get_conn() as conn:
        cur = conn.cursor()
        known_ids = _known_contract_ids(cur)
        if contract_id:
            cur.execute(_CHILD_SQL + " AND c.contract_id = %s", (contract_id,))
        else:
            cur.execute(_CHILD_SQL)
        cols = [d[0] for d in cur.description]
        children = [d for d in (dict(zip(cols, r)) for r in cur.fetchall())
                    if _is_unparented(d, known_ids)]
        if limit:
            children = children[:limit]

        # Read once per pass so one pass can never mix two values.
        min_score, separation = MIN_SCORE(), SEPARATION()
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
            module = _profile_module(child)
            link_type = _link_type(child)
            verb = {"child_of": "sit under", "amends": "amend",
                    "attaches_to": "attach to"}[link_type]
            scored = sorted(
                ((module.score(scoring_child, parent), parent)
                 for parent in candidates),
                key=lambda pair: -pair[0]["F"],
            )
            best, best_parent = scored[0]
            if best["F"] < min_score:
                # Candidates existed and were scored; none was good enough. Not
                # the same as never finding one, so it is counted apart.
                below_threshold += 1
                continue
            if module is _am and _reference_status(best) != "OK":
                # An amendment is identified by what it amends. Buyer, currency,
                # law and signatory are shared by all of a supplier's contracts
                # and cannot say WHICH one, so without a resolving reference
                # nothing is proposed (spec 4, Revision 1). The score stays as
                # computed; the gate refuses it.
                below_threshold += 1
                continue

            runner_up = scored[1][0]["F"] if len(scored) > 1 else None
            separated = runner_up is None or (best["F"] - runner_up) >= separation
            routing = "suggested" if separated else "contested"
            alternatives = [p["contract_id"] for _s, p in scored[1:4]]

            source_file = _source_file_for(cur, child["contract_id"])

            # The scored output reads MISSING for a dangling reference and for an
            # absent one alike. We still hold the raw field, so say which it was.
            claimed = _claimed_references(child)
            dangling_note = ""
            if claimed and not scoring_child["_ref_resolves"]:
                dangling_note = (
                    f"It names {', '.join(claimed)}, which no contract matches, so "
                    f"that reference was ignored rather than counted against it. ")
            cur.execute(
                _PROPOSAL_UPSERT,
                (
                    source_file, child["contract_id"], FIELD_NAME, ISSUE_TYPE,
                    best_parent["contract_id"],
                    ", ".join(alternatives) or None,
                    (
                        f"this {child['resolved_doc_type'].split('.')[-1]} appears to "
                        f"{verb} contract {best_parent['contract_id']} "
                        f"(score {best['F']:.1f}, band {best['decision']}, {routing}). "
                        + _evidence_clause(best["signals"])
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
                            "F": best["F"], "routing": routing,
                            "link_type": link_type, "profile": module.PROFILE})

    result = {"proposed": proposed, "contested": contested,
              "no_candidate": no_candidate, "below_threshold": below_threshold,
              "considered": considered,
              "details": details}
    log.info("contract parent proposals%s: %s",
             f" for {contract_id}" if contract_id else "",
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
