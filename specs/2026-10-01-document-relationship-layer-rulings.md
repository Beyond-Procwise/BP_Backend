# Document Relationship Layer — the rulings, in full

Plan: `specs/2026-10-01-document-relationship-layer-plan.md`  
Discovery: `specs/2026-10-01-document-relationship-layer-discovery.md`  
Branch: `Development`. Written 2026-10-01, at the end of the final fix wave.

## Why this file exists

The SDD ledger this layer was built from lives under `.superpowers/sdd/`, which is
gitignored. Every ruling below — 67 from the triage list, four from the pre-flight
conflict scan, two more parked items, and the open questions for the product owner —
would have vanished on merge, leaving a codebase full of decisions whose reasoning
no longer existed. This file is the record. It is copied VERBATIM from the ledger,
not paraphrased, so the cost stated against each decision is the cost that was
actually weighed at the time.

Read it when something here looks wrong. Several of these entries are rulings that
were LATER FOUND TO BE WRONG and corrected by a subsequent entry (see Task 1's first
ruling, which this fix wave reversed, and Task 6's rules (A) and (B), both overruled
by their author). They are kept because the sequence is the useful part: it records
which premises were tested against real documents and which were reasoned from.

Conventions: "Ruling" is a decision taken; "parked" and "deferred" are findings
accepted as residuals with their cost stated; "ADJUDICATION" is a contested finding
settled after argument on both sides.

## Pre-flight rulings (plan vs plan conflicts, before any code)

- Ruling A: the task texts are authoritative over the File Structure summary table; `concepts/routing.py`, `test_category_routing.py` and `test_type_resolution_review_items.py` get built. Cost if wrong: three files land that the summary table never listed — cosmetic, the plan's own table needs a line added.
- Ruling B: `2026-10-02_bp_rule_relationship_group.sql` is NOT built in this plan. The Scope-deviation section and the discovery's R-REL drop are the later word, and a `rule_group`/`blocks_promotion` column with no R-REL rule to carry it is a column checking nothing. Cost if wrong: the next (linking) plan creates it instead — one migration moves plans.
- Ruling C: implementers apply migrations to `bp_testdb` only (the `.env` DB). I apply to `bp_sqldb` myself after the task review passes, so no subagent writes DDL to the prod-lineage database. Cost if wrong: nothing — the same additive, idempotent DDL lands, just from the controller.
- Ruling D: no git worktree. The plan's Global Constraints pin the branch to `Development` in this checkout and mandate named-path commits because another session shares the index. Cost if wrong: a dirty shared index needs care on every commit — mitigated by `git commit -o <paths>` and `git status` checks.

## Rulings, parked items and adjudications, by task

### Task 1

- Ruling: the brief's own seed failed the brief's own guard `test_an_alias_never_equals_another_concepts_code` (alias "framework" vs role.framework; "notice" vs role.notice). The guard stands as written and the seed gives way — implementer dropped bare "framework" (keeping "framework agreement"/"framework contract") and renamed "notice" to "general notice". Reason: the guard is the spec's anti-confusion mandate in executable form, and a bare word losing its alias records an honest `unknown` (Principle 4) which a human can fix with one UPDATE, whereas a narrowed guard silently stops catching name clashes forever. Cost if wrong: a document whose label is exactly "framework" or "notice" classifies as unknown and raises a review item instead of matching. Carried to Nick's open-questions list beside the eight role names.
- Ruling: step 7's literal deliberate-break recipe (duplicate alias in seed.py only) does NOT go red, because the seed-vs-table alias test compares against the union of all table aliases. Accepted the implementer's substitute proof — the duplicate planted in the TABLE turned two guards red (2 failed, 13 passed) and was reverted. The gap is real: a duplicate alias inside seed.py is caught by nothing in Task 1. Task 3's `check_aliases_are_unambiguous` runs over a built Vocabulary and is the right owner, so this is carried into the Task 3 dispatch rather than patched here. Cost if wrong: a seed-side duplicate alias could land between now and Task 3 and only surface once it reached the table.
- Ruling: Important #2 ("proposed never resolves" is only documentation here) is NOT a Task 1 gap — Task 2's brief owns it with `test_a_proposed_type_never_resolves` and loads `status='active'` filtered in SQL. Cost if wrong: nothing; if Task 2 ever drops that test the invariant is unguarded, so the Task 2 review must see it land.
- Ruling: Important #1 (no full-column seed↔table comparison) stands against the plan — the brief's test list was incomplete, and the plan's authorship does not excuse a guard that cannot catch the drift the tables exist to prevent. Fix round 1 dispatched, with a mandatory deliberate-break proof on both tables. Cost if wrong: one extra test and a few minutes.
- minor (deferred): `import json` sits inside the loop in test_document_type_rows_equal_the_seed_column_for_column — style only.
- minor (deferred): the migration's INSERTs were hand-generated from seed.py with no committed generator; the new full-column test is what now catches drift.

### Task 2

- Ruling: the implementer's two flagged coverage gaps (no committed test for the version probe, none for the SQL status='active' filter) are real and get closed BEFORE review, not deferred. Both are the load-bearing behaviours of this module, both were verified only by an uncommitted scratch script or by hand, and "keep the brief's test set exact" meant do not weaken what the brief specified — not decline to guard what it omitted. Each must be proven red (probe degraded to count-only; filter relaxed to all statuses). Cost if wrong: two extra live-gated tests.
- Ruling: promoted the reviewer's Minor #1 (`build_vocabulary` outside the try, so a malformed row raises out of a function documented never to raise) to the fix round. Reason: it breaks the same never-blank/never-raise contract as Important #1 and #3, it sits in the same lines being edited, and a caller in the per-document path getting an exception instead of the last good vocabulary is the exact failure this task exists to prevent. Cost if wrong: one extra try boundary and a test.
- minor (deferred): an `invalidate()` arriving during an in-flight load is cleared by that load's publish — a lost invalidation.
- minor (deferred): `alias_index` is a plain dict inside a frozen dataclass, so callers can mutate it (MappingProxyType would close it).
- minor (deferred): a process kill between the live test's UPDATE and its `finally` leaves recorded_at at now() — autocommit, shared cluster.
- minor (deferred): no test covers dropping `count(*)` from the probe SQL (dropping `max(recorded_at)` is covered).
- minor (deferred): `resolve_alias("policy") == ()` will break if a future active type legitimately claims "policy".
- minor (deferred): if _fetch_version() returns None and _version is also None the probe reports "unchanged" until the TTL — pre-existing, made slightly more reachable by storing None on a failed version read.
- minor (deferred): no test pins the claim that a probe-triggered reload reuses the probe's version (correct by reading).

### Task 3

- minor (deferred): the execution_mode dangling-reference branch (`validate.py`'s `check_every_reference_resolves`, its `execution_mode` branch — line numbers moved when the half-promotion check landed) has no planted test, though every other reference kind does — one test would close it.
- minor (deferred): `validate.py`'s `check_aliases_are_unambiguous` docstring (line numbers moved when the half-promotion check landed) says "active document type" but the check relies on build_vocabulary having dropped non-active rows; say so.
- minor (deferred): `validate.py`'s `_PIPELINES` constant is a hard-coded copy of the four pipeline names that can drift from the extraction code (plan-mandated).
- minor (deferred): the workflow's `cache: pip` keys on requirements.txt but the install step does not use it — vestigial, consistent with the sibling workflow.

### Task 4

- Ruling: fix the SEED, not the assertion — add "quotes" to doctype.quote and "contracts" to doctype.contract_unspecified. Neither word is claimed by another type, so no collision is introduced. Do NOT edit the original migration (its seed is ON CONFLICT DO NOTHING, so an edit would never reach a database that already holds the rows); ship a new additive idempotent pair 2026-10-01_concept_vocabulary_aliases.sql + _rollback.sql that appends only where absent. Implementer applies to bp_testdb; I apply to bp_sqldb after review. Cost if wrong: two extra aliases in the vocabulary and one small migration — far cheaper than a legacy upload spelling silently classifying as unknown.
- Ruling: accepted the implementer's deviation on the brief's provenance test. PARENT_TABLE_FOR_DOC_TYPE is keyed `Invoice`/`Purchase_Order`/`Quote`/`Contract` while pipeline_doc_type values are lower-case, so the brief's test as written reported all four pipelines missing — the BRIEF was wrong, not the seed and not the map. Comparing `{k.lower() for k in map}` is the honest fix and the break proof shows it still goes red for a genuinely missing pipeline. Cost if wrong: the guard tolerates a case difference in that one physical map, which has its own capitalisation convention anyway.
- Ruling: carried the legacy-category test's weaker half to TASK 5, not looped here. The test proves each old spelling resolves, but not that it resolves to the SAME pipeline the old map pointed at — and pipeline mapping is Task 5's subject. Task 5's dispatch must assert each CATEGORY_TO_DOC_TYPE key maps to the pipeline that map names. Cost if wrong: a spelling could resolve to a type whose pipeline differs from the legacy target and nothing would catch it until Task 5.
- minor (deferred): seed alias tuple order is now coupled to the migration's append order, because Task 1's full-column drift test compares aliases in order.
- Ruling: promoted the reviewer's Minor #3 (one-directional guard) to a fix round. The plan's intent is explicitly "a type the vocabulary knows and a map does not, OR THE REVERSE", the test's own docstring promises both, and I verified the reverse is implementable today with no allowlist — all four physical maps hold exactly the four pipelines. A docstring claiming a guard it does not have is the precise failure mode this plan has hit three times already. Cost if wrong: one extra assertion and a monkeypatched break proof.

### Task 5

- REGRESSION found outside the diff — process_monitor_watcher.py:627 sets `_da = "unsupported" if "unsupported" in str(exc).lower() else None` and passes it to _mark_failed as doc_action. The old gate's message contained that word; none of the four new messages does, so every refusal now records doc_action=None. doc_action is an operator/UI-visible status column (see [[project_process_monitor_doc_action]]), so this silently changes what a human sees. Ruling: fix by exception TYPE (isinstance on the two new errors), not by re-adding the magic word — wording-based dispatch is what broke, and re-adding the word leaves the next rewording free to break it again. Cost if wrong: the handler gains one isinstance branch.
- minor (deferred): the old-map spelling test duplicates the legacy test's cases; the collision guard is covered only via build_vocabulary, not the live table; no case-variant test for INVOICE documenting the old .lower() fall-through; `declared_concept` is bound and unused until Task 7 (plan-mandated).

### Task 6

- Ruling: fix the inherited scoring. credit() adds weight per OCCURRENCE unbounded, so 40 body mentions of "Agreement" (5.0 title vs 40.0 body) make a real contract resolve CONFIDENTLY WRONG to contract_unspecified with a 35.0 margin — regressing the very example the implementer's fix repaired, for any document longer than a one-line page. And _MIN_SCORE's comment says "one passing mention is not a classification" while a single body hit (1.0) passes both its threshold and its margin: a 753-char covering letter saying "Please see the invoice attached" returns matched/doctype.invoice. Score on BEST evidence per kind (or distinct aliases), not per occurrence; require strictly more than one body hit. Cost if wrong: classification quality shifts on real documents, which is why I mandated re-running the live check over contract-class documents, not just PO/invoice/quote.
- Ruling: add a FIFTH agreement value `"neither"` for declared None + evidence None — the one case where the current answer is flatly false (nothing was declared, yet it reports declared_only). Keep declared_only for the tie cases, where it is literally true, and mandate that TASK 7 reads status + candidates rather than agreement alone so a contradicting tie still reaches the human through the candidate list. Cost if wrong: one more enum value for Task 7 to carry; the alternative (Task 7 deriving the pairing itself) duplicates logic in two places.
- Ruling: the implementer's Concern 1 was OVERSTATED and the reviewer corrected it — structural signals are NOT dead in production. doctype.order's "ship-to address" and doctype.invoice's "bill-to address" (seed.py:195, 203) are phrase-shaped and fold's hyphen→space rule makes them match; the reviewer verified a verbatim structural_signal span at the right offset on a real-shaped PO page. So: no seed edit, rename the mis-named test, and whether structural_signals should carry phrase-form cues rather than prose goes to Nick with the golden set. Cost if wrong: most seeded signals contribute nothing until curated, and alias matching carries classification meanwhile.
- Ruling: whitespace-split headings DEFERRED with a flag for Nick — fold collapses whitespace on the alias side only (deliberately, to preserve offsets), so a PDF heading parsed as "FRAMEWORK\nAGREEMENT" or "TAX  INVOICE" misses entirely. That matters for exactly the contract class this layer exists for, and a re.escape-per-word-joined-by-\s+ pattern would close it while keeping offsets exact. Not in this round: it is beyond the brief and the round is already large. Cost if wrong: a framework agreement whose title wraps across lines classifies on body mentions alone, or unknown.
- minor (deferred): the vocabulary=None default branch (ensure_vocabulary()) is unexercised — Task 7's wiring must cover the DB-backed load path.
- Ruling (round 2) — STOP retuning weights; replace the summed score with a TWO-TIER LEXICOGRAPHIC comparison. Tier 1 = an alias hit on the heading line, which DOMINATES all other evidence and cannot be outbid by any variety or volume term. Tier 2 = everything else, title zone and body treated ALIKE (the 600-char zone is a crude heading proxy and is what produced Breakage 1), ranked by a bounded sub-score of distinct aliases plus damped occurrences. Compare tier first, sub-score only within a tier; equal tier + equal sub-score stays unresolved. Evidence `kind` vocabulary unchanged because Task 7 reads it. Reason: the flat numeric space let every new term outbid the last and decided real documents by exactly 1.0 — tiers are ordinal, so a sub-score change can no longer leap a tier. Cost if wrong: a third scoring round, and the tier boundary becomes the thing to tune instead of the weights.
- Ruling — INVOICE line 1 / QUOTE line 2 resolves to invoice, accepting the implementer's change; the reviewer independently agrees. A type word on the title line IS stronger evidence than one on the next line, and the locked decision bars inventing a winner where evidence is EQUAL, not recognising that it is not equal. Cost if wrong: one class of genuinely ambiguous two-line headings resolves instead of queueing for a human.
- Ruling (round 3) — TIER 1 IS A HEADING-LIKE LINE, NOT A CHARACTER OFFSET. An alias hit on a short line that the alias substantially fills is tier 1 wherever that line appears; every other alias hit is tier 2 with the existing bounded sub-score. This fixes both of round 2's regressions AND keeps every earlier fix: own-line titles after a letterhead become tier 1; "Ref: your quotation of 1 January" is a prose line where the alias is a small fraction, so it stays tier 2 and 20 body mentions of invoice still win (original Breakage 1 stays fixed); FRAMEWORK AGREEMENT on its own line stays tier 1; and the character cliff disappears entirely because no offset decides anything. Two heading-like lines naming different types both sit in tier 1 and tie -> unresolved. Cost if wrong: "heading-like" needs its own thresholds (line length, alias coverage fraction), but they are structural rather than evidence-weight, and no single character can flip them.
- Ruling — restore _MIN_MARGIN = 1.0 and raise _SIGNAL to 1.0, accepting the reviewer's recommendation over the implementer's unprompted 0.5. At 0.5, three occurrences versus two (2.585 vs 2.000) resolves where it used to tie, and three passing mentions against two is noise a human should see. The reviewer verified this exact pair keeps all 35 tests green AND preserves the single-signal tie-break. Cost if wrong: slightly more goes to the review queue, which is the right direction for a layer whose whole point is reporting rather than deciding.
- Ruling — keep `title_chars` and Evidence.kind, but document honestly that the parameter only LABELS evidence and influences no decision (verified identical outcomes across 0/20/600/10000). Cost if wrong: Task 7's caller could still misread it as a tuning knob; the docstring is the mitigation.
- Ruling (round 4) — replace the DEFINITION, stop tuning thresholds. (A) A segment is title-like when, after a bare .strip() plus markup/punctuation/trailing-number stripping, it EQUALS a matched alias — equality, not a fraction; delete _HEADING_COVERAGE and _HEADING_LINE_MAX, and the bare .strip() also kills the surviving \r\t one-character cliff. (B) A table row with 2+ non-empty cells is FIELD DATA, not a title — this is what separates the live `| INVOICE |` title from `| PO # | 4412 |`, `| Contract sum | ... |` and `| Quotation to | Smith Ltd |`, and it retires the 0.023 knife-edge. (C) Among title-like segments the FIRST IN DOCUMENT ORDER is the title; later ones do not compete, and repetition plays NO part in tier 1 — this kills the schedule problem because FRAMEWORK AGREEMENT precedes Schedule 1. Ordinal, not offset-based: no character count decides it. (D) If that first segment names 2+ concepts the result is unresolved with those candidates, which is how a tie stays reachable. Cost if wrong: the equality rule may be too strict for real titles carrying a suffix, which the live re-run will show.
- Ruling — (C) supersedes my earlier acceptance of "INVOICE line 1 / QUOTE line 2 -> invoice" AND the round-3 implementer's tie resolution of it. Under first-title-like-segment-wins, INVOICE wins because it is the document's title, which is principled rather than a positional bonus. Same-segment ties stay asserted.
- Ruling — DELETE rather than test any rule the new definition makes unreachable (the phrase cap, the repeated-phrase rule, possibly the sentence/label rule), proving unreachability by sweep first. A rule that cannot fail is not a guard; it hides which code is load-bearing. The repeated-phrase rule is exactly this: still green when deleted, with a test that cannot reach it and a reported proof that does not reproduce.
- Ruling — ACCEPT rather than fix: a lone `Invoice` footer line on a covering letter will be the first title-like segment and will resolve matched. It produces a review item, not a routing change, and another rule costs more than it saves.
- Ruling — stay on branch Development rather than cutting a worktree. Tasks 1-5 all landed there, and Task 7 step 9 proves on the live local server which runs from this checkout. Cost if wrong: shared-checkout contention with the other session, mitigated by `git commit -o <paths>` only.
- Ruling — STAND DOWN from round 4. The other session is further along (tests rewritten, break proofs running) and lands it. Reason: two implementers on the same two files clobber each other, and the skill's one-implementer-at-a-time rule binds across sessions, not just within mine. Cost if wrong: my implementer's live evidence is wasted unless handed over, so it was handed over by cross-session message to both bp-backend sessions and written into task-6-round4-findings.md. Ownership question sent to both; awaiting an answer before anyone dispatches again.
- Ruling — MY RULE (B) WAS WRONG and is corrected in task-6-round4-findings.md. "A table row with 2+ non-empty cells is field data, never a title" rests on a premise that is false in the live corpus: there is no `| INVOICE |` row, the real shape is `| Ironbridge Managed IT Ltd | | | INVOICE | |`. Implemented literally it loses every workbook title and returns matched/doctype.order on the live invoice MCP-INV-1148 — a new member of the confident-wrong-answer family, i.e. the rule would have caused exactly what it was written to stop. Correct rule: only the LAST non-empty cell of a row may be title-like (live: 10/10 invoices, 5/5 POs). Both my implementer and the other session reached this independently. Cost if wrong: none now — it is strictly better than what it replaces and is measured, not reasoned.
- Ruling (round 5, FINAL) — six items, all categorical: (1) extend (B) to the vertical shape — a segment is a KEY when the next non-empty segment is a bare reference token (digits, or a token containing a digit, naming no type), which is (B)'s own sentence for a pipe-less layout and removes three of the five shapes including the column-header case; (2) OVERRULE MY OWN rule (A) and keep `, ; :` on the tail, on the reviewer's better reasoning that a trailing colon's entire meaning is "the value follows" — 'Order Date:' is literally a parsed segment in a live quote PDF; (3) strip a trailing `#`-prefixed reference token in _normalise, the cheapest real-data win left, which turns QUOTE_WSG100024 and QUOTE_WSG100025 from confidently-wrong matched/doctype.order/disagreed into `agreed`; (4) fix the NEGATIVE slice at :427 that ADDS rows past the cap (a 13-concept title returns 13 rows, a 15-candidate page 27) — (D) is what made >12 candidates reachable where the deleted phrase cap bounded it at 2; (5) fix the zero-evidence case at :324-333, where a title with internal multi-whitespace names a concept with no hit in its own span, so an `unresolved` can show a person NEITHER side; (6) fix the comment at :75-76 that asserts a corpus-wide fact the local workbooks contradict. Cost if wrong: the vertical-key rule could suppress a genuine title that happens to precede a reference line.
- Ruling — ACCEPTED AS RESIDUALS, not fixed: the covering-page/email title preceding the real document; a stray parser-artefact line above the real title; the contents page with no title of its own; the lone `Invoice` footer; `| Document type | Purchase Order |` resolving by accident; titles whose last cell is a date or "Page 1 of 2" losing the title and returning unknown; _REF_TOKEN's roman-numeral branch making "QUOTE MIX" a quote; and 6 of 12 local SpendIQDocs files having no title-like segment at all. None can reach routing.
- ADJUDICATION 1 — PARKED. The new confident-wrong class from item 1(ii)'s plain-line form (type_resolver.py:460-462): a title directly above a bare number, date or page number loses its title and the body's clauses can answer confidently wrong ("INVOICE\nINV-2026-0001\n" + a PO-heavy body returns doctype.order where round 4 returned doctype.invoice). Ruling: PARK, do not fix. Reasons: measured incidence is ZERO across 63 real documents and the 50 process_monitor documents; the reviewer established the defect and its fix are the SAME SHAPE, since in the shape item 1 was built to fix the key is also the first segment, so the only discriminator is whether a later title-like segment exists — and adopting that rule provably returns item 1's third shape (a workbook header row ending `| PO |`) to a confident wrong answer; and nothing under src/ imports this module, so it cannot reach routing. Cost if wrong: a page whose title sits immediately above a bare reference loses its title and may be typed from its body clauses — latent risk, not observed error. A future round wanting to revisit it must carry BOTH shapes as tests, because fixing either one alone reopens the other.
- ADJUDICATION 2 — RULED, keep the floor. The evidence bound stays `max(12, len(candidates))` rather than a hard 12, accepting the implementer's argument and the reviewer's recommendation: a person asked to break an N-way tie must be shown N spans, and truncating evidence below the candidate list produces a review row that cannot justify some of its own options — a grounding defect, where a long row is merely an ergonomic one. I explicitly decline the reviewer's alternative of bounding rule D's candidate list instead: truncating candidates would hide genuine ambiguity, which is worse than showing one long row. Cost if wrong: a 13-way tie renders a long review row. Task 7 must treat `evidence` as variable-length.
- ADJUDICATION 3 — LOAD-BEARING, carried into Task 7 rather than parked. A `matched`/`disagreed` result can return 12 evidence rows that ALL belong to the declared concept and none to the winner, because the per-candidate reservation at :583-587 runs only when status == "unresolved" (pre-existing: round 4 returns the same 12 order-only rows for the same page). Task 7 builds the human review queue on exactly this evidence, so a row showing twelve reasons for the type that did NOT win and none for the one that did is a grounding defect in the surface Task 7 creates. Per the skill's breaker rule for a load-bearing finding I am ruling the smallest change that unblocks the dependent work — extend the existing reservation to guarantee one span for `evidence_concept` on a matched result — and carrying it into Task 7's dispatch, which already edits this same module to add type_resolution_discrepancies. Cost if wrong: one more condition in a function that already does this for ties.
- minor (parked): tests/services/extraction/test_type_resolver.py:930's comment misstates why its assertion holds (token count, not a digit test) — the three-token no-digit form "Quotation # FINAL" IS reduced to a title. Same class of inaccuracy as item 6, in a test rather than the module.
- minor (parked) + correction to my own earlier statement: "nothing unknown or unresolved" is true of the 64 MEASURED documents, not of the directory — /home/muthu/Downloads/SpendIQDocs holds 1,930 parseable files, and po_invoice_linked_data.csv (a top-level file outside the measured 14) returns `unresolved`.

### Task 7

- Ruling (pre-flight) — the brief's Files section says "document the two new `issue_type` values" but the task creates THREE (document_type_disagreement, unresolved_document_type, unknown_document_type). Document all three in persistence.py:61's comment. Reason: the comment is the only inventory of legal issue_type values and an incomplete one invites a fourth undocumented type. Cost if wrong: none; it is a comment.
- Ruling — three of the five concerns are correctness problems in the surface a human works and get fixed BEFORE review: (1) an open disagreement row must not outlive the disagreement — a re-read that now agrees must clear or close the row, following whatever convention the existing 5,372 rows express via resolution_action/resolved_by, because a queue whose rows are mostly already untrue is a queue people stop trusting; (2) pk-less documents sharing one open row per issue type is DATA LOSS, one document's finding silently replacing another's, so the key must be unique per document using the process_monitor_id (always available in dispatch) marked visibly as a process-monitor reference rather than a fake primary key — and NOT solved by declining to write the finding, since a document with no extracted ID is exactly the kind a human must see; (3) a caller with nothing declared has nothing to contradict, so skip the finding entirely when declared_concept is None — in the live path the Task 5 gate raises on anything unresolvable, so None only ever means a script caller, and "the type is unknown" is not news about a document nobody typed.
- Ruling — concern 3 (benchmark_live counting any open row as "flagged", and session summaries counting warnings, both inflated by these non-blocking findings) is investigate-then-decide: if excluding the two issue_type values or filtering on blocks_promotion = false is a few lines in those consumers, do it; if larger or it changes what the metrics mean for other callers, report precisely and I park it. Reason: figures from this product get quoted, and a number that moves because a reporting feature shipped is worse than no number. Cost if wrong: two existing metrics overcount until someone filters them.
- Ruling — concern 5 needs no code. Twelve rows sharing one key (issue type + declared + evidence concept) is the right shape; collapsing them into a single human decision is the UI's job and is recorded for whoever builds that surface. Do not build grouping into the backend.
- Ruling — the two named consumers (benchmark_live, the session warning count) are FILTERED, with tests pinning the filters to the builder's issue types. The five others the implementer surfaced — corpus_facts, summary_agent, value_summary_service, value_query_service, analysis_findings — are PARKED, not filtered. Reason: the two I filtered are pure artefacts of this feature (any open row counting as "flagged"; a raw warning tally), whereas those five are user-facing analytics whose numbers get quoted, and each needs its own product judgement about what its figure is FOR. Editing five unrelated analytics consumers would widen Task 7 far past its stated file list on the live path. The report records the exact one-line filter for each, so the decision is cheap whenever Nick wants it. Cost if wrong: five analytics surfaces count non-blocking type findings alongside real problems until someone applies those five lines.
- Ruling — fix round 2, seven items. (A) the implementer's own live probing left SIX fabricated open findings in the queue (ids 8688-8693, source_file a /tmp scratchpad CSV, all open, all pk-less) while the report claimed zero remain — delete them, prove the counts, and correct the false claim, because a false "0 remain" stops the next person looking. It also proves the pk-less collision is LIVE DATA LOSS rather than hypothetical: zzprobe_d's findings were collapsed by the shared NULL key. (B) the auto-close can close a row a HUMAN is working — the decision engine's escalation actions (flag/hold/escalate/assign/investigate/query) keep status='open' with resolution_action NULL and leave NO row-level trace, so the WHERE cannot tell untouched from escalated; narrow it by joining the decision/transition log, never stop closing. (C) WITHDRAW `unknown_document_type`: type_resolver.py:612-613 forces status="matched" whenever declared_concept is truthy, so unknown requires declared_concept is None, which my own skip ruling now excludes — an issue type that cannot fire is the same thing as the guards this plan has been deleting all day, its honest subject is already reported as declared_only, and an empty extraction already raises missing_required. (D) my parked consumer list was built on WRONG FACTS — value_summary_service and value_query_service both already whitelist DISCREPANCY_VALUE_TYPES so neither can be inflated, while extraction_telemetry/telemetry_service.py:85-96 counts every issue type into n_discrepancies and was missing; the corrected parked list is corpus_facts, summary_agent, analysis_findings, telemetry_service. (E) both new filter guards assert SOURCE TEXT (inspect.getsource, and literals in SQL captured by a fake cursor) so they would pass if the clause were commented out — make one behavioural; that would have been the twelfth checking-nothing guard. (F) type_finding_doc_key is unstable across runs so a pk run and a no-pk run leave two open rows neither closable, which undermines Fix 1. (G) if the builder raises after the resolver succeeded, the stale-clear runs with an empty current set and closes a still-valid finding.
- Ruling — corpus_facts' COUNT(DISTINCT doc_pk_candidate) AS documents_affected will now count `file:...` pseudo-documents as documents. That distortion is a direct consequence of my Fix 2 keying ruling. Parked with the other analytics consumers rather than fixed, but recorded as mine.
- Ruling — fix round 3, five items. (1) F only closes the pk-GAINED direction; the finding's own case is pk-LOST, where the later run has doc_pk=None so the pk never enters other_doc_keys and the pk-keyed row can never be closed — use `source_file`, which is already on the row, and test the pk-lost direction specifically. (2) `declared_concept=""` passes the `is not None` gate but is falsy in the resolver, so it takes the undeclared branch, can return unknown, and lands in the unfiltered _other_items lane that is excluded by neither consumer filter and never auto-closed — the one combination escaping everything we built; gate on `if declared_concept:`. (3) the E fix made CI WEAKER while making live stronger: the source-level benchmark guard was deleted, so without PROCWISE_TEST_LIVE_DB=1 the filter has no guard at all, and that is how most runs happen — keep the behavioural live test AND restore a non-live guard beside it, because the earlier mistake in this plan was having only the weak one, not having two. (4) add `AND e.resolved_by IS NULL` to the closer: the Node gateway's resolveDiscrepancy maps `flag` and every unrecognised verb to status='open' with resolution_action NULL and NO bp_decision row, but it DOES set resolved_by, and bp_testdb has zero open rows with resolved_by set — so the predicate closes a real escalation hole at zero over-skip cost, on our side, with no gateway edit. (5) the B test MIRRORS the engine's writes instead of using them, so nothing pins the subject_type/subject_id convention the anti-join depends on and a rename would silently disable the guard with the test green — the thirteenth version of this plan's recurring failure.
- parked — the fail-open audit insert (decision engine, out of scope); the orphan bp_decision probe row (resolution='probe', invisible to the approvals queue which filters resolution='escalated'); that POST /decisions/finding/{id} writes a row on EVERY call including advice-only ones and so freezes auto-closing for that finding forever (nothing loops today); and the `file:<path>` form being closable across two documents from one workbook (177 process_monitor rows over 49 distinct file_paths), which only exists under an out-of-scope regime.
- minor (parked) — a quote-only over-close collision. The source_file branch is scoped to doc_type, field_name, the type-finding issue types, open status, no query sent, no resolved_by and no bp_decision row, but it does NOT apply the keep-list, so a different key under the same file closes regardless of issue type. Real only for doc_type='quote': bp_quote_raw has 4 source_file values mapping to more than one quote_id (invoices, POs and contracts have none). So reading quote B from a multi-quote workbook can close a type finding filed under quote A's pk. Low risk because the type evidence comes from the file's text and the declared concept, which are shared per file, so A and B almost always raise the same finding; it cannot cross doc_type and cannot touch any row a human, the gateway or the decision engine has acted on. Cost if wrong: an occasional still-valid type finding closed for a sibling quote in the same workbook. Not named in the code or tests — worth a comment if anyone revisits.
- minor (parked) — the `declared_concept` gate has no NON-LIVE test, so a future revert to `is not None` would be caught only by a manual live probe; and test_unknown_document_type_is_unreachable_with_a_declared_concept exercises the empty-string TEXT case but not declared_concept="".

### Task 6 (this entry carries no task label in the ledger; it is round 5 of Task 6)

- Ruling: rule (B) is replaced by "within a table row, only the LAST non-empty cell may be title-like" — a label/value heuristic rather than a cell count. It still excludes `| PO # | 4412 |`, `| Quotation to | Smith Ltd |` and `| Contract sum | 1,250.00 |`, which were the three cases (B) existed for. My `| INVOICE |` premise was simply wrong about the corpus. Cost if wrong: a title row whose last cell is a date or page number would be missed, which the re-review is probing.

### Task 8

- Ruling — ACCEPT the DOC_CLASSIFY field cut from 11 to 4. The seven dropped fields (contract_type, is_amendment, parent_contract_id and the rest) are not on the manifest's bp_contracts or bp_supplier profiles, so declaring them would have filtered NOTHING — the same empty-set mistake this plan has deleted thirteen times. Enriching those profiles to carry them would also change the UNSLICED bundle, which is a wider blast radius than this task owns. The honest subject for those columns arrives with the R-REL/linking work, so the profile-enrichment question is deferred with it. Cost if wrong: an agent reasoning about amendments or parent contracts does not get those columns in its slice and someone must enrich the profile first.
- Ruling — ACCEPT that no caller passes task_id yet, so the manifest slice is DORMANT. This is not the same defect as a guard checking nothing: the mechanism is provably functional (11,293 -> 876 when asked, with a test asserting real numbers), it simply has no caller, and wiring orchestrator steps to task ids is beyond this brief. The live win this task actually delivers today is the negotiation prompt (18,815 -> 52 chars in a prompt that ships to a model). Recorded as the honest state rather than hidden. Cost if wrong: the slicing sits unused until a step is wired to a task id.
- Ruling — fix round 1, three items. (1) concern 4 confirmed: the byte guard is decorative for columns (unfiltered columns measure 549 chars against the sliced bundle's 859, because relationships and workflow dominate), so assert on loaded["rows"] against the real column count — the metric that actually moves when columns are filtered — and rename the byte assertion for what it truly catches. (2) the silent `if profile is None: continue` at agent_manifest.py:129 quietly drops a declared table, so it must surface rather than produce a thinner slice than it declares. (3) correct the report to label the negotiation reduction latent and the slice dormant, and to state what production bloat remains. Cost if wrong: one renamed test and one raised error.
- minor (parked) — `startswith(t)` relationship matching would also match a longer table sharing the prefix (proc.bp_supplier_x); harmless today and narrowing it risks the relationship filtering the reviewer confirmed works.

### Task 9

- minor (parked) — a parent package's __init__ is not walked as an edge, so an __init__ that imports i18n would be missed (no non-i18n module imports i18n today except the router); this limit is unstated and belongs in the CANNOT-CATCH docstring. reader_modules counts the API router as a reader because its DOCSTRING mentions the table, which is conservative and the reviewer recommends leaving. The migration test bans the bare substrings `label` and `locale` without word boundaries, so a future `labelled` or `allocated` would trip it. The migration test strips only `--` comments, so a /* */ comment would be scanned.

### Further parked items recorded only in the ledger

- Task 6: minor (deferred, for Task 7's thresholds): every confident answer in the breakage cases wins by exactly 1.0, so one added seed alias can flip a real document — the tier model is the structural answer.
- Task 7: minors parked — the unbounded developer-format `notes` listing reaching a buyer-facing field (~800 chars on a 508KB page); resolved history rows accumulating outside the partial unique index (correct by the table's design, but summary_agent counts all rows); type_finding_doc_key(None,None,None) returning None, unreachable from dispatch but public and tested; ~1.0s added synchronously per 508KB page, bounded not a failure mode.

## Open for the product owner

Carried out of the plan unanswered. Numbering is the ledger's.

1. GOLDEN SET, now quantified: proc.process_monitor holds 133 quote + 30 invoice + 14 po and ZERO contract-category rows. Exactly one real contract document exists locally (a Marketing Agreement PDF) and it passes only because it happens not to list schedules — the very shape that broke round 1. The contract class (framework / MSA / SOW / call-off / order form / schedule / addendum / CCN / task order / variation / avenant) is the whole subject of this layer and rests on one file. 10-20 real documents would change what can be proven.
2. THE EIGHT ROLE NAMES are seeded as DATA and still unconfirmed: framework, master, transaction, variation, attachment, notice, termination, supporting. Renaming one is an UPDATE, not a rebuild.
3. TWO ALIASES WERE DROPPED to satisfy the plan's own anti-confusion guard: bare "framework" (collides with role.framework) and bare "notice" (collides with role.notice, now seeded as "general notice"). A document labelled exactly "framework" or "notice" classifies as unknown and raises a review item.
4. THE UPLOAD GATE NOW ACCEPTS MORE than it did: "bill", "estimate", "order", "agreement" and every other seeded alias route, where the old four-entry map raised a hard error. Nothing that routed before stops routing. Whether an uploader's free-text category should route on an alias is a product ruling.
5. STRUCTURAL SIGNALS are mostly prose ("lists incorporated documents") and cannot match a page; only "ship-to address" and "bill-to address" are phrase-shaped. Curating matchable cues needs the golden-set documents, so it is deferred rather than invented.
6. WHITESPACE-SPLIT HEADINGS miss entirely: fold collapses whitespace on the alias side only (deliberately, to keep offsets byte-exact), so a PDF heading parsed as "FRAMEWORK\nAGREEMENT" or "TAX  INVOICE" does not match. Deferred with a flag; it matters most for exactly the contract class.
7. SHARED-INDEX INCIDENT: a Task 4 implementer ran a stray `git reset -q`, unstaging the other session's 498 staged deletions. Restored, and our 13 commits touched only our own paths. UNVERIFIABLE: whether any of the 14 modified files were staged beforehand — the other session may need to re-stage them.
8. CI WORKFLOW UNPROVEN IN CI: .github/workflows/reference-data-checks.yml runs its steps correctly locally and exits non-zero on a planted violation, but nothing here can confirm GitHub triggers it or that the runner installs. Its first run on push to Development is the proof.

> Item 3 above is now OBSOLETE: the final fix wave reversed that ruling. Bare
> `framework` and bare `notice` are restored as aliases of
> `doctype.framework_agreement` and `doctype.notice_general`
> (`deploy/sql/2026-10-01_concept_vocabulary_restored_aliases.sql`), because the
> guard that cost them compared aliases against every concept's local name while
> the alias index is built from `proc.bp_document_type` alone — so a `role.*` code
> could never have contested an alias. The consequence is a routing one: an alias
> is also an acceptable upload category, so `framework` now routes at the contract
> pipeline and `notice` reaches `doctype.notice_general`, which has no pipeline and
> so still refuses — with "no pipeline" instead of "no document type claims it".

## What the final fix wave changed about this record

The whole-branch review returned *Ready with conditions*. The one fix wave that
followed it did these things to the entries above, and nothing else:

- **Task 1's first ruling: REVERSED.** See the note under item 3.
- **Adjudication 1: PINNED, still parked.** It is now a positive assertion in
  `tests/services/extraction/test_type_resolver.py`
  (`test_a_title_directly_above_a_bare_reference_loses_its_title_ACCEPTED`), whose
  docstring carries the ruling, so a future "fix" goes red and the person reads why.
- **Task 7 item 61 and Task 6 items 47 and 52: COMMENTED IN THE CODE** at
  `src/services/extraction/persistence.py` (the `source_file` over-close branch) and
  in `src/services/extraction/type_resolver.py` (the accepted title-detection
  residuals, and the scope of the word "measured").
- **Task 7 ruling 57(D): the telemetry consumer is now FILTERED,** not parked. It
  was the only one of the four writing its count into a persisted per-document row.
  `corpus_facts`, `summary_agent` and `analysis_findings` remain parked: they compute
  on read.
- **A new invariant was found and closed.** `status` lived on both
  `proc.bp_concept` and `proc.bp_document_type` for the same type, so promoting one
  half produced a document type that resolved aliases and routed live uploads while
  its concept was absent — with every validation check passing. `build_vocabulary`
  now drops such a type and `validate.check_concepts_exist_for_every_document_type`
  names it.


## A coupling nobody should relax without reading this

`src/services/extraction/type_resolver.py:616-617`:

```python
        if status != "unresolved":
            status = "matched"
```

That one line, inside the branch taken when a concept **was** declared, forces a
declared-but-silent page to `status="matched"` rather than `"unknown"`. It is
load-bearing in three separate places, none of which is obvious from the line:

1. **It is why `status == "matched"` can coexist with `evidence_concept is None`.**
   A reader — or a future dashboard — will read "matched" as "the page confirmed the
   type". It means "a human declared one and the page said nothing". `agreement`
   carries the truth (`declared_only`); `status` does not. Renaming it was considered
   and deliberately NOT done in the final fix wave, because Task 7 consumes this
   vocabulary and the change is larger than that wave could safely carry.
2. **It is why `unknown_document_type` was withdrawn.** `status == "unknown"` requires
   `declared_concept` to be falsy, and the live gate (`routing.pipeline_for_category`)
   raises rather than returning an empty concept, so the pipeline can never produce
   that status. The issue type was removed from the advertised set on that basis
   (`src/services/extraction/persistence.py`, the `TYPE_FINDING_ISSUE_TYPES` comment).
   Relax this line and the withdrawn type becomes reachable again while still being
   withdrawn — the worst of both.
3. **A Task 7 test passes only because of it.**
   `test_declared_only_produces_no_review_item` expects NO review item for a declared
   page whose text says nothing. It passes because this line prevents Task 7's
   `status == "unknown"` branch from firing. Relax the line and silent declared pages
   start raising review items for every upload whose text the parser could not read.

So: if you are here because `status="matched"` looked wrong, you are right that it
reads wrong — but change it deliberately, with those three consequences in hand, and
expect to touch the withdrawn issue type and that test in the same change.

Credit: the third consequence was found by a concurrent session's pre-flight scan of
Task 7 (`bp-backend-28`), which independently reviewed the same plan for part of the
day before the two controllers discovered each other and it stood down. Its note
called the coupling "a coupling, not a defect", which is the right reading.

## Three things this file was missing (added after the final re-review)

### 1. Three of this layer's test files run in no automated job

`.github/workflows/reference-data-checks.yml` runs nine files. These three do not,
because they need dependencies the job does not install:

| File | Needs | What is unguarded |
|---|---|---|
| `tests/services/concepts/test_gate_wiring.py` | spacy | all 8 tests over the **live upload gate** |
| `tests/services/test_agent_manifest_slices.py` | botocore + qdrant_client | 1 of 15 tests (see below) |
| `tests/services/extraction/test_type_findings_lifecycle.py` | numpy + scipy | 1 of 19 tests (see below) |

**Two of those three are avoidable at zero dependency cost**, verified in a simulated
CI environment by the final re-reviewer: under the job's *existing* install set,
`test_agent_manifest_slices` is 14 passed / 1 failed and `test_type_findings_lifecycle`
is 5 passed / 13 skipped / 1 failed. Deselecting the two heavy tests —

```
  --deselect tests/services/test_agent_manifest_slices.py::test_the_negotiation_prompt_no_longer_carries_the_knowledge_bundle
  --deselect tests/services/extraction/test_type_findings_lifecycle.py::test_session_warning_count_mentions_every_type_finding
```

— adds **19 tests to CI for free**. Only `test_gate_wiring.py` genuinely needs spacy
for all of its tests. Separately: the second deselected test needs numpy only because
it imports `src.services.session_postprocess` to call `inspect.getsource` on it;
reading that module as a text file instead would remove the dependency entirely.

This was not done in the final fix wave because the wave had already been dispatched
and the process allows only one. It is a two-line change to the workflow.

### 2. `bp_sqldb` — what IS done, and the one thing that is not

The final re-reviewer could not connect to `bp_sqldb` (barred by its brief) and
inferred that neither alias migration had been applied there. **That inference was
wrong.** Verified directly:

- `2026-10-01_concept_vocabulary.sql` — applied (47 concepts, 19 document types).
- `2026-10-01_concept_vocabulary_aliases.sql` — applied; `quotes` and `contracts` present.
- `2026-10-01_concept_vocabulary_restored_aliases.sql` — applied; `framework` and
  `notice` present. Alias md5 parity with `bp_testdb` confirmed after each.
- **Half-promotion check, which the new loader rule makes consequential — clean.**
  Zero active `bp_document_type` rows whose `bp_concept` row is not active. The only
  row with a non-active concept on *either* database is `doctype.policy_document`,
  whose document-type row is itself `proposed` — the legitimate seeded state. So the
  new rule drops nothing that resolves today, on either database.

**Still not done, and deliberately referred rather than actioned:** `bp_sqldb` has no
`ix_bp_extraction_discrepancy_open_key`. Because `write_discrepancies` has inferred an
`ON CONFLICT` on that key since 2026-07-30, Postgres rejects the inference (42P10), so
**every** discrepancy write on that database already fails — pre-existing, not caused
by this layer. 1,072 open rows there, 151 duplicate groups.
`deploy/sql/2026-07-30_discrepancy_dedup.sql` would fix it by **DELETING 587 rows** and
rewriting the status of others, and it has no rollback. That is destructive and
irreversible on a production-lineage database, so it was not run. Until it is,
document-type findings are a `bp_testdb`-only feature.

### 3. Restoring bare `framework` and `notice` changes classification on prose

The reversal of Task 1's first ruling restored two aliases that are ordinary English
words, unlike the rest of the seeded vocabulary. The final re-reviewer measured the
before/after and the effect is wider than "one evidence span":

- a call-off contract whose prose says "the Framework" three times: `matched
  doctype.call_off_contract` → `unresolved {call_off_contract, framework_agreement}`
- a letter saying "notice" three times: `unknown` → `matched doctype.notice_general`
- a heading-like line reading `Framework` or `Notice`: `unknown` → `matched`

Bounded, and worth stating why it was still the right reversal: the classifier only
reports, routing comes from the uploader's declared category, and `notice_general` has
no pipeline so it cannot route at all. The cost is review-queue noise, not a routing
change. `framework` does now route, at the `contract` pipeline — the same one every
other contract-class type uses, and a category that previously hard-errored.

No corpus re-run was done after restoring them. If the review queue looks noisy,
these two aliases are the first place to look, and removing one is an `UPDATE`.

## Resolved after the fact: the `order form` alias is dropped

The record above says 12 of 64 documents disagree with their declared type, all of
them the `order form` class, and that the alias is a product decision. **That decision
has been taken: drop it.**

`order form` is a genuine name for a call-off contract, which is why it was seeded.
On this corpus it is also the title cell that every quote-template workbook carries,
so it produced **12 disagreements out of 12 uses and no true positive**. Measured over
the same 50 live documents before and after, with every evidence span re-checked
byte-exact:

| | agreed | disagreed | unresolved |
|---|---|---|---|
| before | 38 | 12 | 0 |
| after | 47 | **0** | 3 |

The twelve were `Aureus_Workflow_Ltd` V1/V2/V3, `Lattice_Systems_Ltd` V1/V2/V3,
`Meridia_Cloud_Platforms_Ltd` V1/V2/V3 and `Orbis_Platform_Solutions_Ltd` V1/V2/V3 —
each declared `quote`, each reporting evidence of `doctype.call_off_contract`.

**Nine became `agreed`. Three became `unresolved`, and that is worth knowing**:
`Meridia V3`, `Orbis V2` and `Orbis V3` now tie between `doctype.quote` — the right
answer, carrying its own title evidence (`Quotation`, `Quote`) — and a type named only
in their body text (`contract_unspecified` twice, `schedule` once). Both sides are
tier 2, within the 1.0 margin, so the resolver declines to choose. They will raise an
`unresolved_document_type` review item carrying both candidates rather than a false
disagreement. That is the designed behaviour, and it is the generic-word problem again,
the same family as restoring bare `framework` and `notice`.

Shipped as:
- `src/services/concepts/seed.py` — the alias removed, with the reason inline.
- `deploy/sql/2026-10-01_concept_vocabulary_drop_order_form_alias.sql` + rollback —
  idempotent, scoped to one `concept_code`. **Applied to both databases**; alias md5
  parity re-verified (`a588615cdda3ca8c4fc8871f511f0a16`).
- `tests/services/extraction/test_type_resolver.py` —
  `test_order_form_is_not_a_call_off_alias_and_quote_workbooks_stay_agreed` pins it
  positively and carries the reason, so re-adding the alias goes red. Proven: with the
  alias restored in the seed, `resolve_alias` returns the call-off code and the real
  workbook row returns `disagreed / doctype.call_off_contract`; the file was restored
  byte-identically afterwards (sha1 `871d38fa…`).

**One consequence to note:** an alias is also an acceptable upload category, because
the gate and the classifier read the same column. `order form` is therefore no longer
accepted as a `process_monitor.category`. Nothing that routes today stops routing —
the only categories in use are `quote`, `invoice` and `po`.

Two test fixtures were rebuilt rather than relaxed, because both had been written on
this alias: the last-non-empty-cell row test keeps its call-off case on
`Call-Off Contract`, and the review-item grouping test now groups on a framework
agreement. Each says so, and what each tests is unchanged.

## The open-row key changed under this layer (same day, another session)

`099b123` added `source_file` to the open-row key:
`(doc_type, doc_pk_candidate, coalesce(source_file,''), issue_type, field_name)`,
migration `deploy/sql/2026-10-01_discrepancy_open_key_per_document.sql`. Every
reference above to the four-column key is therefore historical.

The reason is sound and it supersedes part of what this layer built: `doc_pk_candidate`
is a number read OUT of a document, not an identifier OF it, so two different documents
carrying one invoice number collided and the later overwrote the earlier — which is the
very case `duplicate_invoice_detector` exists to catch. A re-read keeps the same
`source_file`, so refresh-rather-than-stack still holds.

What it means for Task 7's keying: the `process_monitor:<id>` / `file:<path>` fallback
this layer added for pk-less documents is now belt-and-braces rather than the only thing
separating two such documents. It is left in place — it still carries the document's
identity where no pk was read at all.

One of this layer's tests had to be corrected, not relaxed:
`test_re_extraction_refreshes_the_open_finding_instead_of_stacking` varied `source_file`
across its three writes, which under the new key means three *different documents*. A
real re-read keeps it constant, so the fixture now does, and it asserts on `raw_id` to
prove the surviving row carries the latest run. A second test was added for the case the
new key half exists for — two files sharing one read value stay two findings.
