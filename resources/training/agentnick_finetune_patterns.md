# AgentNick — patterns worth learning from (capture log)

Generalizable patterns only. **No raw user data, no document contents, no supplier
identities** — every entry is a *shape* or a *rule*, which is the Phase 3 constraint
("learn from PATTERNS, not stored user data").

Captured while fixing extraction + orchestration on 2026-07-12. Each entry records what
actually went wrong, why a model/pipeline got it wrong, and the invariant that would have
caught it.

---

## A. Extraction — what the model must be taught

### A1. The document's own printed totals outrank any computed figure
**Observed:** a PDF printed `SUB TOTAL 1169.58 / TAX (20%) 233.920 / GRAND TOTAL 1403.50`.
The pipeline persisted `584.79 / 116.96 / 701.75` — the **unit price**, plus tax and total
*derived from it*. Every derived figure was self-consistent, so every arithmetic check passed.

**Pattern:** when a document explicitly labels a total, that literal is authoritative.
A value derived from another value must never outrank a value read from the page.

**Training signal:** prefer `label -> value on the same line` over positional/column inference.
Negative example: binding a line's PRICE column to the invoice's subtotal.

### A2. `qty x unit_price != line_amount` is the single highest-value invariant
**Observed:** every invoice in the corpus had **qty = 1**, where unit price coincidentally
equals the line amount. A 2x understatement was therefore invisible for the entire history of
the system. It only appeared on the first qty=2 document ever ingested.

**Pattern:** a corpus whose qty is always 1 cannot validate money extraction.
**Eval-set rule:** the eval set MUST contain qty > 1 documents, or this whole class of error
scores 100%.

### A3. A NULL is not a pass
**Observed:** `line_amount = NULL` caused the reconciliation check to **skip**, not fail. A
line with a quantity and a unit price but no amount sailed through at 94.44% confidence.

**Pattern:** for a field the schema says is derivable, absence is a *finding*, not a silent
skip. "Could not read" and "read and it was fine" must never collapse to the same outcome.

### A4. Confidence must not be computed from self-consistent derived values
**Observed:** confidence 94.44% on a document whose money was wrong by 2x, because the
derived figures agreed with each other.

**Pattern:** confidence must be conditioned on *how many fields were captured vs inferred*.
A row whose tax and total were both inferred deserves lower confidence than one where both
were read. (Now recorded explicitly as the `value_derived` discrepancy.)

### A5. Grounding must be checked against the FULL text, not a lossy parse
**Observed:** the parser (docling; and separately PyMuPDF) **clipped the last table column**.
The correct value therefore did not appear in the text the grounding guard checks against, so
the guard rejected the model's *correct* reading as a hallucination.

**Pattern:** a hallucination guard is only as good as its grounding corpus. If the guard
rejects a value, the first hypothesis should be "the corpus is incomplete", not "the model
lied". Union multiple text backends before grounding.

**Training signal:** do NOT train on "model said X, guard rejected X" as a negative example
without first confirming X is genuinely absent from the page. Those are poisoned labels.

### A6. Free-form output where JSON was expected = total loss
**Observed:** AgentNick read a line-item table **perfectly**
(`Acer | TravelMate P2 | Qty: 2 | Price: 584.79 | Total: 1169.58`) but answered in prose. The
JSON parser returned `[]` and the pipeline silently fell back to the (wrong) regex reading.

**Pattern:** the model is *capable* and the harness threw the answer away. Two lessons:
1. Always grammar-constrain (Ollama `format=<schema>`) — do not rely on prompt discipline.
2. Prose-when-JSON-was-asked is a **finetune signal**: the model needs more schema-following
   examples for the line-item task specifically. It follows the schema on the header task
   (which was constrained) and not on the line-item task (which was not).

---

## B. Orchestration — patterns for routing/agent behaviour

### B1. An empty result must not erase a populated blackboard value
**Observed:** an agent returning `supplier_candidates: []` **overwrote** a caller-supplied
list via a blanket `dict.update()`. The downstream edge condition then evaluated False and the
ranking agent was skipped on **every run since the graph was written**.

**Pattern:** "I found nothing" is not the same as "there is nothing". A node may introduce a
key with any value; it may not blank out a key another party populated.

### B2. Silent-skip is worse than failure
**Observed:** a graph node that never fires, a table nobody reads, an agent whose findings go
to a table that does not exist. All three reported *success*.

**Pattern:** a pipeline should be able to answer "did this stage actually do anything?" A
`skipped` status that looks identical to `completed` in the response is a defect.

### B3. Docstrings drift; wiring is truth
**Observed:** `build_extraction_workflow`'s docstring claimed
`extract_documents -> discrepancy_detection` and had claimed it for as long as the function
existed. **No such node was ever added.**

**Pattern:** when reasoning about the system, never trust a comment over the wiring. Applies
to the model too: do not let a docstring in context override observed behaviour.

---

## C. Eval-set requirements this implies

An eval set that would have caught today's bugs:
1. An invoice with **qty > 1** on at least one line (catches A2 — the entire corpus lacked one).
2. An invoice whose table **runs to the page edge** / has a clipped final column (catches A5).
3. An invoice that **states** SUB TOTAL / TAX / GRAND TOTAL explicitly (catches A1).
4. An invoice with a **missing** line amount (catches A3).
5. A document whose printed totals are **internally inconsistent** — the source is wrong and
   must be captured wrong, then flagged (tests "do not silently correct").
6. A multi-supplier tender (catches the "sum of all quotes treated as one quote" error).

## D. What must NOT be learned/stored
- No supplier names, invoice numbers, amounts, addresses, or document text.
- Patterns are structural: field-binding rules, invariant shapes, output-format compliance.
- A rejected-by-grounding value is NOT automatically a hallucination (A5) — do not mine those
  as negatives without verifying against the source.
