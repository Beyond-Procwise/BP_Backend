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

### A7. A prior in the prompt becomes a hallucination in the output
**Observed (2026-07-13):** the extraction prompt contained the line
*"UK invoices very commonly use 20% VAT."* A **US** invoice that printed
`Tax (10%): $2,861.00` two lines above its total was persisted with `tax = 5,316.00` —
exactly `subtotal × 20%`. Across eight variants of one document the tax came out as
20%, 10%, 10%, 10%, 20%, **7.69%**, 20%, 20%: non-deterministic, and mostly invented.

**Pattern:** a helpful-sounding statistical prior ("usually 20%") is indistinguishable, at
generation time, from a fact about *this* document. The model reaches for it whenever the
real value is hard to read. **Do not put population statistics in an extraction prompt.**
State the rule instead: *there is no default rate — read the one printed, or output null.*

### A8. The word "Total" in a label does not make it the total
**Observed (2026-07-13):** a vehicle invoice printed
`Vehicle Price: $26,580.00 / Accessories Total: $2,030.00 / Tax (10%): $2,861.00 /
Dealer Fee: $450.00 / Total Sale Price: $31,921.00`.
The model reported `tax = 2,030.00` (the **Accessories Total**) and the net as
`26,580.00` (the **Vehicle Price** — one component of five).

**Pattern:** documents label components with the word "Total" (`Accessories Total`,
`Line Total`, `Sub|total|`, a `Total` table-column header). The grand total is the value
that settles the **whole document** — normally the last money value and the largest.
Teach the distinction explicitly; a regex that matches bare `total` makes the identical
mistake (it read `Sub**total**: $1964.00` as the invoice total until anchored to line-start).

### A9. Where a document itemises and never prints a subtotal
**Observed (2026-07-13):** on the layout above there is no `Subtotal:` line at all. The
model reported the largest single component (the vehicle price) as the invoice net.
The true pre-tax net is `grand_total − tax = 31,921.00 − 2,861.00 = 29,060.00`
(= 26,580 + 2,030 + 450 ✓).

**Pattern:** absence of a `Subtotal` label is not permission to substitute a component.
Derive the net from two values that ARE printed. Arithmetic over grounded inputs is
legitimate; picking a component and calling it the net is not.

### A10. Charges sit between the subtotal and the total — closure must allow them
**Observed (2026-07-13):** `Subtotal: $1964.00  Shipping: $9.20  Tax: $196.40 /
Total: $2169.60`. A closure check of `subtotal + tax == total` **fails** here by exactly
the shipping (9.20) and would reject a perfectly consistent invoice.

**Pattern:** the invariant is `subtotal + charges + tax == grand_total`, where charges are
shipping / freight / handling / dealer fees. Get this wrong and the checker rejects good
documents, which trains everyone to ignore it.

### A11. Self-check the arithmetic BEFORE answering
**Pattern (the invariant that would have caught A7–A10):** a money block that does not add
up is wrong even when each figure looks plausible on its own. `subtotal + charges + tax`
must equal `grand_total`. Teach the model to re-read the block when it does not balance.

**How it was applied:** three independent signals (model / printed-label reads / closure),
with the *document's own arithmetic* as referee. The model wins whenever its figures close
— so layouts it already handles are untouched — and printed values only take over when the
model's set does not close and theirs does. When neither closes, keep what is grounded and
leave the rest NULL for the discrepancy engine. **Never synthesise the missing number.**

**Measured effect of teaching this in the prompt (2026-07-13):** on the apparel invoice
AgentNick alone went from `tax = null` (→ fabricated 98.20) to reading `196.40` correctly;
on the vehicle invoice from a fabricated `5,316.00` to the printed `2,861.00`. It still
reports a component as the net on the itemised layout and still emits a `tax_percent` that
is not printed — i.e. **prompt teaching moved it a long way and did not finish the job**.
The closure reconciliation is what makes the result correct; the prompt is what makes the
model's own answer usable. Both are needed. This is the A6 lesson again: do not rely on
prompt discipline alone.

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
7. A **non-UK invoice stating a non-20% tax rate** (catches A7 — the 20% prior). The whole
   corpus was UK/20%, which is precisely why a hallucinated 20% scored as correct.
8. An invoice with a **component labelled "… Total"** that is not the grand total
   (catches A8), and one that **itemises with no `Subtotal` line at all** (catches A9).
9. An invoice with **shipping/freight between subtotal and tax** (catches A10 — a closure
   check of `subtotal + tax == total` wrongly rejects it).
10. An invoice in a currency whose **symbol contradicts the corpus default** (a `$` document
    in a GBP-heavy corpus was booked as GBP, silently rescaling `converted_amount_usd`).

> Note on scoring: items 7–10 are all *money* failures that an eval built from the existing
> corpus reports as 100% correct, because the corpus contains no example of any of them.
> An eval set that cannot fail is not measuring anything — same trap as the qty=1 blind spot
> in item 1.

## D. What must NOT be learned/stored
- No supplier names, invoice numbers, amounts, addresses, or document text.
- Patterns are structural: field-binding rules, invariant shapes, output-format compliance.
- A rejected-by-grounding value is NOT automatically a hallucination (A5) — do not mine those
  as negatives without verifying against the source.
