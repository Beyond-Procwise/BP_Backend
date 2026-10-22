---
name: root-cause-analysis
description: "Find the cause behind a recurring problem - a quality defect, a late delivery pattern, an invoice error pattern - with the five whys and a cause-and-effect (fishbone) structure, from the findings and records we hold and the facts the user supplies, separating what the evidence shows from what is suspected, and ending in a cause statement with the evidence that supports it and the test that would confirm it. Use when asked why something keeps happening or for a root cause. Dormant pending review. Not for the supplier-facing 8D (eight-d-problem-solving) or the corrective action itself (capa-management)."
---

# Root Cause Analysis

## Purpose

Get to a cause the evidence supports, not the first plausible story: each why answered with a fact or marked as a guess to test, so the corrective action fixes the cause and not the symptom.

## When to use and when not

Use when a problem has recurred and the question is why, or when a cause is claimed and needs testing.

Not for the structured supplier response (eight-d-problem-solving), for the corrective action plan (capa-management), for a one-off bad delivery (delivery-issue-handling) or for the claim (dispute-and-claims).

## Inputs and preconditions

- The problem statement: what, where, when, how often, with the user's evidence.
- The data the user holds: inspection results, incident reports, timelines.

- Use `tool:invoice_reconciliation` for the findings pattern on a supplier's documents (price, quantity, no-order) as evidence
- Use `tool:calc_supplier_scorecard` for the supplier's delivery and findings pattern over the period
- Use `tool:open_orders` for order dates, totals and buyer ids where the problem is an ordering pattern
- Use `tool:supplier_items` for the items involved and when they were bought
- Use `tool:run_calculation` with `deal.share_pct` for a cause's share of occurrences, `quality.ppm` for a defect rate the user's inspection counts give, and `deal.pct_change` for the change after a suspected trigger
- Use `tool:get_policy` for the problem-solving rules (when a formal RCA is required, who owns it)
NOT AVAILABLE: log_non_conformance. no non-conformance register; the user's inspection and incident records are the evidence, labelled
NOT AVAILABLE: log_inspection. as above

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_whys`.

## Workflow

1. Define the problem precisely: what, where, when, how often, how big, with the evidence (findings, scorecard, the user's records); a problem without a measure is sized first.
2. Timeline: when it started, what changed around then (supplier, spec, process, season) from the records and the user; the change after a suspected trigger via `deal.pct_change`.
3. Causes by category (fishbone): material, method, machine, people, measurement, environment; for each, the candidate causes with the evidence for and against; a cause with no evidence is a hypothesis.
4. Five whys on the leading candidates, each why answered with a fact or marked 'to test'; stop at a cause we can act on, at most `config.thresholds.max_whys` levels.
5. Weigh: each candidate's share of occurrences via the registry where the data allows; the cause statement with its evidence; the test that would confirm it (a measurement, a trial) and who runs it.
6. Hand to capa-management for the action, or to eight-d-problem-solving where the supplier owns the problem.

## Frameworks and heuristics

- A cause you can act on is where the whys stop; 'human error' is never a root cause.
- What changed when the problem started is the best clue.
- Evidence for and against; a cause nobody tried to disprove is a story.
- Several small causes beat one big guess; share them out.

## Policy hooks (SC-007)

Problem-solving rules come from `tool:get_policy` and are quoted. The skill analyses; it changes no process, blames no person and records no non-conformance.

## Output specification

1. Problem statement with measure and evidence.
2. Timeline and changes.
3. Fishbone: causes by category with evidence for and against.
4. Five-whys chains with facts and tests.
5. Cause statement, share of occurrences, confirmation test, owner; hand-off.

## Quality bar

- Every cause has evidence or is marked a hypothesis.
- Shares and rates come from the registry.
- The cause is actionable.
- No person is named as the cause.

## Escalation and approval

Escalate to quality for a confirmation test, to the supplier through eight-d-problem-solving where the cause is theirs, and to capa-management for the action.

## Worked example

Asked why the packaging line keeps stopping on film breaks, the skill sizes the problem from the user's incident log, finds the breaks started the month a new film supplier was used (supplier_items), lays out the fishbone with the film's specification, the machine tension and the operator change as candidates, walks the whys on the film with the user's test results, finds most breaks on one supplier's rolls via the registry, states the cause as that supplier's film gauge below specification with the test (gauge measurement on the next delivery) and hands to eight-d-problem-solving.

## Anti-patterns

- Stopping at the first plausible cause.
- Blaming a person.
- A fishbone with no evidence column.
- Skipping the confirmation test.
