---
name: powerpoint-pack-builder
description: "Turn finished content into a presentation pack - the storyline (one message per slide), the slide order (summary first, evidence, decisions), the tables and charts each slide carries from the figures already established, speaker notes with the sources, and the house style - built with the deck tool. Use when asked to make slides, a deck or a pack out of an analysis, a report or a recommendation that already exists. Dormant pending review. Not for the analysis itself (the owning skill) or the style rules (brand-house-style)."
---

# Presentation Pack Builder

## Purpose

Make a deck that carries the argument: one message per slide, the evidence beside it, the source in the notes, the decision at the end; nothing on a slide that the content did not already establish.

## When to use and when not

Use when content is done and needs to become slides.

Not for producing the content (the owning skill), for the style policy itself (brand-house-style), for chart choice rules in depth (chart-kpi-design-standards) or for a written document (build_docx through the owning skill).

## Inputs and preconditions

- The content: the answer, report or recommendation with its figures and sources.
- The reader, the time slot, the decision wanted.

- Use `tool:build_pptx` to build the deck with titles, bullets, small tables and speaker notes
- Use `tool:render_chart` for a chart per slide where a movement or a share needs one, from established figures
- Use `tool:get_policy` for the house style rules if governed
NOT AVAILABLE: apply_template. no template engine; the style is applied in the content the deck tool receives

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_slides`.

## Workflow

1. Storyline: the one-sentence message the reader should leave with; the slide messages that build to it, at most `config.thresholds.max_slides` slides for the time slot, each title a sentence.
2. Order: summary and the ask first; the evidence slides in the order of the argument; risks and caveats; the decision slide with options and the recommendation; appendix for detail.
3. Per slide: the bullets (few), a small table or a chart only where it carries the message, with the figures exactly as the content states them; speaker notes with the source of every figure and the caveats.
4. Style: the house style rules applied (numbers, dates, units, source lines).
5. Build with the deck tool; charts drawn from the established figures; the export id returned; nothing presented or sent.

## Frameworks and heuristics

- A title is a sentence; a label is a wasted slide.
- One message per slide; a second message is the next slide.
- The notes carry the sources; the slide carries the point.
- The decision slide is why the deck exists.

## Policy hooks (SC-007)

House style comes from `tool:get_policy` through brand-house-style. The deck carries only figures the content established; the skill decides nothing and sends nothing.

## Output specification

1. Storyline and slide list with messages.
2. Deck export id; chart export ids.
3. Notes summary: sources and caveats per slide.

## Quality bar

- Every figure on a slide is in the content with its source.
- Every title is a sentence.
- Slides within the limit for the slot.
- Nothing sent.

## Escalation and approval

Escalate to the owning skill where a slide needs a figure the content lacks, and to brand-house-style where the style rules are unclear.

## Worked example

Given the award recommendation and a fifteen-minute slot, the skill writes the storyline (award to B on quality and delivery with A as the alternative), lays out seven slides (summary and ask, the evaluation method, scores, cost comparison chart, risks and screening, the decision with options, appendix), puts the sources and caveats in the notes, draws the cost chart from the established figures, applies the house style and builds the deck.

## Anti-patterns

- A figure on a slide the content did not establish.
- Twenty slides for ten minutes.
- Titles that are labels.
- Sending the deck.
