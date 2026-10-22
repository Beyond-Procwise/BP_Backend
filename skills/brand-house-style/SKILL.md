---
name: brand-house-style
description: "The formatting, tone and template rules every output follows - headings, numbers, dates, currency, tables, the words the organisation uses and avoids, what every document, deck and chart carries (title, date, source line, caveats) - read from the governed style policy and applied when a file is built. Use when asked how an output should look, to apply the house style, or to check a draft against it. Dormant until the style policy exists; until then the defaults here are named as defaults. Not for the content of a document (the owning skill) or for building it (the export tools do that)."
---

# Brand and House Style

## Purpose

Make every output look like it came from the same organisation: the same structure, the same way of writing a number or a date, the same source line and caveat, the same tone. The skill reads the governed style and applies it; it writes no content of its own.

## When to use and when not

Use when a document, deck, chart or answer is being prepared for a reader and the user asks for the house style, a style check, or how an output should be laid out.

Not for what the output says (the owning skill), for building the file (build_docx, build_pptx, build_xlsx, render_chart) or for a UI design question.

## Inputs and preconditions

- The draft or the output to be built, and its reader.
- The governed style policy, or the user's instructions labelled user-supplied.

- Use `tool:get_policy` for the style rules (numbers, dates, currency, headings, mandatory lines, tone, banned words)
- Use `tool:build_docx` to rebuild a document to the style when the user asks
- Use `tool:build_pptx` to rebuild a deck to the style when the user asks
- Use `tool:render_chart` to redraw a chart to the style when the user asks
NOT AVAILABLE: apply_template. no template engine exists; the style is applied by building the file with the export tools to the rules read here

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.decimal_places`.

## Workflow

1. Read the style policy once; where none governs, use the defaults in this playbook and say they are defaults.
2. Structure: every document opens with a title, the date, the reader and a one-paragraph summary; every deck with a title slide and a summary slide; every chart with a title, a unit and a source line; every table with a header row and units in the header.
3. Numbers: thousands separators, at most `config.thresholds.decimal_places` decimals, currency code after the figure, percentages as whole points unless the figure is under ten; dates as day month year in words; periods stated once at the top.
4. Words: plain English, active voice, one idea per sentence; no internal names (tables, tools, routes); the organisation's preferred terms where the policy lists them; caveats in their own line, not in brackets.
5. Check a draft against these rules and list the deviations with the fix; rebuild the file with the export tools when the user asks.

## Frameworks and heuristics

- Consistency beats elegance; a reader should not notice the style.
- A number without a unit, a date without a year and a chart without a source are the three commonest faults.
- The summary is written last and read first.

## Policy hooks (SC-007)

The style rules come from `tool:get_policy` and are quoted; this playbook's defaults apply only where the policy is silent and are named as defaults. No content is changed, only its form; nothing is sent.

## Output specification

1. The rules applied, with their source (policy or default).
2. For a check: the deviations and their fixes.
3. For a rebuild: the export id of the rebuilt file.

## Quality bar

- Every rule cited is from the policy or named as a default.
- No figure, date or caveat is changed in substance, only in form.
- No internal name reaches the output.

## Escalation and approval

Escalate to the communications owner where the policy is silent on a case that recurs, and to the owning skill where a style fix would change meaning.

## Worked example

Asked to put the award memo into house style, the skill reads the style policy (date format, currency code after figures, summary first, source line on every table), lists the memo's deviations (dates in numerals, currency symbols, no summary), and rebuilds the memo with build_docx in the right form with the content unchanged.

## Anti-patterns

- Rewriting the substance while fixing the form.
- Inventing a style rule the policy does not hold and calling it policy.
- Leaving a chart without a unit or a source.
