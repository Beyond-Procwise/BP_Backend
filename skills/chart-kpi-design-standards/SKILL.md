---
name: chart-kpi-design-standards
description: "Choose the right chart for a message and define a KPI properly - the chart type that fits the data and the question, the layout rules (title as the message, one unit, source line, no truncated axes, colour for meaning only), the red-amber-green thresholds and their basis, and the KPI definition sheet (what it measures, formula from the registry, data source, period, owner, target basis) - so every chart and KPI across reports reads the same way. Use when asked which chart to use, how to define a KPI, or to set traffic-light thresholds. Dormant pending review. Not for building the chart (render_chart) or the report (procurement-reporting)."
---

# Chart and KPI Design Standards

## Purpose

Keep charts honest and KPIs defined: a chart that shows what the data says and nothing more, a KPI that a second person could compute the same way, and thresholds whose basis is written down.

## When to use and when not

Use when a chart type, a chart's layout, a KPI's definition or a threshold is being chosen or checked.

Not for drawing (render_chart), for the report's content (procurement-reporting, executive-supply-chain-reporting), for the figures themselves (the data tools) or for the house style's typography (brand-house-style).

## Inputs and preconditions

- The data and the message the chart or KPI should carry.
- The reader and the cadence.

- Use `tool:list_calculations` for the registered calculations a KPI formula may use
- Use `tool:get_policy` for governed KPI definitions, targets and threshold rules
- Use `tool:render_chart` to draw the chosen chart when the user asks
- Use `tool:calc_supplier_scorecard` as the source of the delivery and price-variance measures a supplier KPI rests on
- Use `tool:savings_tracker` as the source of value measures
NOT AVAILABLE: render_kpi_card. not built yet; a KPI card is described (value, movement, target, status) and a chart drawn instead where useful

Settings this skill reads from its own configuration, named by key and never restated as numbers: `config.thresholds.max_donut_slices`.

## Workflow

1. Message first: what the reader should take away in one sentence; that sentence is the chart title.
2. Chart choice by data shape: parts of a whole with at most `config.thresholds.max_donut_slices` slices as a donut; a trend over time as a line; comparison across categories as bars; one number against a target as a KPI card; avoid pies for more than a few slices and dual axes.
3. Layout rules: one unit, axis from zero for bars, no truncated axes, a source line, colour only for meaning (status, highlight), at most two highlights.
4. KPI definition sheet: name, what it measures, the registered calculation or the tool that computes it, data source, period, owner, target and its basis (policy, contract, baseline), thresholds and their basis; a KPI with no registered calculation or tool is defined as not computable yet.
5. Thresholds: red, amber, green with the rule and its basis; a threshold without a basis is a default and is named as one.
6. Draw with render_chart when the user asks; otherwise hand the specification to the report skill.

## Frameworks and heuristics

- If the title is not a sentence, the chart has no message.
- A truncated axis is a lie by omission.
- A KPI nobody can recompute is an opinion with a number.
- Three colours, three meanings; more is decoration.

## Policy hooks (SC-007)

Governed KPI definitions, targets and thresholds come from `tool:get_policy` and are quoted; defaults are named as defaults. The skill draws only with the values the tools returned and changes no figure.

## Output specification

1. The message sentence and the chart type with the reason.
2. Layout checklist applied.
3. KPI definition sheet, including whether it is computable today.
4. Thresholds with their basis.
5. The chart's export id when drawn.

## Quality bar

- The chart type fits the data shape and the message.
- Every KPI names its calculation or tool, data source and period.
- Every threshold has a basis or is named a default.
- No figure is changed.

## Escalation and approval

Escalate to the KPI owner where a definition is disputed, to the policy owner where a target has no basis, and to the data owner where the source cannot compute the KPI.

## Worked example

Asked which chart to use for supplier spend concentration, the skill writes the message (the top five suppliers hold most of the spend), chooses a bar chart sorted by spend with the top five highlighted, applies the layout rules, defines the concentration KPI (share of spend held by the top five via the registry's share calculation, source the spend cube, period the fiscal year, owner the CPO, target from policy or absent), sets thresholds as defaults named as such, and draws the chart with render_chart from the cube's figures.

## Anti-patterns

- A pie with twelve slices.
- A KPI defined by a formula the registry does not hold.
- A threshold with no basis presented as policy.
- Changing a figure to make the chart cleaner.
