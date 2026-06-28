# AgentNick — Live Capability Analysis (2026-06-28)

End-to-end exercise of AgentNick across its real jobs on live models.
Full transcript: `agentnick_analysis.log`.

| Capability | Model | Result | Verdict |
|---|---|---|---|
| **Extraction** | `:extract` | doc_accuracy **0.865** (12 gold docs) | **Strong** — its tuned strength |
| **Domain Q&A** | `:latest` | PO vs Invoice + 3-way match, all correct (3/3) | **Strong** — real procurement knowledge |
| **Reasoning / planning** | `:latest` | **Refused/deflected** — debated "outside the scope of document extraction" instead of planning; 296 s (timeout+retry) | **Weak (mis-conditioned)** |
| **Negotiation strategy** | `:latest` | **Refused** — "I must not generate a counter-offer. I must only do document extraction." | **Weak (mis-conditioned)** |
| **Summary** | `:latest` | Hesitant/verbose, debated whether to comply before answering | **Partial** |

## Key finding (actionable)

The underlying model (qwen3:30b) is **capable** — it nailed domain Q&A and has the
knowledge. But `AgentNick:latest`'s **system prompt is over-conditioned for
extraction** ("Your primary role is document extraction with 100% accuracy"), so
for agentic tasks (planning, negotiation, summarisation) it spends its output
**debating whether it's even allowed to answer** — and often **refuses**. This
cripples the agentic intelligence the other 12 agents depend on.

This confirms the prior "Dynamic Planner Diagnosis" (extraction-tuned AgentNick
refuses planning). It is a **configuration** problem, not a model-capability one.

Secondary: the qwen3:30b "thinking" path is **slow and verbose** for agentic calls
(73–296 s), emitting long internal deliberation.

## Recommended fix (next effort)

Give the agentic/reasoning paths a model variant whose system prompt embraces the
**full** AgentNick role (extraction AND planning/negotiation/summarisation), or a
separate agentic Modelfile — so the model stops refusing non-extraction work.
Keep the extraction-specialist (`:extract`) as-is (it's strong). This is the real
lever for "AgentNick power & intelligence", and it's gateable the same way
(measure agentic task success before/after).
