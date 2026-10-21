# Platform Skills: Design and Plan (batch 1)

> Source: "Procurement & Supply Chain Agent Library — Build Prompts" (21 Oct 2026, 125 prompts, SES v1). This batch builds four of its 50 skills. Design approved by Nick in chat 2026-10-21.

**Goal:** Skills (packaged playbooks) the agent loads when a user needs an action done, starting with SC-007 Procurement Policy & Compliance, SC-036 Incoterms Selection, SC-027 Negotiation Planning, SC-017 Savings Methodology.

**Architecture:** Each skill is a folder `skills/<slug>/` (SKILL.md, config.yaml, references/, templates/, evals/). A loader in `src/services/skills/` validates and serves them. Two tools, `list_skills` and `use_skill`, are added to AgentNick's tool set. Skills advise only: writes still go through the existing approval flow. A skill never states a number that did not come from a tool result, a user document or a labelled assumption.

**Tech stack:** Python 3.12, PyYAML, pytest. Run tests with `venv/bin/python -m pytest` (see memory: venv runs tests, .venv runs the app).

## Global constraints (copied from SES v1 section C)

- SKILL.md has YAML frontmatter: `name`, and `description` of at most 1,024 characters saying what it does and when to trigger.
- SKILL.md sections, in this order: Purpose · When to use and when not · Inputs and preconditions · Workflow · Frameworks and heuristics · Policy hooks (SC-007) · Output specification · Quality bar · Escalation and approval · Worked example · Anti-patterns.
- SKILL.md under about 500 lines; long material goes in `references/`.
- Every number traces to a tool result, a user-supplied document or a labelled assumption. Never invent prices, scores, supplier facts or regulations.
- Evals: at least 15 scenarios (typical, edge, missing data, conflicting data, adversarial/pressure to skip policy), at least 3 flagged for expert grading, plus a trigger eval of 10 prompts that should activate and 10 that should not.
- Org-tunable thresholds live in `config.yaml`, never in SKILL.md prose.
- Local rules: AgentNick is the only base model; absent data stays NULL (say "not available", never fill); commit only my own files; nothing pushed.

## Review focus (failure modes no task's tests would otherwise hit)

1. A skill names a tool that does not exist (the library's tools are 75, ours are about 12). Pinned by Task 3's tool-dependency check.
2. A skill tells the agent to approve, send or award. Pinned by Task 3's "advisory only" check.
3. `use_skill` called with an unknown or path-like name (`../x`). Pinned in Task 2.
4. A thresholds value in config.yaml that SKILL.md prose contradicts. Pinned in Task 3 (every number in config appears nowhere as a different literal for the same key is hard to test; we test that SKILL.md refers to config keys by name instead of repeating the value).
5. A skill folder with a missing eval file loading silently. Pinned in Task 1.

## File structure

- `skills/<slug>/` content, one folder per skill.
- `src/services/skills/__init__.py`, `loader.py` (read/validate/serve), `checks.py` (structure + dependency + advisory checks, used by tests and by the loader).
- `src/orchestration/agentnick_control.py`: add `_skill_tools()` and include it in `build_tools`.
- Tests: `tests/skills/test_loader.py`, `test_checks.py`, `test_skill_tools.py`, `test_each_skill.py` (parametrised over every folder).

## Tasks

### Task 1: Loader and structure checks
Files: create `src/services/skills/{__init__,loader,checks}.py`, `tests/skills/test_loader.py`, `tests/skills/fixtures/` (one valid mini skill, plus broken variants).
Produces: `list_skills() -> list[SkillSummary]`, `load_skill(name: str) -> Skill` (raises `SkillNotFound`, `SkillInvalid`), `Skill(name, description, body, config, references: dict[str,str], eval_files: list[str])`, `check_structure(path) -> list[str]` (problems; empty = valid). Root directory from env `SKILLS_DIR`, default `<repo>/skills`.
- [ ] Failing tests: valid fixture loads; each of these returns a named problem: description over 1,024 chars; a required section missing; sections out of order; more than 500 lines; config.yaml missing or not a mapping; `evals/scenarios.yaml` or `evals/triggers.yaml` missing.
- [ ] Implement; tests green.
- [ ] Mutation: delete the section-order check, confirm the out-of-order test goes red, restore.
- [ ] Commit.

### Task 2: use_skill and list_skills tools
Files: modify `src/orchestration/agentnick_control.py` (add `_skill_tools()`, extend `build_tools`), create `tests/skills/test_skill_tools.py`.
Produces: tool `list_skills` (no args; slug + description) and `use_skill(name)` returning `{name, playbook, config, references_available}`. Unknown name returns `{found: false, available: [...], note}`; names containing `/`, `..` or not matching `^[a-z0-9-]+$` are refused the same way. Tool descriptions tell the model a skill is advice: calls to tools and approvals still apply.
- [ ] Failing tests: both tools appear in `build_tools(...)`; `use_skill("../etc/passwd")` and an unknown name return found=false; a valid fixture skill returns its playbook.
- [ ] Implement; green. Also confirm `tests/test_auto_registry.py` and `tests/test_agent_execute_accepts_tool_loop.py` still pass.
- [ ] Mutation: remove the name pattern check, confirm the traversal test goes red, restore. Commit.

### Task 3: Whole-skill checks (apply to every skill folder)
Files: extend `checks.py`; create `tests/skills/test_each_skill.py` (parametrised over `skills/*`).
Checks:
- Tool dependency: every backticked tool name in Workflow is either a name returned by `build_tools` or listed under `unavailable_tools:` in config.yaml; for each unavailable tool SKILL.md must contain a line `NOT AVAILABLE: <tool>` saying what the agent does instead.
- Advisory only: SKILL.md may not instruct the agent to call a tool in a denied list (`commit_*`, `send_message`, `run_email_dispatch`, `e_signature`).
- Config by reference: no numeric literal listed in config.yaml `thresholds:` is repeated as a bare value in SKILL.md; the prose names the key (`config.thresholds.x`).
- Evals: scenarios.yaml has at least 15 entries covering all five kinds, at least 3 with `expert_graded: true`, at least one `kind: pressure_to_skip_policy`; triggers.yaml has 10 `should_trigger` and 10 `should_not_trigger`.
- [ ] Failing tests against a deliberately bad fixture for each check; then implement.
- [ ] Mutation per check (break the check, watch the matching test go red). Commit.

### Task 4: SC-007 Procurement Policy & Compliance
Files: `skills/procurement-policy-compliance/...`, using the live policy tools (`get_policy`, `list_governance`, `get_corpus_facts`, `run_calculation`). Spec = prompt SC-007 verbatim for sections, verdict contract and the 20 scenarios.
Decisions: the policy source is the governed `bp_policy` rows, not a new rules file; where no row covers a rule (for example bid thresholds), the verdict is `INSUFFICIENT_INFO` naming the missing policy, never a guess. Tools the library names that do not exist here (`supplier_lookup`, `search_contracts`, `query_erp`) go under `unavailable_tools` with a fallback (`get_corpus_facts`, or ask the user). Verdict values: COMPLIANT, COMPLIANT_WITH_CONDITIONS, NON_COMPLIANT, INSUFFICIENT_INFO.
- [ ] Write evals first (20 scenarios incl. 99,500 EUR under a 100,000 threshold, 4 POs of 24,000 in 3 weeks, 3-year SaaS 40k a year = 120k commitment, self-approval, expired waiver, pressure to skip the bid).
- [ ] Write SKILL.md, config.yaml, templates/verdict.json; `test_each_skill` green. Commit.

### Task 5: SC-036 Incoterms Selection
Uses `run_calculation` with `landed_cost.total_cost`. Config: rule-by-mode table (EXW/FCA/CPT/CIP/DAP/DPU/DDP any mode; FAS/FOB/CFR/CIF sea only), insurance minimums (CIF = ICC (C), CIP = ICC (A), 110%). `duty_calc`, `get_freight_quote`, `hs_code_lookup` are unavailable: the agent asks for a duty rate and freight figure and labels them user-supplied. The 15 scenarios are the prompt's (FOB for 40' containers must redirect to FCA; "Incoterms 2010 DAT" maps to DPU). Commit.

### Task 6: SC-027 Negotiation Planning
Uses `negotiation.*` calculations, the opportunity lever tools, `get_corpus_facts`, `get_policy`. `should_cost_model`, `commodity_index_lookup`, `financial_health_score` are unavailable: target and walk-away without a should-cost are labelled estimates or the agent asks; index pass-through needs a user-supplied index move. Review-gated: output states a named category lead must approve targets before use. Commit.

### Task 7: SC-017 Savings Methodology
Uses `deal.realised_savings`, `deal.pct_change`, `get_corpus_facts`, the value-ledger facts. `savings_tracker` and `commodity_index_lookup` unavailable: the skill produces the validation sheet and leaves status DRAFT with a missing-evidence list; index neutralisation requires a user-supplied index series and is otherwise reported as "not neutralised". Rules kept from the prompt: cost avoidance never added to hard savings; soft savings valued at WACC but excluded from P&L totals; volume effect separate. Commit.

### Task 8: Whole-branch verification
- [ ] `venv/bin/python -m pytest tests/skills tests/test_auto_registry.py tests/test_agent_execute_accepts_tool_loop.py -q` green.
- [ ] Live check on the local server: ask AgentNick a Policy question with `use_skill` on offer and confirm it fetches the skill and then calls `get_policy` (needs a restart; ask before restarting the shared server).
- [ ] Summarise what each skill cannot do yet and which library tools unlock it.

## Self-review notes

Spec coverage: loader, tools, checks and four skills are all tasked. Items deliberately not built: a `bp_skill` database registry (files first; revisit after batch 1), the 15 skills waiting for tools, and any UI.
