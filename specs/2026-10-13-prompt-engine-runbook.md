# Prompt engine runbook

How to see what is in use, put a new version into use, undo that, and find out
exactly what an agent was given in a past run.

Design and decision log: `2026-10-13-prompt-engine-phase1-design.md`.

All commands run on the server, from the project folder, and act on the
database the environment points at. There are two databases; a change made on
one is not made on the other.

```
cd /home/muthu/PycharmProjects/BP_Backend
set -a; . ./.env; set +a
PY="./venv/bin/python"; export PYTHONPATH=.:src
# for the other database, put PGDATABASE=bp_sqldb in front of a command
```

## 1. See what is in use

```
$PY scripts/prompt_pointer.py show <prompt name>
```

Lists every saved version (who wrote it, when, why), marks the one in use, and
shows every time the pointer moved and who approved it.

## 2. Put a new version into use (promote)

A version is written first and promoted second, by two different people.

1. **Write it.** Through the product (`POST /prompts/<name>/versions`), or for
   a prompt that was imported as it was:
   `$PY scripts/prompt_convert_to_typed.py <name> --names a,b,c --author <who> --apply`
   Saving changes nothing in use.
2. **Promote it.**
   `$PY scripts/prompt_pointer.py promote <name> <version> --approved-by <who> --note "why"`

It takes effect on the very next render. No restart.

Refused if: the approver wrote the version; the version does not exist; no
approver is named.

**Before promoting a typed version of a prompt that an agent used to fill
itself:** the agent's new code must already be running. Older code would read
the typed wording and fill it wrongly. Check with section 5.

## 3. Undo it (roll back)

```
$PY scripts/prompt_pointer.py rollback <name> --approved-by <who> --note "why"
```

One command. Returns to the version that was in use just before; add
`--to <version>` for an earlier one that was in use at some point. Takes effect
on the very next render. The version rolled away from is kept.

## 4. Find out what an agent was given in a run

```
$PY scripts/prompt_renders_for_run.py <run id>
$PY scripts/prompt_renders_for_run.py <run id> --text
```

The first lists each render in the run: when, which step and attempt, which
prompt and version, which fragment versions, which obligations were applied,
how many tokens, and the hash.

`--text` also prints the wording exactly as the agent received it, **with
customer values filled in**. Use it only where that may be shown.

The wording is kept for 7 days after it was last used (a policy value). After
that the listing still shows everything above; only the wording is gone.

## 5. Check the engine is healthy

```
$PY scripts/prompt_pointer.py show joshi          # the registry answers
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/prompts   # 401 = the routes are up
journalctl -u procwise --since "1 hour ago" | grep -i "prompt"
```

What the log lines mean:

| Line | Meaning | Do |
|---|---|---|
| `rendering prompt X failed (...); trying again` | The database did not answer. The step is waiting and retrying. | Nothing, unless it repeats. |
| `rendering prompt X did not complete in 30s` | It gave up. The step was refused; nothing ran on other wording. | Check the database. |
| `instructions X could not be recorded in the prompt registry` | An agent was saved but its instructions did not reach the registry. | Run `$PY scripts/import_prompts_to_registry.py --apply`. |
| `prompt render retention sweep failed` | The daily clean-up did not run. | Check the policy row states `manifest_retention_days`. |

## 6. The policy rows that govern it

In `proc.bp_policy`, changeable without a release (then
`POST /agents/reload-governance`):

| Row | Value | Controls |
|---|---|---|
| `PromptEnginePolicy` | `manifest_retention_days` 7 | How long rendered wording is kept after its last use. |
| | `wording_read_passthrough` true | Whether a prompt's wording is shown unaltered on its own screens. |
| | `render_retry_seconds` 30 | How long a render that can pass is retried before the step is refused. |
| `PromptWordingAuthorityPolicy` | Buyer | Who may read a prompt's wording. Every attempt is logged. |
| `PromptAuthorAuthorityPolicy` | Buyer | Who may save a new version. |
| `PromptPromoteAuthorityPolicy` | Approver | Who may promote or roll back. |

## 7. Tracing

Each render adds one span, `prompt.render`, to the trace of the step that asked
for it. It carries the run, step, attempt and decision ids, the prompt name and
version, the hash, the fragment versions, the token count and the time taken.
It never carries wording or values, and a failed render is marked with its code
only.

It is off until the tracing library is installed in the service's environment
and a trace store is running; until then rendering is unaffected.

## 8. If something is badly wrong

- **A bad version went live:** roll back (section 3).
- **An agent that was moved to the engine misbehaves:** roll its prompt back to
  the imported version; the agent returns to filling the wording itself.
- **Remove the engine's tables:** the rollback scripts in `deploy/sql/`
  (`2026-10-13_bp_prompt_render_rollback.sql`, then
  `2026-10-13_bp_prompt_registry_rollback.sql`, then
  `2026-10-13_prompt_engine_policies_rollback.sql`). This deletes every saved
  version and all history. The older prompt table was never altered.
