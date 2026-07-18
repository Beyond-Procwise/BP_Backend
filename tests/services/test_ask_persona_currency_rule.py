"""The /ask persona must forbid cross-currency arithmetic.

Regression for a live defect: asked for "total invoiced spend", the assistant replied
"£190,400.61 (GBP) and $97,519.00 (USD), summing to a total of £287,919.61" — adding
two currencies 1:1 and labelling the result GBP (a ~9.5% overstatement; the converted
figure is £262,938.75).

The data layer was never at fault. corpus_facts._fetch("spend") deliberately emits only
a per-currency breakdown, with the comment "a single grand total would be a lie". The
sum was the model's own arithmetic over that split, and nothing forbade it: the rule
existed only in the AgentNick tool-calling prompt (orchestration/agentnick_control.py),
not on the RAG /ask path. The persona's "never name an amount not in the context" rule
does not catch it either — both operands ARE in the context; the total is a new number
derived from them.

So the guarantee under test is a prompt guarantee, and these assert on the persona text
that reaches the model. They deliberately do NOT call the LLM: an inference-dependent
assertion would be flaky and slow, and the failure mode being prevented is "the rule
silently stopped being sent", which is exactly what text inspection catches.
"""

import re

import pytest


def _persona_texts():
    """Both persona sources: the code fallback and, when reachable, the bp_prompt row.

    The DB row is what actually governs at runtime (_ask_persona prefers it), so a rule
    present only in the constant would not protect production. The DB half is skipped
    rather than failed when bp_sqldb is unreachable, so the suite still runs offline.
    """
    from src.services.model_selector import RAGPipeline

    texts = {"fallback": RAGPipeline._ASK_PERSONA_FALLBACK}

    try:
        import json
        import os

        import psycopg2
        from dotenv import load_dotenv

        load_dotenv()
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=5,
        )
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT prompts_desc FROM proc.bp_prompt "
                "WHERE prompt_type = 'ask_persona' AND prompt_name = 'joshi' "
                "AND COALESCE(prompts_status, 1) = 1 LIMIT 1"
            )
            row = cur.fetchone()
        conn.close()
        if row and row[0]:
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            template = payload.get("prompt_template") or payload.get("template")
            if template:
                texts["bp_prompt"] = str(template)
    except Exception:  # noqa: BLE001 - offline is a skip, not a failure
        pass

    return texts


@pytest.fixture(scope="module")
def personas():
    texts = _persona_texts()
    assert texts["fallback"], "the built-in persona fallback must not be empty"
    return texts


@pytest.mark.parametrize("source", ["fallback", "bp_prompt"])
def test_persona_forbids_adding_different_currencies(personas, source):
    if source not in personas:
        pytest.skip("bp_sqldb not reachable — DB-governed persona not checked")
    text = personas[source].lower()
    assert "different currencies" in text, (
        f"the {source} persona no longer forbids cross-currency addition; "
        "the assistant will resume reporting GBP+USD as a single total"
    )
    # The rule has to say what to do instead, or the model picks its own repair.
    assert "separately" in text, (
        f"the {source} persona forbids summing but does not tell the model to report "
        "each currency separately"
    )


@pytest.mark.parametrize("source", ["fallback", "bp_prompt"])
def test_persona_requires_a_named_basis_for_any_combined_total(personas, source):
    """A converted total is legitimate — but only when its rate is disclosed."""
    if source not in personas:
        pytest.skip("bp_sqldb not reachable — DB-governed persona not checked")
    text = personas[source].lower()
    assert "conversion rate" in text or "converted figure" in text, (
        f"the {source} persona does not describe the one legitimate route to a combined "
        "total (an explicitly converted figure), so the model has no sanctioned alternative"
    )


def test_corpus_facts_spend_never_emits_a_grand_total():
    """The data layer's half of the contract: a split, never a summed total.

    If a well-meaning change ever adds a convenience 'total' row here, the persona rule
    above becomes unenforceable — the model would be quoting a supplied figure.
    """
    import inspect

    from src.services import corpus_facts

    source = inspect.getsource(corpus_facts)
    spend_block = source[source.find("totals_by_currency") :][:1200]

    # A GROUP BY currency is what keeps the totals per-currency.
    assert re.search(r"group\s+by\s+currency", spend_block, re.IGNORECASE), (
        "totals_by_currency no longer groups by currency — it may now return a single "
        "cross-currency total, which is the exact figure the persona is told not to state"
    )
