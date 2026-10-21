"""The critic's model call: no tools, no output gate, same audited egress.

`BaseAgent.reason()` is wrong for this agent, in two measured ways:

  * It prepends AgentNick's controller prompt, which says "You do not answer
    from memory. You establish facts by calling tools", and offers every
    registered tool. The critic's own governed prompt says "Return JSON only",
    and its evidence is assembled before it is called. Live on 2026-09-12 the
    model obeyed the preamble: six rounds of tool calls, no final answer.
  * Every answer returning from `run_tools` passes through `_gate`, which is
    `services.output_safety`. That guard is right for prose shown to a person
    -- it stops an answer narrating the backend -- but a critique is a machine
    record. Measured against a realistic critique: `is_safe()` is False,
    flagging `SUPPLIER_NOT_IN_CONTRACT_MASTER` as an env var and "repository"
    as a mechanism. A correct critique would be sent back to be re-framed and,
    failing twice, replaced by a canned reply that parses as nothing.

So the critic talks to the model directly. What it does NOT skip is
`egress.post`: the model call stays audited, with the same purpose every other
inference carries. The endpoint, model and timeout come from tool_runtime so
there is one source of truth for where the model lives.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.services import agentnick_driver
from src.services.tool_runtime import _DEFAULT_MODEL, _DEFAULT_TIMEOUT_S

logger = logging.getLogger(__name__)


def ask_for_critique(
    system: str,
    task: str,
    *,
    model: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(answer, error)``. Never raises.

    One round: the model is given its instructions and the candidate, and must
    answer. No tools are offered, so there is nothing for it to call instead.
    """
    # No tools are passed, deliberately: see the module docstring.
    try:
        proposal = agentnick_driver.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": task}], None,
            model=model or _DEFAULT_MODEL, timeout=timeout_s or _DEFAULT_TIMEOUT_S)
        if proposal.refused:
            return None, "model call was refused by the egress guard"
        content = proposal.content
    except Exception as exc:  # noqa: BLE001 - a dead model is an error, not a crash
        logger.error("critic model call failed: %s", exc)
        return None, str(exc)[:300]

    if not content:
        return None, "the model returned an empty answer"
    return content, None
