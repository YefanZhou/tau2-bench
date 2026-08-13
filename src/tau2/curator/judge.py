"""LLM-as-a-judge outcome signal for tau2-bench (binary write-gate for the curator).

Modeled on ``judge_alfworld.py`` (SkillOS Fig-13, single binary ``success`` verdict) — because
tau2's environment reward is itself strictly binary (0/1 per task, verified over airline/retail/
telecom), a single pass/fail verdict is the faithful shape (WebShop's fractional sub-scores exist
only because WebShop's env reward is fractional; tau2 has no partial credit).

The prompt VOCABULARY is borrowed from tau2's own LLM evaluator
(``src/tau2/evaluator/evaluator_nl_assertions.py``): "a conversation between an agent and a
customer", "expected outcome / satisfies", per-item ``reasoning``, JSON-only. But unlike that
evaluator (which grades a task's GOLD ``nl_assertions`` list), this judge is SELF-CONTAINED — it
sees only (customer request + transcript), no gold — so it can gate memory writes when the env
reward is unavailable (RL rollouts) or when running ``--reward-source judge``. The Fig-13
strictness clauses are adapted to the tool-agent-user setting (credit only tool-confirmed
outcomes; ignore the agent's own claims/promises; ambiguous or partial → failure).

CONTRACT (faithful to the references): the env reward always stays the reported accuracy; this
judge's ``success`` ONLY decides what the curator stores (see the wiring in ``run_curator.py``).

Public API (mirrors judge_alfworld):
  build_judge_messages(instruction, trajectory_text) -> [system, user]
  parse_judge_output(text) -> {"success": bool, "reasoning": str}
  judge_success(instruction, trajectory_text, llm_fn) -> (success: bool, reasoning: str)
"""

from __future__ import annotations

import re
import json
import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------- #
# tau2 judge SYSTEM prompt — binary verdict (ALFWorld Fig-13 structure) in tau2's evaluator
# vocabulary. Single {success, reasoning} object, JSON only. Do NOT add sub-scores: tau2 reward is
# 0/1, so the judge must emit 0/1.
# ---------------------------------------------------------------------------------------------- #
JUDGE_SYSTEM_TAU = """You are an expert judge evaluating whether a customer-service agent successfully satisfied a customer's request. Output a single JSON object and nothing else.
# Task
You will be given (1) the customer's request and (2) the full conversation between the agent and the customer. The conversation contains [USER] lines (the customer), [AGENT] lines (the agent's messages, or its tool calls written as fn(arg=value)), and [TOOL] lines (the tool results). Determine whether the agent fully satisfied the customer's request.
## What "success" means
- The agent must have actually carried out what the customer asked — the correct action (e.g. booking, cancellation, update, refund, troubleshooting fix) must be completed via the appropriate tool call, and the tool result must confirm it succeeded.
- Credit only outcomes that the [TOOL] results confirm. Do not credit effects the agent merely stated, promised, or planned. Ignore the agent's own claims of completion; rely on the tool results.
- If the request required following a process (verifying identity/eligibility, confirming before an irreversible action), that process must be evidenced in the transcript.
## Strictness
- If the transcript is ambiguous about whether the request was fully satisfied, output success=false.
- Partial completion is failure: either the customer's request is fully satisfied or the conversation is a failure.
- A conversation that ends by giving up, escalating to a human, or hitting the step limit without completing the request is a failure.
# Output
Output exactly one JSON object with these fields and nothing else:
{ "success": <true|false>, "reasoning": "<one or two sentences citing the specific tool results that prove success or failure>" }"""


def build_judge_messages(instruction: str, trajectory_text: str) -> List[Dict[str, str]]:
    """Assemble the judge chat messages: JUDGE_SYSTEM_TAU + a user turn with the request+transcript."""
    user = (
        "# Inputs\n"
        "## Customer request\n"
        f"{instruction}\n"
        "## Conversation\n"
        f"{trajectory_text}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_TAU},
        {"role": "user", "content": user},
    ]


def parse_judge_output(text: str) -> Dict:
    """Parse the judge reply into {success, reasoning}. Tolerates ```json fences, surrounding prose,
    and a stray <think>...</think> block. NEVER raises — on any parse failure returns success=False
    with the raw text as reasoning (a failed parse conservatively scores the task a failure), exactly
    like judge_alfworld.parse_judge_output."""
    raw = text or ""
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        logger.warning("tau2 judge: no JSON object in reply; scoring FAIL. reply=%r", raw[:200])
        return {"success": False, "reasoning": f"[parse-fail] {raw[:300]}"}
    try:
        obj = json.loads(candidate)
    except Exception as e:
        logger.warning("tau2 judge: JSON parse error (%s); scoring FAIL. candidate=%r", e, candidate[:200])
        return {"success": False, "reasoning": f"[parse-fail] {candidate[:300]}"}
    # normalize success (accept bool, or "true"/"false"/1/0/"yes")
    sv = obj.get("success", False)
    if isinstance(sv, str):
        success = sv.strip().lower() in ("true", "1", "yes")
    else:
        success = bool(sv)
    return {"success": success, "reasoning": str(obj.get("reasoning", ""))}


def judge_success(instruction: str, trajectory_text: str,
                  llm_fn: Callable[[list], str]) -> Tuple[bool, str]:
    """Run one judge LLM call and return (success, reasoning).

    `llm_fn` takes a chat-message list and returns the model's text reply. Never raises: any
    LLM/parse error yields a not-success verdict (so the curator write-gate fails closed).
    """
    messages = build_judge_messages(instruction, trajectory_text)
    try:
        reply = llm_fn(messages)
    except Exception as e:
        logger.warning(f"tau2 judge LLM error: {e}")
        return False, f"[llm-error] {e}"
    parsed = parse_judge_output(reply)
    return parsed["success"], parsed["reasoning"]
