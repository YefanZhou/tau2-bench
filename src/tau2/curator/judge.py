"""LLM-as-a-judge outcome signal for tau2-bench (write-gate for the curator).

Analogous to ``judge_alfworld.py`` (Fig-13, single success verdict) and ``judge_webshop.py``
(Fig-15, decomposed sub-scores). tau2 already produces a ground-truth DB-state reward, so the
judge is NOT needed for the standard eval — it exists for the SkillCurator write-gating design
(``--reward-source judge``) and for settings where the env reward is unavailable (e.g. RL
rollouts). The env reward always stays the reported accuracy; the judge only decides what the
curator stores.

The tau2 judge decomposes a customer-service outcome the way the domains are actually graded
(see docs/evaluation.md — reward gates on DB writes + required actions + communicated info):

  1. task_completion   — did the agent carry out what the user asked (the right mutating action)?
  2. policy_adherence  — did it follow the domain policy (eligibility checks, confirmations)?
  3. communication     — did it convey the information the user needed?

Public API (mirrors the other judges):
  build_judge_messages(instruction, trajectory_text) -> [system, user]
  parse_judge_output(text) -> {subscores, score, success, rationale}
  judge_success(instruction, trajectory_str, llm_fn, threshold=0.5)
      -> (success: bool, score: float, subscores: dict, rationale: str)
"""

from __future__ import annotations

import re
import json
import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_TAU = """You are an expert judge evaluating whether a customer-service agent correctly handled a user's request in a tool-using dialog simulator. Output a single JSON object and nothing else.
# Task
You are given (1) the user's request and (2) the agent's transcript. In the transcript, [USER] lines are the customer, [AGENT] lines are the agent's messages or its tool calls written as fn(arg=value), and [TOOL] lines are the tool results. Score how well the agent resolved the request.
## How to score
Decompose your evaluation into the following sub-scores, then average them into a single score in [0, 1]:
1. **Task completion**: 1 if the agent performed the action(s) the user actually requested (e.g. the correct booking / cancellation / update was executed via the appropriate tool), 0 otherwise.
2. **Policy adherence**: 1 if the agent followed the required process — verified eligibility/identity where needed and obtained the user's explicit confirmation before any irreversible or mutating action; 0 if it skipped a required step or acted without confirmation.
3. **Communication**: 1 if the agent conveyed the information the user needed (e.g. quoted amounts, confirmed outcomes) clearly and without contradicting the tool results; 0 otherwise.
The final `score` is the mean of the three sub-scores. Define `success` as `score >= 0.5`.
## Strictness
- Award task-completion credit only when a mutating tool call actually succeeded in the transcript; a promise to act, or a tool error, does not count.
- If the transcript is ambiguous about whether the requested action was completed, set task_completion = 0.
- Do not infer policy adherence from the absence of a violation; require positive evidence of the confirmation / eligibility check.
# Output
Output exactly one JSON object and nothing else: { "subscores": { "task_completion": < 0 | 1>, "policy_adherence": < 0 | 1>, "communication": < 0 | 1> }, "score": <float in [0,1], the mean of subscores>, "success": <true|false>, "rationale": "<one or two sentences>" }"""


def build_judge_messages(instruction: str, trajectory_text: str) -> List[Dict[str, str]]:
    """Assemble the judge chat messages (system = JUDGE_SYSTEM_TAU + user turn)."""
    user = (
        f"# User request\n{instruction}\n\n"
        f"# Agent transcript\n{trajectory_text}\n\n"
        f"Now output the JSON object."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_TAU},
        {"role": "user", "content": user},
    ]


def parse_judge_output(text: str, threshold: float = 0.5) -> Dict:
    """Parse the judge reply into {subscores, score, success, rationale}.

    Tolerates ```json fences and surrounding prose. On any failure returns a 0.0 / not-success
    verdict with the raw text as rationale, so a run is never crashed by a malformed judge reply.
    `score` is recomputed here as the mean of subscores (do not trust the model's arithmetic).
    """
    raw = (text or "").strip()
    candidate = raw
    # strip a ```json ... ``` fence if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            candidate = brace.group(0)
    try:
        obj = json.loads(candidate)
    except Exception:
        return {"subscores": {}, "score": 0.0, "success": False,
                "rationale": f"[parse-fail] {raw[:300]}"}

    subs = obj.get("subscores", {}) or {}
    vals = []
    for v in subs.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    score = (sum(vals) / len(vals)) if vals else 0.0
    success = score >= threshold
    rationale = str(obj.get("rationale", "")).strip()
    return {"subscores": subs, "score": round(score, 6),
            "success": success, "rationale": rationale}


def judge_success(instruction: str, trajectory_text: str, llm_fn: Callable[[list], str],
                  threshold: float = 0.5) -> Tuple[bool, float, Dict, str]:
    """Run one judge LLM call and return (success, score, subscores, rationale).

    `llm_fn` takes a chat-message list and returns the model's text reply. Never raises: any
    LLM/parse error yields a not-success verdict (so the curator write-gate fails closed).
    """
    messages = build_judge_messages(instruction, trajectory_text)
    try:
        reply = llm_fn(messages)
    except Exception as e:
        logger.warning(f"tau2 judge LLM error: {e}")
        return False, 0.0, {}, f"[llm-error] {e}"
    parsed = parse_judge_output(reply, threshold=threshold)
    return parsed["success"], parsed["score"], parsed["subscores"], parsed["rationale"]
