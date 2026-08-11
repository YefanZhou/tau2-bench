"""ReasoningBank memory method for tau2-bench.

Faithful port of ``reasoningbank_alfworld_api.py::ReasoningBankAlfworld``, adapted to tau2:
  * WRITE runs an LLM on EVERY episode (success AND failure) to distill <=3 markdown memory
    items — a success-reflection prompt for wins, a failure-reflection prompt for losses.
  * READ is BM25 top-k over stored task texts, then CONCATENATE the retrieved memory items
    (NO read-time LLM — the opposite of the curator, which synthesizes a briefing on read).

This is the ReasoningBank baseline for the curator comparison. Same public seam as
``CuratorTau`` (``add(task_id, task, trajectory, reward)`` / ``retrieve(query, n) -> str``)
so the runner wiring is identical. BM25 = the vendored dependency-free ``BM25Okapi`` (langchain
is not a tau2 dep). Curation LLM via litellm (OPENAI_API_KEY/OPENAI_BASE_URL), CURATION_* env
knobs, gpt-5.x temperature fallback, gateway X-Api-Key.

The two reflection system prompts are the domain-generalized versions of the ALFWorld ones
(``SUCCESSFUL_SI`` / ``FAILED_SI``) — reworded from "household task planning" to tau2's
tool-agent-user customer-service setting, same <=3-item markdown output contract.
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, List, Optional

from .bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _cur_float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _cur_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


_CUR_TEMP = _cur_float("CURATION_TEMPERATURE", 0.7)
_CUR_TOP_P = _cur_float("CURATION_TOP_P", None)
_CUR_MAX_TOK = _cur_int("CURATION_MAX_TOKENS", 1024)
_X_API_KEY = os.environ.get("X_API_KEY") or None


def _completion_with_temp_fallback(**kwargs):
    """litellm.completion with a one-shot retry that drops temperature/top_p if the model
    rejects a non-default value (gpt-5.x). Other models never raise, so unaffected."""
    from litellm import completion
    try:
        return completion(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        retried = dict(kwargs)
        changed = False
        if "temperature" in msg and ("does not support" in msg or "only the default" in msg or "unsupported value" in msg):
            retried.pop("temperature", None); changed = True
        if "top_p" in msg and ("does not support" in msg or "unsupported value" in msg):
            retried.pop("top_p", None); changed = True
        if not changed:
            raise
        return completion(**retried)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# Domain-generalized reflection prompts (tau2 customer-service). Same <=3-item markdown
# contract as the ALFWorld ReasoningBank (SUCCESSFUL_SI / FAILED_SI).
SUCCESSFUL_SI = """You are an expert in customer-service task handling. You will be given a customer request and a transcript showing how an agent successfully resolved it by talking to the user and calling tools. In the transcript, [USER] lines are the customer, [AGENT] lines are the agent's messages or tool calls written as fn(arg=value), and [TOOL] lines are tool results.

## Guidelines
Extract and summarize useful insights as memory items that would help an agent resolve similar customer requests in the future while following the domain policy.

## Important notes
  - Think about why the trajectory succeeded, then summarize the insights.
  - Extract *at most 3* memory items.
  - Do not repeat similar or overlapping items.
  - Focus on generalizable strategies (e.g., which tools to call and in what order, what to confirm with the user, which policy conditions to check), not specific IDs, prices, or dates.

## Output Format
Your output must strictly follow this Markdown format:

```
# Memory Item i
## Title <short title>
## Description <one sentence summary>
## Content <1-3 sentences of actionable insight>
```
"""

FAILED_SI = """You are an expert in customer-service task handling. You will be given a customer request and a transcript showing how an agent attempted but failed to resolve it. In the transcript, [USER] lines are the customer, [AGENT] lines are the agent's messages or tool calls written as fn(arg=value), and [TOOL] lines are tool results.

## Guidelines
Extract lessons learned as memory items to help avoid the same mistakes on similar future requests.

## Important notes
  - Reflect on why the trajectory failed, then summarize what to do differently.
  - Extract *at most 3* memory items.
  - Do not repeat similar or overlapping items.
  - Focus on generalizable strategies (e.g., a missing confirmation, a skipped eligibility check, a wrong or out-of-order tool call), not specific IDs, prices, or dates.

## Output Format
Your output must strictly follow this Markdown format:

```
# Memory Item i
## Title <short title>
## Description <one sentence summary>
## Content <1-3 sentences of actionable insight>
```
"""


class ReasoningBankTau:
    def __init__(
        self,
        storage_path: str = "./memory/reasoning_bank.jsonl",
        retrieve_num: int = 3,
        curation_model_name: Optional[str] = None,
        curation_base_url: Optional[str] = None,
        is_gateway: bool = False,
    ):
        self.storage_path = storage_path
        self.retrieve_num = retrieve_num
        self.curation_model_name = curation_model_name
        self.curation_base_url = curation_base_url
        self.is_gateway = is_gateway

        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        self.memory_bank: List[Dict] = self._load_jsonl(self.storage_path)
        self._bm25: Optional[BM25Okapi] = None
        self._rebuild_bm25()

    def _load_jsonl(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _rebuild_bm25(self):
        if not self.memory_bank:
            self._bm25 = None
            return
        toks = [_tokenize(r["query"]) for r in self.memory_bank]
        self._bm25 = BM25Okapi([t if t else ["__empty__"] for t in toks])

    def _save_record(self, record: Dict):
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _llm(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.curation_model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=_CUR_TEMP,
            max_completion_tokens=_CUR_MAX_TOK,
            num_retries=10,
        )
        if self.curation_base_url:
            kwargs["base_url"] = self.curation_base_url
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        if _CUR_TOP_P is not None:
            kwargs["top_p"] = _CUR_TOP_P
        if self.is_gateway and _X_API_KEY:
            kwargs["extra_headers"] = {"X-Api-Key": _X_API_KEY}
        resp = _completion_with_temp_fallback(**kwargs)
        return resp.choices[0].message.content or ""

    def _extract_memory_items(self, task: str, trajectory: str, success: bool) -> List[str]:
        si = SUCCESSFUL_SI if success else FAILED_SI
        raw = self._llm(si, f"**Request:** {task}\n\n**Transcript:**\n{trajectory}")
        raw = raw.split("</think>")[-1]  # ignore any <think> block
        raw = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", raw, flags=re.MULTILINE)
        items = [
            block.strip().rstrip("`").strip()
            for block in raw.split("# Memory Item")
            if block.strip().strip("`").strip()
        ]
        return ["# Memory Item " + item for item in items]

    # ------------------------------------------------------------------ #
    # Public API (matches CuratorTau)                                      #
    # ------------------------------------------------------------------ #

    def add(self, task_id: str, task: str, trajectory: str, reward) -> None:
        """Distill <=3 memory items from a completed episode (LLM on every add) and store."""
        success = float(reward) > 0
        memory_items = self._extract_memory_items(task, trajectory, success=success)
        record = {
            "task_id": task_id,
            "query": task,
            "memory_items": memory_items,
            "status": "success" if success else "fail",
        }
        self._save_record(record)
        self.memory_bank.append(record)
        self._rebuild_bm25()
        logger.info(f"ReasoningBank: indexed {record['status']} ({len(memory_items)} items) for {task_id}")

    def retrieve(self, query: str, n: int = None, curator_question: str = None) -> str:
        """BM25 top-k over stored task texts, concatenate the retrieved memory items (no LLM)."""
        n = n or self.retrieve_num
        if self._bm25 is None:
            return ""
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        parts = []
        for idx in order:
            items = self.memory_bank[idx].get("memory_items", [])
            if items:
                parts.append("\n\n".join(items))
        return "\n\n---\n\n".join(parts) if parts else ""
