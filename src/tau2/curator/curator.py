"""``CuratorTau`` — the read-time memory curator for tau2-bench.

Faithful port of ``curator_alfworld_v1_api.py::CuratorAlfworld`` with two tau2 adaptations:
  * BM25 retrieval uses ``rank_bm25.BM25Okapi`` directly (langchain is not a tau2 dep);
  * trajectories are tau2 ``[AGENT]/[USER]/[TOOL]`` text (see ``trajectory.py``).

Semantics preserved from the SkillCurator original:
  * Q2 — empty store => ``retrieve`` returns "" with NO curator LLM call.
  * Q3 — the retrieval / curator key is the natural-language task text (here
    ``str(task.user_scenario)``), not an id.
  * WRITE is append-only, no LLM; ``curation_mode`` gates whether failures are stored and
    whether retrieved memories are marked ``Result: Success/Failure``.
  * READ = BM25 top-k -> ``build_curator_messages`` -> curator LLM -> ``_strip_think`` ->
    inject. Every read is logged to a sibling ``curator_calls.jsonl`` with full provenance.

Curation sampling is controlled by the same ``CURATION_*`` env knobs as the SkillCurator
runners so behavior is comparable across codebases:
  ``CURATION_TEMPERATURE`` (0.7), ``CURATION_TOP_P``, ``CURATION_MAX_TOKENS`` (1024),
  ``CURATION_MODEL`` (curator LLM; defaults to the agent model passed in), plus the standard
  ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` litellm env. ``X_API_KEY`` is attached as the
  Salesforce-gateway ``X-Api-Key`` header when set. gpt-5.x temperature rejection is handled
  by a one-shot fallback (retry without temperature), matching the ALFWorld curator.
"""

from __future__ import annotations

import os
import re
import math
import json
import logging
from typing import Dict, List, Optional

from .bm25 import BM25Okapi

from .prompts import (
    _MODE_SYSTEM_PROMPT,
    _STORE_FAILURES_MODES,
    _MARK_RESULTS_MODES,
    CURATION_MODES,
    CURATOR_SYSTEM_SUCCESS_ONLY,
)
from .trajectory import messages_to_text

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Curation sampling knobs (env-overridable; defaults = ALFWorld curator's). #
# ------------------------------------------------------------------ #
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
_LOG_CALLS = os.environ.get("CURATOR_LOG_CALLS", "1").lower() in ("1", "true", "yes")


def _completion_with_temp_fallback(**kwargs):
    """``litellm.completion`` with a one-shot retry that drops ``temperature`` (and ``top_p``)
    if the model rejects a non-default value. gpt-5 / gpt-5.x reasoning models raise
    'temperature does not support X; only the default (1) value is supported'; without this a
    gpt-5.x curator would crash and leave the store empty. Other models never raise it."""
    from litellm import completion
    try:
        return completion(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        retried = dict(kwargs)
        changed = False
        if "temperature" in msg and (
            "does not support" in msg or "only the default" in msg or "unsupported value" in msg
        ):
            retried.pop("temperature", None)
            changed = True
        if "top_p" in msg and ("does not support" in msg or "unsupported value" in msg):
            retried.pop("top_p", None)
            changed = True
        if not changed:
            raise
        return completion(**retried)


def build_curator_messages(
    query: str, retrieved_text: str, curation_mode: str = "success_only"
) -> List[Dict[str, str]]:
    """Isolated, module-level prompt builder for the read-time curation call.

    Kept module-level (not a method) so a future RL rollout generator can import the
    *identical* prompt construction used at eval time — the training-awareness contract.
    """
    system = _MODE_SYSTEM_PROMPT.get(curation_mode, CURATOR_SYSTEM_SUCCESS_ONLY)
    if retrieved_text:
        user_content = f"Request: {query}\n\nRetrieved Memories:\n{retrieved_text}"
    else:
        user_content = f"Request: {query}\n\nNo past memories available yet."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _record_success(record: Dict) -> bool:
    """Whether a stored record is a success — numeric ``reward`` (>0) is source of truth,
    falling back to the ``status`` string for records written before reward existed."""
    if "reward" in record and record["reward"] is not None:
        return float(record["reward"]) > 0
    return record.get("status", "success") == "success"


def _strip_think(text: str) -> str:
    """Named parser hook: drop ``<think>...</think>`` so curator CoT never enters memory.
    A later format/reward check can attach here (training-awareness contract)."""
    if not text:
        return ""
    text = text.split("</think>")[-1]
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """Lowercased alnum tokenizer for BM25 (rank_bm25 needs pre-tokenized docs/queries)."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class CuratorTau:
    """Read-time BM25 + LLM-briefing memory curator for tau2-bench.

    Public API mirrors the ALFWorld/WebShop curator (``add`` / ``retrieve``) so the tau2
    runner wiring is a thin copy.
    """

    def __init__(
        self,
        storage_path: str = "./memory/curator_tau_memory.jsonl",
        retrieve_num: int = 3,
        curation_model_name: Optional[str] = None,
        curation_base_url: Optional[str] = None,
        curator_on_empty: bool = False,
        curation_mode: str = "success_only",
        is_gateway: bool = False,
    ):
        self.storage_path = storage_path
        self.retrieve_num = retrieve_num
        self.curation_model_name = curation_model_name
        self.curation_base_url = curation_base_url
        self.curator_on_empty = curator_on_empty
        self.is_gateway = is_gateway

        assert curation_mode in CURATION_MODES, (
            f"curation_mode must be one of {CURATION_MODES}"
        )
        self.curation_mode = curation_mode
        self._store_failures = curation_mode in _STORE_FAILURES_MODES
        self._mark_results = curation_mode in _MARK_RESULTS_MODES

        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        self.calls_log_path = os.path.join(
            os.path.dirname(os.path.abspath(self.storage_path)), "curator_calls.jsonl"
        )

        self.memory_bank: List[Dict] = self._load_jsonl(self.storage_path)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []
        self._rebuild_bm25()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_jsonl(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _rebuild_bm25(self):
        if not self.memory_bank:
            self._bm25 = None
            self._corpus_tokens = []
            return
        self._corpus_tokens = [_tokenize(r["query"]) for r in self.memory_bank]
        # rank_bm25 raises on an all-empty corpus; guard with a sentinel token.
        safe = [t if t else ["__empty__"] for t in self._corpus_tokens]
        self._bm25 = BM25Okapi(safe)

    def _save_record(self, record: Dict):
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _llm_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Run the curation LLM via litellm (OpenAI-compatible / Salesforce gateway).

        Reads OPENAI_API_KEY / OPENAI_BASE_URL from env like the rest of tau2. Uses
        ``max_completion_tokens`` (gpt-5.x rejects ``max_tokens``) and the temp-fallback wrapper.
        """
        kwargs = dict(
            model=self.curation_model_name,
            messages=messages,
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

    def _format_case(self, idx: int, record: Dict) -> str:
        """Render a retrieved record as a numbered case for the curator's context.
        In the ``_and_fail`` family each memory carries a ``Result: Success/Failure`` line."""
        lines = [f"Memory {idx}:"]
        if self._mark_results:
            lines.append(f"Result: {'Success' if _record_success(record) else 'Failure'}")
        lines.append(f"Request: {record.get('query', '')}")
        lines.append(f"Trajectory:\n{record.get('trajectory', '')}")
        return "\n".join(lines)

    def _log_call(self, query, retrieved_text, briefing, messages=None,
                  retrieved=None, briefing_raw=None):
        if not _LOG_CALLS:
            return
        try:
            rec = {
                "query": query,
                "retrieved": retrieved if retrieved is not None else [],
                "retrieved_text": retrieved_text,
                "messages": messages if messages is not None else build_curator_messages(
                    query, retrieved_text, curation_mode=self.curation_mode),
                "curation_mode": self.curation_mode,
                "model": self.curation_model_name,
                "briefing": briefing,
                "briefing_raw": briefing_raw if briefing_raw is not None else briefing,
            }
            with open(self.calls_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # logging must never break a run
            logger.warning(f"Failed to log curator call: {e}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(self, task_id: str, task: str, trajectory: str, reward) -> None:
        """Store a raw episode trajectory (curator write — no LLM, append-only).

        Args:
            task_id: unique task identifier (tau2 ``task.id``).
            task: the NL task text — the BM25 retrieval key (``str(task.user_scenario)``).
            trajectory: the rendered ``[AGENT]/[USER]/[TOOL]`` text (see ``trajectory.py``).
            reward: the write-gating reward. success = reward > 0. Stored as float.

        success_only (default): store wins only. success_and_fail: store both, tagging
        ``reward``/``status`` so the curator prompt can distinguish them at read time.
        """
        reward = float(reward)
        success = reward > 0
        if not success and not self._store_failures:
            logger.info(f"Curator: skipping failed trajectory (not stored): {task_id}")
            return
        record = {
            "task_id": task_id,
            "query": task,
            "trajectory": trajectory,
            "reward": reward,
            "status": "success" if success else "fail",
        }
        self._save_record(record)
        self.memory_bank.append(record)
        self._rebuild_bm25()
        logger.info(
            f"Curator: stored {record['status']} (reward={reward}) trajectory for: {task_id}"
        )

    def retrieve(self, query: str, n: int = None, curator_question: str = None) -> str:
        """Retrieve similar past episodes and curate them into a briefing (curator read).

        Args:
            query: the BM25 retrieval key (the NL task; also what was stored).
            n: number of records to retrieve (defaults to ``self.retrieve_num``).
            curator_question: text shown to the curator LLM as ``Request: {..}``; defaults
                to ``query``. Lets an enriched current-task string be shown without changing
                the BM25 key or the stored records.

        Returns the synthesized briefing, or "" if the store is empty (Q2: no LLM call).
        """
        n = n or self.retrieve_num
        curator_question = curator_question if curator_question is not None else query

        retrieved_text = ""
        retrieved_meta: List[Dict] = []
        if self._bm25 is not None:
            q_tokens = _tokenize(query)
            scores = self._bm25.get_scores(q_tokens)
            # top-n store indices by BM25 score (descending).
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
            parts = []
            for rank, store_idx in enumerate(order, 1):
                record = self.memory_bank[store_idx]
                parts.append(self._format_case(rank, record))
                retrieved_meta.append({
                    "store_index": store_idx,
                    "score": round(float(scores[store_idx]), 6),
                    "rank": rank,
                    "question": record.get("query", ""),
                    "status": record.get("status", "success"),
                })
            retrieved_text = "\n\n".join(parts)

        if not retrieved_text and not self.curator_on_empty:
            return ""

        messages = build_curator_messages(
            curator_question, retrieved_text, curation_mode=self.curation_mode
        )
        briefing_raw = self._llm_from_messages(messages)
        briefing = _strip_think(briefing_raw)
        self._log_call(curator_question, retrieved_text, briefing,
                       messages=messages, retrieved=retrieved_meta, briefing_raw=briefing_raw)
        return briefing if briefing else ""
