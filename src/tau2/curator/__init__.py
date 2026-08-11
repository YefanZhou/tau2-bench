"""MemCurator (curator_v1) for tau2-bench.

A read-time memory curator ported from SkillCurator's
``evaluation/agent_eval/curator_alfworld_v1_api.py`` and adapted to tau2-bench's
three-party (agent / user-simulator / environment) simulation loop.

Design (identical philosophy to the ALFWorld/WebShop curator_v1):
  * WRITE is trivial — store raw episode trajectories append-only with their numeric
    ``reward`` and a derived ``status``. No LLM call, no update/delete. ``curation_mode``
    controls what is stored: success_only (default) keeps wins; success_and_fail keeps
    both and marks each retrieved memory ``Result: Success/Failure`` at read time.
  * READ is the smart part — BM25-retrieve the top-k most similar past episodes, then call
    a "Memory Curator" LLM to synthesize them into one concise, actionable briefing, and
    inject that briefing into the agent's system prompt (after </policy>).

The only tau2-specific pieces are:
  * the retrieval key = ``str(task.user_scenario)`` (the NL customer scenario);
  * the trajectory renderer = ``[AGENT]/[USER]/[TOOL]`` lines (see ``trajectory.py``);
  * BM25 runs on ``rank_bm25.BM25Okapi`` directly (langchain is NOT a tau2 dependency,
    but ``rank-bm25`` is).

Public entry points:
  * ``CuratorTau`` — the curator object (``add`` / ``retrieve``).
  * ``build_curator_messages`` — module-level prompt builder (training-awareness contract).
  * ``apply_system_prompt_patch`` — enables per-instance ``_system_prompt_suffix`` injection.
"""

from .curator import CuratorTau, build_curator_messages, CURATION_MODES
from .patch import apply as apply_system_prompt_patch
from .trajectory import simulation_to_text
from . import judge

__all__ = [
    "CuratorTau",
    "build_curator_messages",
    "CURATION_MODES",
    "apply_system_prompt_patch",
    "simulation_to_text",
    "judge",
]
