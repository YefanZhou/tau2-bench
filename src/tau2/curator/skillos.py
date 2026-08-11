"""SkillOS memory method for tau2-bench.

Port of the SkillOS skill-memory method (``evaluation/agent_eval/SkillOS/skills_memory.py``
+ its curation loop in ``run_unified_dev_async_curator_api.py``), adapted to tau2 and made
self-contained (no numpy/sklearn/embeddings — the original's embedding path isn't in the tau2
env; the in-repo runner retrieves via ``search_method='bm25'`` anyway).

Method:
  * The store is a list of SKILLS, each ``{title, content}`` where content is markdown with a
    YAML frontmatter (``name`` + ``description``).
  * WRITE (per episode): show the curator LLM the task, the retrieved past skills, the
    trajectory, and the Success/Failure result, and let it CALL TOOLS to
    ``new_skill_insert`` / ``skill_update`` / ``skill_delete`` — i.e. it maintains an
    evolving skill library (the defining SkillOS behavior vs ReasoningBank's append-only items).
  * READ: BM25 top-k over each skill's title + YAML description (NOT the full body — matches
    the in-repo SkillOS retrieval-precision fix), return the formatted skill blocks.

Same public seam as ``CuratorTau`` (``add(task_id, task, trajectory, reward)`` /
``retrieve(query, n) -> str``). Curation LLM via litellm native tool-calling; CURATION_* env
knobs; gpt-5.x temperature fallback; gateway X-Api-Key. The system prompt is the SkillOS
"skills curator" prompt (kept generic — it already reads as domain-agnostic agent-task text).
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple

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


# SkillOS "skills curator" system prompt (verbatim structure from skills_memory.render_system_prompt
# status='memorie'; already domain-generic — "agent tasks", not ALFWorld-specific).
SKILLOS_SYSTEM = """# Role
You are an expert with a sophisticated skills curator. Our overall goal is to accomplish agent tasks. Your primary task is to convert past experiences of agent task execution into reusable, general skills, so that they can benefit and inspire future tasks.

# Input Data
1. **Task Description**: The task to be accomplished.
2. **Past Skills**: A list of previously stored relevant skills, each with a skill name (identifier) and content.
3. **Agent Trajectory**: The step-by-step execution trace of a customer-service agent talking to a user and calling tools.
4. **Result**: Whether the agent successfully completed the task or not.

# Critical Constraints:
- **Skill Format**: Extract and store important information as skills using following Markdown format **strictly**.
- **No Specifics**: Avoid problem-specific details. Remove specific IDs, prices, dates, and names. Replace with variables/concepts.
- **No Hallucination**: Do not invent facts.
- Each skill must be **Atomic, modular, and reusable**.
- Never store anything that contradicts the domain policy.

# Skill Markdown Format and Content Instructions:
- **YAML Frontmatter (MANDATORY)**: Each skill MUST start with a YAML frontmatter block delimited by `---`. The YAML block MUST contain exactly two keys: `name` and `description`.
    - **Example Structure**:
    ---
    name: <Human-readable skill name>
    description: <One-sentence what/when/why/how summary, concise and actionable, this will be used for future references>
    ---
- **Markdown Body**: Immediately after the second `---`, provide instructions using Markdown headings.
    - Suggested sections: `# Workflow`, `# When NOT to use`, `# Prerequisite Constraints`. Use what's appropriate for clarity.
    - Ensure the content is atomic, general, and devoid of specific instance IDs.

# Action Guidelines
1. Analyze the agent trajectory and its result. Identify what went well and what didn't.
2. If the trajectory is correct, extract reusable knowledge or skills. If the trajectory is incorrect, identify the failure point and extract skills that can help fix the issue.
3. Compare the extracted skills with past skills. Determine whether to **insert a new skill**, **update an existing skill**, or **delete an existing skill** using the tools provided.
"""

MEMORY_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "new_skill_insert",
        "description": "If there is no existing relevant skill, create new skill with desired skill name and content.",
        "parameters": {"type": "object", "properties": {
            "skill_name": {"type": "string", "description": "The name of the new skill to create."},
            "content": {"type": "string", "description": "The markdown content for the new skill."},
        }, "required": ["skill_name", "content"]}}},
    {"type": "function", "function": {
        "name": "skill_update",
        "description": "If the existing skill can be improved, update the specific skill by its skill_name.",
        "parameters": {"type": "object", "properties": {
            "skill_name": {"type": "string", "description": "The name of the skill to update. Must exactly match an existing skill title."},
            "new_name": {"type": "string", "description": "The new skill name (optional)."},
            "new_content": {"type": "string", "description": "The new full content for the skill (optional)."},
        }, "required": ["skill_name"]}}},
    {"type": "function", "function": {
        "name": "skill_delete",
        "description": "Delete an existing skill by its title.",
        "parameters": {"type": "object", "properties": {
            "skill_name": {"type": "string", "description": "The name of the skill to delete."},
        }, "required": ["skill_name"]}}},
]


def _retrieval_doc(skill: Dict) -> str:
    """BM25 index text = title + YAML `description` only (SkillOS retrieval-precision fix:
    indexing the full body caused wrong-skill injection). Falls back to title+content."""
    title = skill.get("title", "")
    content = skill.get("content", "") or ""
    m = re.search(r"description:\s*(.+)", content)
    desc = m.group(1).strip() if m else content
    return f"{title} {desc}"


class SkillOSTau:
    def __init__(
        self,
        storage_path: str = "./memory/skillos_skills.json",
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
        self.skills: List[Dict[str, str]] = self._load_skills()

    # ------------------------------------------------------------------ #
    # Persistence + skill CRUD                                             #
    # ------------------------------------------------------------------ #
    def _load_skills(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.storage_path):
            return []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_skills(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, indent=2, ensure_ascii=False)

    def _skill_index(self, title: str) -> int:
        for i, s in enumerate(self.skills):
            if s.get("title") == title:
                return i
        return -1

    def _insert(self, skill_name: str, content: str) -> str:
        if self._skill_index(skill_name) >= 0:
            # de-dup: treat as an update to the existing skill
            self.skills[self._skill_index(skill_name)]["content"] = content
        else:
            self.skills.append({"title": skill_name, "content": content})
        return skill_name

    def _update(self, title: str, new_name: str = None, new_content: str = None):
        i = self._skill_index(title)
        if i < 0:
            raise ValueError(f"skill not found: {title}")
        if new_name:
            self.skills[i]["title"] = new_name
        if new_content:
            self.skills[i]["content"] = new_content

    def _delete(self, title: str):
        i = self._skill_index(title)
        if i >= 0:
            self.skills.pop(i)

    def _apply_tool_call(self, name: str, args: dict):
        if name == "new_skill_insert":
            self._insert(args["skill_name"], args.get("content", ""))
        elif name == "skill_update":
            self._update(args["skill_name"], args.get("new_name"), args.get("new_content"))
        elif name == "skill_delete":
            self._delete(args["skill_name"])

    # ------------------------------------------------------------------ #
    # Retrieval (BM25 over title + description)                            #
    # ------------------------------------------------------------------ #
    def _retrieve_skills(self, query: str, n: int) -> List[Dict]:
        if not self.skills:
            return []
        toks = [_tokenize(_retrieval_doc(s)) for s in self.skills]
        bm25 = BM25Okapi([t if t else ["__empty__"] for t in toks])
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.skills[i] for i in order]

    @staticmethod
    def _format_skills(skills: List[Dict]) -> str:
        parts = []
        for i, s in enumerate(skills):
            parts.append(f"**Skill {i + 1}: {s.get('title')}**\n{s.get('content')}")
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Curation LLM (native tool-calling)                                   #
    # ------------------------------------------------------------------ #
    def _curate(self, task: str, trajectory: str, reward: bool, retrieved_text: str):
        user_content = f"""# Task Context
## Task Description:
```
{task}
```

## Past Skills:
```
{retrieved_text if retrieved_text else "(none)"}
```

## Agent Trajectory:
```
{trajectory}
```

## Result:
```
{"Success" if reward else "Failure"}
```

# Output Format:
Your output must contain the following sections:
- Analysis: Analyze the trajectory, associated skills, and the final result. Identify what went well and what didn't.
- Tool Calls: Based on your analysis, determine whether to insert a new skill, update an existing skill, or delete an existing skill.
"""
        kwargs = dict(
            model=self.curation_model_name,
            messages=[{"role": "system", "content": SKILLOS_SYSTEM},
                      {"role": "user", "content": user_content}],
            temperature=_CUR_TEMP,
            max_completion_tokens=_CUR_MAX_TOK,
            num_retries=10,
            tools=MEMORY_TOOL_SCHEMAS,
            tool_choice="auto",
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
        msg = resp.choices[0].message
        return getattr(msg, "tool_calls", None) or []

    # ------------------------------------------------------------------ #
    # Public API (matches CuratorTau)                                      #
    # ------------------------------------------------------------------ #
    def add(self, task_id: str, task: str, trajectory: str, reward) -> None:
        """Curate the skill library from a completed episode via tool-calling (insert/update/delete)."""
        success = float(reward) > 0
        retrieved = self._retrieve_skills(task, self.retrieve_num)
        retrieved_text = self._format_skills(retrieved)
        try:
            tool_calls = self._curate(task, trajectory, success, retrieved_text)
        except Exception as e:
            logger.warning(f"SkillOS curation failed for {task_id}: {e}")
            return
        n_applied = 0
        for tc in tool_calls:
            try:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                self._apply_tool_call(name, args)
                n_applied += 1
            except Exception as e:
                logger.warning(f"SkillOS tool-call apply failed ({name}): {e}")
        if n_applied:
            self._save_skills()
        logger.info(f"SkillOS: applied {n_applied} skill ops for {task_id} "
                    f"({'success' if success else 'fail'}); library size={len(self.skills)}")

    def retrieve(self, query: str, n: int = None, curator_question: str = None) -> str:
        """BM25 top-k over stored skills' title+description; return formatted skill blocks (no LLM)."""
        n = n or self.retrieve_num
        skills = self._retrieve_skills(query, n)
        return self._format_skills(skills) if skills else ""
