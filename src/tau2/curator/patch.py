"""Monkey-patch for tau2-bench ``LLMAgent`` — adds a ``_system_prompt_suffix`` field.

Rewrites the ``system_prompt`` property so external code (the curator) can append a
memory briefing AFTER the ``</policy>`` section without mutating ``domain_policy`` — keeping
the domain policy section byte-identical to a no-memory run.

Verified against the vendored LLMAgent at commit 668d3bc:
  * ``SYSTEM_PROMPT`` / ``AGENT_INSTRUCTION`` are still module-level constants in
    ``tau2.agent.llm_agent``;
  * ``system_prompt`` is a ``@property`` that returns
    ``SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)``.

The patch reproduces that exact format string and only appends the suffix when set, so an
agent with no suffix behaves identically to the unpatched agent (zero-diff on the no-memory
path). Idempotent; call once at process startup before building agents.
"""

_PATCHED = False


def apply():
    """Apply the ``_system_prompt_suffix`` monkey-patch (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from tau2.agent.llm_agent import LLMAgent, SYSTEM_PROMPT, AGENT_INSTRUCTION

    @property
    def _patched_system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        suffix = getattr(self, "_system_prompt_suffix", "")
        if suffix:
            return base + "\n" + suffix
        return base

    LLMAgent.system_prompt = _patched_system_prompt
