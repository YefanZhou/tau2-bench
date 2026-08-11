"""Render a tau2 ``SimulationRun`` (or its message list) as a compact text trajectory.

Produces ``[AGENT]/[USER]/[TOOL]`` lines with tool calls in functional notation — the
format the Memory Curator LLM reads (and that gets stored in the memory jsonl). Ported from
``sea-mem-policy/curator/tau_server/runner.py::_format_trajectory``; the tau2 message types
and ``to_functional_format`` helper are verified present at commit 668d3bc.
"""

from __future__ import annotations

from typing import List, Optional

_TOOL_CONTENT_CAP = 500  # truncate long tool outputs (DB dumps) — keep the briefing readable


def messages_to_text(messages: Optional[list]) -> str:
    """Format a tau2 message list as ``[AGENT]/[USER]/[TOOL]`` lines.

    Imports the tau2 data-model types lazily so this module is import-safe outside a tau2
    process (e.g. unit tests that stub the store).
    """
    from tau2.data_model.message import (
        AssistantMessage,
        UserMessage,
        ToolMessage,
        MultiToolMessage,
    )
    from tau2.utils.tools import to_functional_format

    lines: List[str] = []
    for msg in messages or []:
        if isinstance(msg, AssistantMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"[AGENT] {to_functional_format(tc)}")
            elif msg.content:
                lines.append(f"[AGENT] {msg.content}")
        elif isinstance(msg, UserMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"[USER] {to_functional_format(tc)}")
            elif msg.content:
                lines.append(f"[USER] {msg.content}")
        elif isinstance(msg, ToolMessage):
            content = msg.content or ""
            if len(content) > _TOOL_CONTENT_CAP:
                content = content[:_TOOL_CONTENT_CAP] + "..."
            lines.append(f"[TOOL] {content}")
        elif isinstance(msg, MultiToolMessage):
            for tm in msg.tool_messages:
                content = tm.content or ""
                if len(content) > _TOOL_CONTENT_CAP:
                    content = content[:_TOOL_CONTENT_CAP] + "..."
                lines.append(f"[TOOL] {content}")
    return "\n".join(lines)


def simulation_to_text(simulation) -> str:
    """Format a ``SimulationRun`` (uses ``simulation.messages``, half-duplex text runs)."""
    return messages_to_text(getattr(simulation, "messages", None))
