"""Tool registry.

A Tool is anything the agent can call to affect or observe the outside
world: search the web, do arithmetic, read/write files. Adding a new
capability to the agent means writing one class here and registering
it — no orchestration code changes.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None


class Tool(abc.ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abc.abstractmethod
    def run(self, **kwargs) -> ToolResult:
        ...

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Render this tool as an Anthropic `tools=[...]` entry."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Holds the set of tools available for a given agent run."""

    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def invoke(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, content="", error=f"Unknown tool '{name}'")
        try:
            return tool.run(**tool_input)
        except Exception as exc:  # noqa: BLE001 — tool failures must not crash the agent
            return ToolResult(ok=False, content="", error=f"{type(exc).__name__}: {exc}")
