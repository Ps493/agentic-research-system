"""Shared state carried across the whole plan -> act -> observe -> respond run.

This is what turns a sequence of isolated LLM calls into a coherent
*workflow*: every step can see what earlier steps found, and the final
synthesis step sees everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ToolTrace:
    tool_name: str
    tool_input: dict
    result_summary: str


@dataclass
class StepRecord:
    step_id: int
    description: str
    tool_traces: list[ToolTrace] = field(default_factory=list)
    finding: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentMemory:
    """Append-only run log + accessor for building LLM context."""

    def __init__(self, goal: str):
        self.goal = goal
        self.plan: list[dict] = []
        self.steps: list[StepRecord] = []

    def set_plan(self, plan: list[dict]) -> None:
        self.plan = plan

    def start_step(self, step_id: int, description: str) -> StepRecord:
        record = StepRecord(step_id=step_id, description=description)
        self.steps.append(record)
        return record

    def context_for_step(self, current_step_id: int) -> str:
        """Summary of everything learned so far, for the executor's prompt."""
        prior = [s for s in self.steps if s.step_id < current_step_id and s.finding]
        if not prior:
            return "(no prior findings yet)"
        return "\n".join(f"- Step {s.step_id} ({s.description}): {s.finding}" for s in prior)

    def full_transcript(self) -> str:
        """Everything the agent learned, for the final synthesis prompt."""
        lines = [f"GOAL: {self.goal}", ""]
        for s in self.steps:
            lines.append(f"## Step {s.step_id}: {s.description}")
            for t in s.tool_traces:
                lines.append(f"  - used tool `{t.tool_name}`({t.tool_input}) -> {t.result_summary}")
            lines.append(f"  Finding: {s.finding}")
            lines.append("")
        return "\n".join(lines)
