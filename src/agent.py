"""ResearchAgent — the orchestrator.

This is the only class a caller (CLI, API, notebook) needs to know
about. It wires together:

    Planner   (plan)     -> decompose the goal into steps
    Executor  (act)      -> ReAct loop with tools, per step
    Memory    (context)  -> state carried across every step
    Synthesizer (respond)-> final coherent report from all findings

                 ┌─────────┐
   goal ───────► │ Planner │
                 └────┬────┘
                      │ plan (steps)
                      ▼
        ┌───────────────────────────┐
        │  for each step:           │
        │   ┌────────┐   ┌────────┐ │
        │   │Executor│──►│ Tools  │ │  (ReAct: reason/act/observe loop)
        │   └───┬────┘   └────────┘ │
        │       │ finding           │
        │       ▼                   │
        │    Memory (state)         │
        └─────────────┬─────────────┘
                       │ full transcript
                       ▼
                ┌─────────────┐
                │ Synthesizer │ ──► final report (string)
                └─────────────┘
"""
from __future__ import annotations

from . import config
from .llm_client import LLMClient, build_llm_client
from .memory import AgentMemory
from .planner import Planner
from .executor import StepExecutor
from .tools import ToolRegistry
from .tools.web_search import WebSearchTool
from .tools.calculator import CalculatorTool
from .tools.file_io import FileReadTool, FileWriteTool

logger = config.get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are the synthesis module of a research agent. \
You are given the ORIGINAL GOAL and a transcript of everything discovered across \
several research steps (including which tools were used). \
Write the FINAL REPORT that directly and completely answers the goal. \
Structure it with a short title, a 2-3 sentence executive summary, then \
organized sections with the key findings. Be factual and concise. \
Output plain Markdown only.
"""


class ResearchAgent:
    """Plan-and-Execute + ReAct agent for open-ended research tasks."""

    def __init__(self, llm: LLMClient | None = None, tools: ToolRegistry | None = None):
        self._llm = llm or build_llm_client()
        self._tools = tools or ToolRegistry(
            [WebSearchTool(), CalculatorTool(), FileReadTool(), FileWriteTool()]
        )
        self._planner = Planner(self._llm)
        self._executor = StepExecutor(self._llm, self._tools)

    def run(self, goal: str) -> tuple[str, AgentMemory]:
        """Execute the full plan -> act -> observe -> respond workflow.

        Returns (final_report_markdown, memory) — memory is returned too
        so callers/tests can inspect the full audit trail of the run.
        """
        logger.info("=== New run === GOAL: %s", goal)
        memory = AgentMemory(goal)

        plan = self._planner.generate_plan(goal)
        memory.set_plan(plan)

        for step in plan:
            self._executor.execute(step, memory)

        report = self._synthesize(memory)
        logger.info("=== Run complete ===")
        return report, memory

    def _synthesize(self, memory: AgentMemory) -> str:
        response = self._llm.chat(
            messages=[{"role": "user", "content": memory.full_transcript()}],
            system=SYNTHESIS_SYSTEM_PROMPT,
        )
        return response.text
