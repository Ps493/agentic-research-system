"""Execution stage — ReAct pattern, run once per planned step.

For a single step the agent alternates:
  Reason  -> the LLM decides whether it needs a tool
  Act     -> if so, we invoke the tool
  Observe -> the tool result is fed back to the LLM as a message
until the LLM stops calling tools and gives a final text finding for
that step (or the iteration guard trips, to prevent infinite loops).
"""
from __future__ import annotations

from typing import Any

from . import config
from .llm_client import LLMClient
from .memory import AgentMemory, StepRecord, ToolTrace
from .tools import ToolRegistry

logger = config.get_logger(__name__)

EXECUTOR_SYSTEM_PROMPT = """You are the execution module of a research agent, working on ONE step \
of a larger plan. Use the available tools as needed to investigate the step. \
When you have enough information, respond with a concise, factual finding for this \
step in plain text (2-6 sentences) and DO NOT call any more tools.

Context already known from earlier steps:
{context}
"""


class StepExecutor:
    def __init__(self, llm: LLMClient, tools: ToolRegistry):
        self._llm = llm
        self._tools = tools

    def execute(self, step: dict, memory: AgentMemory) -> StepRecord:
        record = memory.start_step(step["id"], step["description"])
        system = EXECUTOR_SYSTEM_PROMPT.format(context=memory.context_for_step(step["id"]))
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"STEP: {step['description']}"}
        ]

        for iteration in range(config.MAX_REACT_ITERATIONS_PER_STEP):
            response = self._llm.chat(messages=messages, system=system, tools=self._tools.schemas())

            if not response.wants_tool_use:
                record.finding = response.text or "(no finding produced)"
                logger.info("Step %s finished after %d iteration(s)", step["id"], iteration + 1)
                return record

            # Assistant turn included tool_use blocks — append it, then run each tool
            # and append the observations as a user turn (Anthropic tool-use protocol).
            messages.append({"role": "assistant", "content": self._to_content_blocks(response)})
            tool_result_blocks = []
            for call in response.tool_calls:
                result = self._tools.invoke(call.name, call.input)
                summary = result.content if result.ok else f"ERROR: {result.error}"
                record.tool_traces.append(
                    ToolTrace(tool_name=call.name, tool_input=call.input, result_summary=summary[:300])
                )
                logger.info("Step %s -> tool `%s`(%s)", step["id"], call.name, call.input)
                tool_result_blocks.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": summary}
                )
            messages.append({"role": "user", "content": tool_result_blocks})

        record.finding = "(step exceeded max reasoning iterations without a final answer)"
        logger.warning("Step %s hit MAX_REACT_ITERATIONS_PER_STEP guard", step["id"])
        return record

    @staticmethod
    def _to_content_blocks(response) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if response.text:
            blocks.append({"type": "text", "text": response.text})
        for call in response.tool_calls:
            block: dict[str, Any] = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            # Provider-specific extra (e.g. Gemini 3's thought_signature) travels
            # in this underscore-prefixed key so it round-trips through memory
            # without any other module needing to know it exists. Providers
            # that don't use the concept (Anthropic, mock) just never set it.
            if getattr(call, "thought_signature", None):
                block["_thought_signature"] = call.thought_signature
            blocks.append(block)
        return blocks
