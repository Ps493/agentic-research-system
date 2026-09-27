"""Unit tests for the agentic workflow.

These run fully offline (MockLLMClient, no network) so the *shape* of
the workflow — plan -> act (tool use) -> observe -> respond, with state
carried across steps — is verified deterministically in CI without an
API key.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import ResearchAgent
from src.llm_client import MockLLMClient
from src.tools import ToolRegistry, Tool, ToolResult
from src.tools.calculator import CalculatorTool
from src.planner import Planner
from src.memory import AgentMemory


class EchoTool(Tool):
    name = "web_search"
    description = "test double for web search"
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def __init__(self):
        self.calls = []

    def run(self, query: str) -> ToolResult:
        self.calls.append(query)
        return ToolResult(ok=True, content=f"mock search results for: {query}")


class TestPlanner(unittest.TestCase):
    def test_generates_ordered_plan(self):
        planner = Planner(MockLLMClient())
        plan = planner.generate_plan("What is the current state of fusion energy?")
        self.assertGreaterEqual(len(plan), 1)
        self.assertIn("description", plan[0])
        self.assertEqual(plan[0]["id"], 1)

    def test_malformed_json_falls_back_gracefully(self):
        class BrokenLLM(MockLLMClient):
            def chat(self, messages, system="", tools=None):
                from src.llm_client import LLMResponse
                return LLMResponse(text="not json at all")

        planner = Planner(BrokenLLM())
        plan = planner.generate_plan("anything")
        self.assertEqual(len(plan), 1)  # graceful single-step fallback, not a crash


class TestCalculatorTool(unittest.TestCase):
    def test_basic_arithmetic(self):
        tool = CalculatorTool()
        result = tool.run(expression="(4 + 6) * 2")
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "20")

    def test_rejects_unsafe_input(self):
        tool = CalculatorTool()
        result = tool.run(expression="__import__('os').system('echo hacked')")
        self.assertFalse(result.ok)


class TestFullAgentLoop(unittest.TestCase):
    def test_end_to_end_run_produces_report_and_uses_tools(self):
        echo = EchoTool()
        registry = ToolRegistry([echo])
        agent = ResearchAgent(llm=MockLLMClient(), tools=registry)

        report, memory = agent.run("Investigate the market for reusable rockets")

        # A plan was produced and stored in memory.
        self.assertGreaterEqual(len(memory.plan), 1)
        # Every planned step was executed and has a finding (state carried through).
        self.assertEqual(len(memory.steps), len(memory.plan))
        for step_record in memory.steps:
            self.assertTrue(step_record.finding)
        # The tool was actually invoked during the ReAct loop.
        self.assertGreater(len(echo.calls), 0)
        # A final coherent report was synthesized.
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)

    def test_memory_context_accumulates_across_steps(self):
        memory = AgentMemory("test goal")
        memory.set_plan([{"id": 1, "description": "a"}, {"id": 2, "description": "b"}])
        r1 = memory.start_step(1, "a")
        r1.finding = "finding A"
        context_for_step_2 = memory.context_for_step(2)
        self.assertIn("finding A", context_for_step_2)


if __name__ == "__main__":
    unittest.main()
