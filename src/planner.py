"""Planning stage — Plan-and-Execute pattern.

The LLM is asked to decompose the goal into a short, ordered list of
concrete research steps *before* any tool is called. This separates
"what should we find out" (planning) from "how do we find it" (acting
in the ReAct loop in executor.py), which keeps each LLM call focused
and makes the run auditable step-by-step.
"""
from __future__ import annotations

import json
import re

from . import config
from .llm_client import LLMClient

logger = config.get_logger(__name__)

PLANNER_SYSTEM_PROMPT = f"""You are the planning module of a research agent.

Break the following GOAL into a short, ordered list of concrete, \
independent research steps needed to fully answer it. \
Return STRICT JSON ONLY: a list of objects with keys "id" (int, starting at 1) \
and "description" (string). No prose, no markdown fences.

Rules:
- Produce at most {config.MAX_PLAN_STEPS} steps.
- Steps must be specific enough to search for (not "do research").
- Order steps logically (background before recent developments before synthesis-only steps).
"""


class Planner:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def generate_plan(self, goal: str) -> list[dict]:
        response = self._llm.chat(
            messages=[{"role": "user", "content": f"GOAL: {goal}"}],
            system=PLANNER_SYSTEM_PROMPT,
        )
        plan = self._parse_plan(response.text)
        logger.info("Generated plan with %d step(s)", len(plan))
        for step in plan:
            logger.info("  Step %s: %s", step["id"], step["description"])
        return plan

    @staticmethod
    def _parse_plan(raw_text: str) -> list[dict]:
        text = raw_text.strip()
        # Be defensive: strip markdown fences if the model added them anyway.
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
            assert isinstance(data, list)
            steps = []
            for i, item in enumerate(data[: config.MAX_PLAN_STEPS], start=1):
                steps.append({"id": item.get("id", i), "description": str(item["description"])})
            if not steps:
                raise ValueError("empty plan")
            return steps
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse plan JSON (%s); falling back to single-step plan.", exc)
            return [{"id": 1, "description": text[:200] or "Research the goal directly."}]
