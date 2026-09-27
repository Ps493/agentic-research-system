"""Central configuration for the agent.

All tunables live here so behaviour can be changed without touching
orchestration logic.
"""
from __future__ import annotations

import logging
import os


# --- LLM ---
# LLM_PROVIDER: "anthropic" | "gemini" | "mock" | "auto" (default — picks based
# on which API key env var is present, preferring Gemini since it has a free tier).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto")
ANTHROPIC_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "2000"))

# --- Agent loop guards ---
MAX_PLAN_STEPS = int(os.environ.get("AGENT_MAX_PLAN_STEPS", "6"))
MAX_REACT_ITERATIONS_PER_STEP = int(os.environ.get("AGENT_MAX_REACT_ITERS", "5"))

# --- Tooling ---
WORKSPACE_DIR = os.environ.get("AGENT_WORKSPACE_DIR", "./workspace")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")  # optional real web-search provider

# --- Output ---
REPORTS_DIR = os.environ.get("AGENT_REPORTS_DIR", "./reports")


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger.

    Every step, tool call and LLM round-trip is logged at INFO level so a
    run can be audited after the fact — this is what makes the workflow
    "observable" rather than a black box.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                               datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("AGENT_LOG_LEVEL", "INFO"))
    return logger
