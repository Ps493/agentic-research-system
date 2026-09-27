"""LLM client abstraction.

Keeping this behind an interface (LLMClient) means the rest of the
system — planner, executor, synthesizer — never talks to the Anthropic
SDK directly. That makes it trivial to:
  * swap in OpenAI / a local model later (implement the same interface)
  * unit test the whole agentic loop with MockLLMClient, with zero
    network calls and fully deterministic output
"""
from __future__ import annotations

import abc
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import config

logger = config.get_logger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    # Opaque, base64-encoded "thought signature" some providers (Gemini 3+)
    # attach to function-call turns. It must be echoed back verbatim on the
    # next turn for multi-turn tool use to work; providers that don't use
    # this concept (Anthropic, the mock) simply leave it as None.
    thought_signature: Optional[str] = None


@dataclass
class LLMResponse:
    """Normalized response, regardless of provider."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None

    @property
    def wants_tool_use(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(abc.ABC):
    """Common interface every LLM backend must implement."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        ...


class AnthropicLLMClient(LLMClient):
    """Real backend, calling the Anthropic Messages API with native tool use."""

    def __init__(self, model: str = config.ANTHROPIC_MODEL, max_tokens: int = config.MAX_TOKENS):
        try:
            import anthropic  # imported lazily so mock mode needs no SDK/API key
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required for AnthropicLLMClient. "
                "Run: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def chat(self, messages, system: str = "", tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=self._sanitize(messages),
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        logger.debug("Calling Anthropic API | messages=%d tools=%s", len(messages), bool(tools))
        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw=response,
        )

    @staticmethod
    def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip provider-agnostic extras (e.g. Gemini's "_thought_signature")
        that Anthropic's API doesn't know about, so switching LLM_PROVIDER
        mid-run history never sends it an unrecognized field."""
        cleaned = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = [{k: v for k, v in block.items() if not k.startswith("_")} for block in content]
            cleaned.append({**msg, "content": content})
        return cleaned


class GeminiLLMClient(LLMClient):
    """Google Gemini backend (free-tier friendly), on the current `google-genai` SDK.

    NOTE: this deliberately does NOT use the older, now-deprecated
    `google-generativeai` package. Gemini 3+ models require a
    `thought_signature` to be echoed back on function-call turns for
    multi-turn tool use to work at all (the deprecated SDK doesn't
    surface this field, which causes an INVALID_ARGUMENT error). See:
    https://ai.google.dev/gemini-api/docs/thought-signatures

    As with GeminiLLMClient's predecessor, the rest of the codebase
    (planner/executor/agent) only ever builds messages in Anthropic's
    tool-use shape (content blocks with "tool_use" / "tool_result").
    This client privately translates that shape into Gemini's
    Content/Part objects on the way in — including round-tripping the
    thought_signature via a private "_thought_signature" key on the
    tool_use block — and translates Gemini's response back into the
    provider-agnostic LLMResponse on the way out.
    """

    def __init__(self, model: str = config.GEMINI_MODEL):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'google-genai' package is required for GeminiLLMClient. "
                "Run: pip install google-genai"
            ) from exc
        import os

        self._types = types
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._model_name = model

    def chat(self, messages, system: str = "", tools=None) -> LLMResponse:
        types = self._types
        contents = self._to_gemini_contents(messages)

        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"], description=t["description"], parameters=t["input_schema"]
                    )
                    for t in tools
                ])
            ]
            # We handle every tool call ourselves via ToolRegistry — disabling the
            # SDK's automatic function calling avoids the "AFC is not recommended
            # in generate_content" warning and any surprise double-invocation.
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

        response = self._call_with_retry(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if (candidate and candidate.content) else []
        for i, part in enumerate(parts):
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                sig = base64.b64encode(part.thought_signature).decode() if part.thought_signature else None
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=part.function_call.name,
                        input=dict(part.function_call.args or {}),
                        thought_signature=sig,
                    )
                )

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            raw=response,
        )

    def _call_with_retry(self, max_attempts: int = 5, backoff_base: float = 2.0, **kwargs):
        """Retry on transient server-side errors (503 UNAVAILABLE / 429 rate
        limit are common on the free tier under load) with exponential
        backoff, instead of letting a single blip fail the whole run."""
        from google.genai import errors

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._client.models.generate_content(**kwargs)
            except errors.ServerError as exc:
                last_exc = exc
                wait = backoff_base ** attempt
                logger.warning(
                    "Gemini server error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, max_attempts, exc, wait,
                )
                time.sleep(wait)
            except errors.ClientError as exc:
                if getattr(exc, "code", None) == 429:
                    # A per-day quota exhaustion won't resolve within this
                    # process's lifetime — retrying just burns the run's
                    # remaining time budget for nothing. Fail fast with a
                    # clear message instead. Per-minute bursts, by contrast,
                    # genuinely clear up within seconds, so those are worth
                    # backing off and retrying.
                    if "PerDay" in str(exc):
                        raise RuntimeError(
                            "Gemini free-tier DAILY quota exhausted for this model. "
                            "This resets at midnight Pacific Time — it will not "
                            "recover by retrying now. Either wait for the reset, "
                            "switch GEMINI_MODEL to a higher-quota model (e.g. "
                            "gemini-3.5-flash-lite has ~500 RPD vs ~20 RPD for "
                            "gemini-3.8-flash), or reduce AGENT_MAX_PLAN_STEPS / "
                            "AGENT_MAX_REACT_ITERS to use fewer requests per run."
                        ) from exc
                    last_exc = exc
                    wait = backoff_base ** attempt
                    logger.warning(
                        "Gemini per-minute rate limit hit (attempt %d/%d) — retrying in %.0fs",
                        attempt, max_attempts, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # noqa: RSE102 — exhausted retries, surface the last real error

    # --- translation helpers -------------------------------------------

    def _to_gemini_contents(self, messages: list[dict]) -> list[Any]:
        """Convert our Anthropic-shaped message list into Gemini Content objects.

        Tracks tool_use_id -> tool name so tool_result blocks (which lack a
        name) can be reattached correctly, and round-trips thought_signature
        so Gemini 3's multi-turn tool-use requirement is satisfied.
        """
        types = self._types
        call_id_to_name: dict[str, str] = {}
        contents: list[Any] = []

        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            parts: list[Any] = []

            if isinstance(content, str):
                parts.append(types.Part(text=content))
            else:
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append(types.Part(text=block["text"]))
                    elif btype == "tool_use":
                        call_id_to_name[block["id"]] = block["name"]
                        part_kwargs: dict[str, Any] = {
                            "function_call": types.FunctionCall(name=block["name"], args=block["input"])
                        }
                        sig = block.get("_thought_signature")
                        if sig:
                            part_kwargs["thought_signature"] = base64.b64decode(sig)
                        parts.append(types.Part(**part_kwargs))
                    elif btype == "tool_result":
                        name = call_id_to_name.get(block["tool_use_id"], "unknown_tool")
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=name, response={"result": block["content"]}
                                )
                            )
                        )
            contents.append(types.Content(role=role, parts=parts))

        return contents


class MockLLMClient(LLMClient):
    """Deterministic, offline stand-in used by tests and demo/no-API-key mode.

    It follows a scripted policy so the *shape* of the agentic loop
    (plan -> act -> observe -> respond) can be exercised without network
    access or an API key: given a goal it returns a canned 2-step plan,
    then on the executor's first turn per step it emits a tool call, and
    on the second turn (after seeing the tool result) it emits a final
    text answer for that step.
    """

    def __init__(self):
        self._step_turn_count: dict[str, int] = {}

    def chat(self, messages, system: str = "", tools=None) -> LLMResponse:
        last_user_msg = messages[-1] if messages else {}
        content = last_user_msg.get("content", "")
        as_text = content if isinstance(content, str) else json.dumps(content)

        # 1) Planning request
        if "Break the following GOAL" in system or "Break the following GOAL" in as_text:
            plan = [
                {"id": 1, "description": f"Gather background facts about: {self._extract_goal(as_text)}"},
                {"id": 2, "description": "Identify recent developments / current state"},
            ]
            return LLMResponse(text=json.dumps(plan))

        # 2) Synthesis request
        if "Write the FINAL REPORT" in system or "Write the FINAL REPORT" in as_text:
            return LLMResponse(text="# Mock Final Report\n\nThis is a mock, offline synthesis "
                                     "of all step findings (no API key was configured).")

        # 3) Step execution turn (ReAct): alternate tool_call -> final answer
        key = self._turn_key(messages)
        turn = self._step_turn_count.get(key, 0)
        self._step_turn_count[key] = turn + 1

        if turn == 0 and tools:
            tool_name = tools[0]["name"]
            return LLMResponse(
                tool_calls=[ToolCall(id="mock_call_1", name=tool_name, input={"query": as_text[:60]})],
                stop_reason="tool_use",
            )
        return LLMResponse(text=f"[mock finding] Completed step using {len(tools or [])} available tool(s).")

    @staticmethod
    def _extract_goal(text: str) -> str:
        return text.strip().splitlines()[0][:80]

    @staticmethod
    def _turn_key(messages: list[dict[str, Any]]) -> str:
        # Group by the first user message so each step gets its own turn counter
        for m in messages:
            if m.get("role") == "user":
                c = m["content"]
                return (c if isinstance(c, str) else json.dumps(c))[:40]
        return "default"


def build_llm_client() -> LLMClient:
    """Factory: picks the backend based on config.LLM_PROVIDER / available keys.

    LLM_PROVIDER="auto" (default) prefers Gemini (free tier) if
    GEMINI_API_KEY is set, then Anthropic if ANTHROPIC_API_KEY is set,
    then falls back to the offline MockLLMClient so the agent always runs.
    Set LLM_PROVIDER explicitly to force one backend.
    """
    import os

    provider = config.LLM_PROVIDER.lower()

    if provider == "gemini" or (provider == "auto" and os.environ.get("GEMINI_API_KEY")):
        if os.environ.get("GEMINI_API_KEY"):
            logger.info("Using GeminiLLMClient (model=%s)", config.GEMINI_MODEL)
            return GeminiLLMClient()
        logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set — falling back to mock.")
        return MockLLMClient()

    if provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        if os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("Using AnthropicLLMClient (model=%s)", config.ANTHROPIC_MODEL)
            return AnthropicLLMClient()
        logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set — falling back to mock.")
        return MockLLMClient()

    logger.warning("No GEMINI_API_KEY or ANTHROPIC_API_KEY found — using MockLLMClient (offline demo mode)")
    return MockLLMClient()
