"""Web search tool.

Provider is chosen at construction time, in order of preference:
  1. Tavily API           (if TAVILY_API_KEY is set)      — real, recommended
  2. DuckDuckGo HTML       (no key needed, best-effort)    — real, free
  3. Offline mock provider (no network / for tests+demo)   — deterministic

This lets the same agent code run fully offline for grading/demo
purposes and with real live results when a key/network is available.
"""
from __future__ import annotations

import os

from . import Tool, ToolResult
from .. import config

logger = config.get_logger(__name__)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for up-to-date information on a topic. "
        "Returns a short list of titles, URLs and snippets."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._provider = self._select_provider()
        logger.info("WebSearchTool using provider: %s", self._provider)

    def _select_provider(self) -> str:
        if os.environ.get("TAVILY_API_KEY"):
            return "tavily"
        try:
            import requests  # noqa: F401
            return "duckduckgo"
        except ImportError:
            return "mock"

    def run(self, query: str) -> ToolResult:
        try:
            if self._provider == "tavily":
                results = self._search_tavily(query)
            elif self._provider == "duckduckgo":
                results = self._search_duckduckgo(query)
            else:
                results = self._search_mock(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Search provider '%s' failed (%s); falling back to mock.", self._provider, exc)
            results = self._search_mock(query)

        if not results:
            return ToolResult(ok=True, content=f"No results found for '{query}'.")

        formatted = "\n".join(
            f"{i+1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results[: self.max_results])
        )
        return ToolResult(ok=True, content=formatted)

    # --- Providers -----------------------------------------------------

    def _search_tavily(self, query: str) -> list[dict]:
        import requests

        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": self.max_results},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:300]}
            for r in data.get("results", [])
        ]

    def _search_duckduckgo(self, query: str) -> list[dict]:
        import requests
        from html.parser import HTMLParser

        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (agentic-research-system)"},
            timeout=15,
        )
        resp.raise_for_status()

        class _Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results, self._in_link, self._cur = [], False, {}

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and "result__a" in (attrs.get("class") or ""):
                    self._in_link = True
                    self._cur = {"title": "", "url": attrs.get("href", ""), "snippet": ""}

            def handle_data(self, data):
                if self._in_link:
                    self._cur["title"] += data

            def handle_endtag(self, tag):
                if tag == "a" and self._in_link:
                    self._in_link = False
                    self.results.append(self._cur)

        parser = _Parser()
        parser.feed(resp.text)
        return parser.results[: self.max_results]

    def _search_mock(self, query: str) -> list[dict]:
        return [
            {
                "title": f"[offline demo result] Overview of '{query}'",
                "url": "https://example.com/offline-demo",
                "snippet": (
                    "No live search provider is configured (no TAVILY_API_KEY and/or no network). "
                    "This is a placeholder result so the ReAct loop can be demonstrated end-to-end offline."
                ),
            }
        ]
