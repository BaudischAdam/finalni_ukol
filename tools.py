"""
Tool definitions for the ReAct agent.

Each tool is a callable with a schema (for Claude tool_use) and an execute function.
Tools are intentionally simple and use free/public APIs — no API keys needed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

_HTTP_HEADERS = {"User-Agent": "ReActAgent/1.0 (homework project)"}

# ---------------------------------------------------------------------------
# Tool schema type
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """Tool definition compatible with Anthropic tool_use format."""
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Any  # async callable(params) -> str


# ---------------------------------------------------------------------------
# 1) Wikipedia Search
# ---------------------------------------------------------------------------

async def _wikipedia_search(params: dict) -> str:
    query = params["query"]
    limit = min(params.get("limit", 3), 5)

    async with httpx.AsyncClient(timeout=10, headers=_HTTP_HEADERS) as client:
        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            },
        )
        data = resp.json()

    results = data.get("query", {}).get("search", [])
    if not results:
        return f"No Wikipedia articles found for '{query}'."

    lines = []
    for r in results:
        snippet = re.sub(r"<.*?>", "", r.get("snippet", ""))
        lines.append(f"- **{r['title']}**: {snippet}")

    return "\n".join(lines)


wikipedia_search = ToolDef(
    name="wikipedia_search",
    description=(
        "Search Wikipedia for articles matching a query. "
        "Returns titles and short snippets."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for Wikipedia.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-5, default 3).",
            },
        },
        "required": ["query"],
    },
    execute=_wikipedia_search,
)


# ---------------------------------------------------------------------------
# 2) Wikipedia Get Article
# ---------------------------------------------------------------------------

async def _wikipedia_get_article(params: dict) -> str:
    title = params["title"]
    max_chars = params.get("max_chars", 2000)

    async with httpx.AsyncClient(timeout=10, headers=_HTTP_HEADERS) as client:
        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json",
            },
        )
        data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        if extract:
            if len(extract) > max_chars:
                extract = extract[:max_chars] + "..."
            return f"# {page['title']}\n\n{extract}"

    return f"Article '{title}' not found on Wikipedia."


wikipedia_get_article = ToolDef(
    name="wikipedia_get_article",
    description=(
        "Get the introductory text of a specific Wikipedia article by exact title. "
        "Use wikipedia_search first to find the correct title."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Exact Wikipedia article title.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 2000).",
            },
        },
        "required": ["title"],
    },
    execute=_wikipedia_get_article,
)


# ---------------------------------------------------------------------------
# 3) Calculator (sandboxed math eval)
# ---------------------------------------------------------------------------

_SAFE_NAMES: dict[str, Any] = {
    k: getattr(math, k)
    for k in [
        "sqrt", "sin", "cos", "tan", "log", "log2", "log10",
        "exp", "pi", "e", "ceil", "floor", "factorial", "pow", "fabs",
    ]
}
_SAFE_NAMES["abs"] = abs  # built-in abs
_SAFE_NAMES["__builtins__"] = {}  # block all builtins — only math allowed


async def _calculator(params: dict) -> str:
    expression = params["expression"]
    # Whitelist: only digits, operators, parens, dots, commas, function names
    if re.search(r"[a-zA-Z_]\w*", expression):
        for token in re.findall(r"[a-zA-Z_]\w*", expression):
            if token not in _SAFE_NAMES:
                return f"Error: '{token}' is not allowed. Only math functions are supported."
    try:
        # Safe: __builtins__ disabled, only math namespace
        result = eval(expression, _SAFE_NAMES)  # noqa: S307 — intentional sandboxed math eval
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


calculator = ToolDef(
    name="calculator",
    description=(
        "Evaluate a mathematical expression. Supports standard math functions: "
        "sqrt, sin, cos, tan, log, exp, pi, e, factorial, pow, abs. "
        "Example: 'sqrt(144) + 2**3'"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate.",
            },
        },
        "required": ["expression"],
    },
    execute=_calculator,
)


# ---------------------------------------------------------------------------
# 4) DuckDuckGo Web Search
# ---------------------------------------------------------------------------

async def _web_search(params: dict) -> str:
    query = params["query"]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        data = resp.json()

    lines = []

    # Abstract (instant answer)
    if data.get("AbstractText"):
        lines.append(f"**{data.get('Heading', 'Result')}**: {data['AbstractText']}")
        if data.get("AbstractURL"):
            lines.append(f"Source: {data['AbstractURL']}")

    # Related topics
    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            lines.append(f"- {topic['Text']}")

    if not lines:
        return f"No instant results for '{query}'. Try rephrasing or use wikipedia_search."

    return "\n".join(lines)


web_search = ToolDef(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo instant answers API. "
        "Good for factual queries, definitions, and quick lookups."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
        },
        "required": ["query"],
    },
    execute=_web_search,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TOOLS: list[ToolDef] = [
    wikipedia_search,
    wikipedia_get_article,
    calculator,
    web_search,
]

TOOL_MAP: dict[str, ToolDef] = {t.name: t for t in ALL_TOOLS}


def get_tool_schemas() -> list[dict]:
    """Return tool definitions in Anthropic API format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in ALL_TOOLS
    ]
