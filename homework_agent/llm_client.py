"""
LLM client for Anthropic Claude API.

Provides two modes:
- chat_completion: simple request/response
- chat_with_tools: request with tool definitions, returns tool_use blocks
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API with native tool_use support."""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def chat_completion(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Simple chat completion — no tools."""
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self.BASE_URL, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Chat with tool definitions — returns content blocks (text + tool_use)."""
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(self.BASE_URL, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat_stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        """Streaming chat — yields text tokens as they arrive."""
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", self.BASE_URL, headers=self._headers(), json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data["type"] == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta["text"]