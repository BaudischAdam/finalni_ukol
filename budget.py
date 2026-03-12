"""
Simple token budget tracker for the ReAct agent.

Tracks token usage per section (system, tools, response) and enforces limits.
Provides auto-stop signal when tool loop exceeds budget threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextBudget:
    """Token budget manager — tracks usage per section, signals when to stop."""

    context_window: int = 200_000
    budget_system: int = 2_000
    budget_tools: int = 20_000
    budget_response: int = 4_096
    tool_stop_threshold: float = 0.8  # stop tool loop at 80% of budget_tools

    # Internal counters
    _used: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._used = {"system": 0, "tools": 0, "conversation": 0}

    def add(self, section: str, tokens: int) -> None:
        self._used[section] = self._used.get(section, 0) + tokens

    @property
    def total_used(self) -> int:
        return sum(self._used.values())

    @property
    def tools_used(self) -> int:
        return self._used.get("tools", 0)

    @property
    def remaining(self) -> int:
        return max(0, self.context_window - self.total_used - self.budget_response)

    @property
    def should_stop_tools(self) -> bool:
        """True when tool usage exceeds threshold — signal to exit tool loop."""
        return self.tools_used >= int(self.budget_tools * self.tool_stop_threshold)

    @property
    def emergency(self) -> bool:
        """True when context is nearly full — must answer immediately."""
        return self.total_used >= (self.context_window - self.budget_response)

    def tool_call_summary(self) -> str:
        pct = (self.tools_used / self.budget_tools * 100) if self.budget_tools else 0
        return (
            f"Budget: {self.tools_used}/{self.budget_tools} tool tokens "
            f"({pct:.0f}%), total {self.total_used}/{self.context_window}"
        )


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)
