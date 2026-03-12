"""
Conversation persistence & semantic memory.

Simulates a database using JSON files in ~/.react_agent/conversations/.
Each conversation is a JSON file with messages, key_facts, and progressive_summary.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STORE_DIR = Path.home() / ".react_agent" / "conversations"


@dataclass
class KeyFacts:
    """Structured semantic memory — extracted after each turn by LLM."""
    topic: str | None = None
    entities: list[str] = field(default_factory=list)
    user_intent: str | None = None
    key_numbers: dict[str, str] = field(default_factory=dict)
    conclusions: list[str] = field(default_factory=list)

    def merge(self, other: KeyFacts) -> None:
        """Merge new facts into existing, newer values win."""
        if other.topic:
            self.topic = other.topic
        if other.user_intent:
            self.user_intent = other.user_intent
        self.entities = list(dict.fromkeys(self.entities + other.entities))[:20]
        self.key_numbers.update(other.key_numbers)
        self.conclusions = list(dict.fromkeys(self.conclusions + other.conclusions))[:10]

    def to_prompt_block(self) -> str:
        """Format key facts as context for the LLM."""
        if not any([self.topic, self.entities, self.key_numbers, self.conclusions]):
            return ""
        lines = ["<key_facts>"]
        if self.topic:
            lines.append(f"  Topic: {self.topic}")
        if self.entities:
            lines.append(f"  Entities: {', '.join(self.entities[:10])}")
        if self.user_intent:
            lines.append(f"  User intent: {self.user_intent}")
        if self.key_numbers:
            nums = "; ".join(f"{k}: {v}" for k, v in list(self.key_numbers.items())[:10])
            lines.append(f"  Key numbers: {nums}")
        if self.conclusions:
            for c in self.conclusions[:5]:
                lines.append(f"  - {c}")
        lines.append("</key_facts>")
        return "\n".join(lines)


@dataclass
class Conversation:
    """A single conversation with full state."""
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: list[dict] = field(default_factory=list)
    progressive_summary: str | None = None
    key_facts: dict = field(default_factory=dict)
    turn_count: int = 0
    total_tokens: int = 0

    def get_key_facts(self) -> KeyFacts:
        kf = KeyFacts()
        if self.key_facts:
            kf.topic = self.key_facts.get("topic")
            kf.entities = self.key_facts.get("entities", [])
            kf.user_intent = self.key_facts.get("user_intent")
            kf.key_numbers = self.key_facts.get("key_numbers", {})
            kf.conclusions = self.key_facts.get("conclusions", [])
        return kf

    def set_key_facts(self, kf: KeyFacts) -> None:
        self.key_facts = asdict(kf)


class ConversationStore:
    """JSON-file backed conversation persistence (simulates DB)."""

    def __init__(self, store_dir: Path = STORE_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conv_id: str) -> Path:
        return self.store_dir / f"{conv_id}.json"

    def create(self, title: str = "New conversation") -> Conversation:
        conv_id = f"conv-{int(time.time())}-{os.getpid()}"
        now = time.time()
        conv = Conversation(
            id=conv_id, title=title, created_at=now, updated_at=now
        )
        self.save(conv)
        return conv

    def save(self, conv: Conversation) -> None:
        conv.updated_at = time.time()
        with open(self._path(conv.id), "w") as f:
            json.dump(asdict(conv), f, indent=2, ensure_ascii=False)

    def load(self, conv_id: str) -> Conversation | None:
        path = self._path(conv_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return Conversation(**data)

    def list_all(self) -> list[Conversation]:
        convs = []
        for path in sorted(self.store_dir.glob("conv-*.json"), reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)
                convs.append(Conversation(**data))
            except (json.JSONDecodeError, TypeError):
                continue
        return convs

    def delete(self, conv_id: str) -> bool:
        path = self._path(conv_id)
        if path.exists():
            path.unlink()
            return True
        return False
