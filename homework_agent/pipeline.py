"""
ReAct Pipeline — the core agent logic.

Five-phase flow:
1. ROUTE   — classify query complexity (simple vs reasoning)
2. THINK   — analyze user query, plan which tools to use (structured JSON)
3. TOOLS   — iterative tool loop with budget tracking + self-summarization
4. ANSWER  — generate final response using gathered context
5. REFLECT — extract key facts + progressive summary (post-turn)

This is a custom framework implementation demonstrating the ReAct
(Reasoning + Acting) pattern with native Claude tool_use.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from budget import ContextBudget, estimate_tokens
from llm_client import AnthropicClient
from memory import Conversation, KeyFacts
from tools import TOOL_MAP, get_tool_schemas

console = Console()

# ---------------------------------------------------------------------------
# Debug output helper
# ---------------------------------------------------------------------------

_debug = True


def set_debug(enabled: bool) -> None:
    global _debug
    _debug = enabled


def dbg(msg: str, style: str = "dim") -> None:
    """Print only when debug mode is on."""
    if _debug:
        console.print(f"  {msg}", style=style, highlight=False)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful research assistant. You have access to tools for searching \
Wikipedia, performing calculations, and web search. Use them when needed to \
provide accurate, well-sourced answers.

Rules:
- Always verify facts using tools before answering
- Cite your sources
- If a calculation is needed, use the calculator tool
- Be concise but thorough
"""

THINK_SYSTEM_PROMPT = """\
You are an analytical module. Given a user query, decide what tools to call \
and in what order. Output ONLY valid JSON with this structure:

{
    "reasoning": "brief analysis of what the user needs",
    "search_plan": [
        {"tool": "tool_name", "params": {"key": "value"}, "reason": "why"}
    ]
}

Available tools: wikipedia_search, wikipedia_get_article, calculator, web_search.
If no tools needed, return empty search_plan.
"""

ANSWER_SYSTEM_PROMPT = """\
You are a helpful research assistant. Using the tool results provided in the \
conversation, generate a clear and comprehensive answer. Cite sources where \
applicable. Be concise but thorough.
"""

ROUTE_SYSTEM_PROMPT = """\
You are a query classifier. Given a user query, decide if it needs tool usage \
(reasoning mode) or can be answered directly (simple mode).

Output ONLY valid JSON:
{"mode": "simple" | "reasoning", "reason": "brief explanation"}

Use "simple" for: greetings, opinions, simple follow-ups, general knowledge.
Use "reasoning" for: factual questions, calculations, comparisons, anything \
that benefits from tool verification.
"""

KEY_FACTS_SYSTEM_PROMPT = """\
Extract structured key facts from this conversation turn. Output ONLY valid JSON:

{
    "topic": "main topic discussed",
    "entities": ["entity1", "entity2"],
    "user_intent": "what the user wanted to know",
    "key_numbers": {"label": "value"},
    "conclusions": ["key conclusion 1", "key conclusion 2"]
}
"""

SUMMARIZE_SYSTEM_PROMPT = """\
Summarize this conversation concisely, preserving:
- Key facts and numbers mentioned
- Conclusions reached
- Important context for future turns
Keep it under 200 words. Output plain text, no JSON.
"""

SELF_SUMMARIZE_PROMPT = """\
The tool context is getting large. Compress the following tool results into \
a concise summary that preserves:
- All source references and citations
- Key numbers, dates, and facts
- Technical values and measurements
Discard verbose descriptions. Keep it under 300 words. Output plain text.
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ReActPipeline:
    """
    Five-phase ReAct agent pipeline with context management.

    Phase 1 (ROUTE):   Classify query -> simple (direct) or reasoning (tools).
    Phase 2 (THINK):   Lightweight LLM call to plan tool usage.
    Phase 3 (TOOLS):   Iterative tool loop with budget + self-summarization.
    Phase 4 (ANSWER):  Streaming final answer with all gathered context.
    Phase 5 (REFLECT): Extract key facts + update progressive summary.
    """

    MAX_TOOL_ITERATIONS = 8

    def __init__(self, client: AnthropicClient):
        self.client = client
        self.budget = ContextBudget()

    async def run(self, user_query: str, conversation: Conversation) -> str:
        """Execute the full ReAct pipeline and return the final answer."""

        context_messages = self._build_context(conversation)

        # ── Phase 1: ROUTE ─────────────────────────────────────────────
        mode = await self._route(user_query, context_messages)
        mode_name = mode["mode"]
        mode_reason = mode.get("reason", "")

        if _debug:
            mode_color = "green" if mode_name == "simple" else "cyan"
            console.print(Panel(
                f"[bold {mode_color}]{mode_name.upper()}[/] — {mode_reason}",
                title="[bold]Phase 1: ROUTE[/]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            ))

        if mode_name == "simple":
            answer = await self._simple_answer(user_query, context_messages)
            await self._reflect(user_query, answer, conversation)
            self._print_budget()
            return answer

        # ── Phase 2: THINK ─────────────────────────────────────────────
        think_result = await self._think(user_query, context_messages)

        if _debug:
            plan = think_result.get("search_plan", [])
            think_text = Text()
            think_text.append(think_result.get("reasoning", "N/A"))
            if plan:
                think_text.append(f"\n\nPlan ({len(plan)} steps):", style="bold")
                for i, step in enumerate(plan, 1):
                    think_text.append(f"\n  {i}. ", style="bold cyan")
                    think_text.append(f"{step['tool']}", style="green")
                    think_text.append(f"({step.get('params', {})})", style="dim")
                    if step.get("reason"):
                        think_text.append(f" — {step['reason']}", style="dim")

            console.print(Panel(
                think_text,
                title="[bold]Phase 2: THINK[/]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            ))

        # ── Phase 3: TOOL LOOP (ReAct) ────────────────────────────────
        tool_messages, loop_answer = await self._tool_loop(user_query, context_messages)

        # ── Phase 4: ANSWER ───────────────────────────────────────────
        if loop_answer:
            answer = loop_answer
        else:
            answer = await self._answer(user_query, context_messages, tool_messages)

        # ── Phase 5: REFLECT ──────────────────────────────────────────
        await self._reflect(user_query, answer, conversation)
        self._print_budget()

        return answer

    def _print_budget(self) -> None:
        if not _debug:
            return
        pct = (self.budget.tools_used / self.budget.budget_tools * 100) if self.budget.budget_tools else 0
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar_color = "green" if pct < 60 else ("yellow" if pct < 80 else "red")
        bar = f"[{bar_color}]{'█' * filled}{'░' * (bar_len - filled)}[/]"

        console.print(Panel(
            f"Tool budget: {bar} {self.budget.tools_used:,}/{self.budget.budget_tools:,} ({pct:.0f}%)\n"
            f"Total used:  {self.budget.total_used:,}/{self.budget.context_window:,} tokens",
            title="[bold]Budget[/]",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    # -------------------------------------------------------------------
    # Context builder
    # -------------------------------------------------------------------

    def _build_context(self, conversation: Conversation) -> list[dict]:
        """Build message list from conversation state."""
        messages = []

        if conversation.progressive_summary:
            messages.append({
                "role": "user",
                "content": f"<conversation_summary>\n{conversation.progressive_summary}\n</conversation_summary>",
            })
            messages.append({
                "role": "assistant",
                "content": "I understand the context from our previous conversation. How can I help?",
            })

        kf = conversation.get_key_facts()
        facts_block = kf.to_prompt_block()
        if facts_block:
            messages.append({"role": "user", "content": facts_block})
            messages.append({
                "role": "assistant",
                "content": "I have the key facts from our conversation loaded.",
            })

        recent = conversation.messages[-12:]
        messages.extend(recent)

        return messages

    # -------------------------------------------------------------------
    # Phase 1: ROUTE
    # -------------------------------------------------------------------

    async def _route(self, query: str, context: list[dict]) -> dict:
        messages = list(context)
        messages.append({"role": "user", "content": query})

        resp = await self.client.chat_completion(
            system=ROUTE_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=100,
            temperature=0.0,
        )

        text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"mode": "reasoning", "reason": "fallback to reasoning"}

    # -------------------------------------------------------------------
    # Phase 2: THINK
    # -------------------------------------------------------------------

    async def _think(self, query: str, context: list[dict]) -> dict:
        messages = list(context)
        messages.append({"role": "user", "content": query})

        resp = await self.client.chat_completion(
            system=THINK_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=500,
            temperature=0.1,
        )

        text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        usage = resp.get("usage", {})
        self.budget.add("system", usage.get("input_tokens", 0))

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                json_str = text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                return json.loads(json_str.strip())
            return {"reasoning": text, "search_plan": []}

    # -------------------------------------------------------------------
    # Phase 3: TOOL LOOP
    # -------------------------------------------------------------------

    async def _tool_loop(
        self, query: str, conversation: list[dict]
    ) -> tuple[list[dict], str | None]:
        """Iterative tool loop with self-summarization."""
        messages = list(conversation)
        messages.append({"role": "user", "content": query})

        tool_schemas = get_tool_schemas()
        iteration = 0
        tool_results_text: list[str] = []

        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1

            if self.budget.emergency:
                dbg("⚠️  Emergency: context full, skipping to answer", "red bold")
                break
            if self.budget.should_stop_tools:
                dbg(f"⚠️  Tool budget 80% reached, stopping", "yellow")
                break

            resp = await self.client.chat_with_tools(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tool_schemas,
                max_tokens=4096,
                temperature=0.3,
            )

            usage = resp.get("usage", {})
            self.budget.add("conversation", usage.get("input_tokens", 0))

            stop_reason = resp.get("stop_reason")
            content_blocks = resp.get("content", [])

            assistant_content = []
            tool_uses = []

            for block in content_blocks:
                if block["type"] == "text":
                    assistant_content.append(block)
                    if block["text"].strip() and _debug:
                        console.print(f"  [dim italic]💭 {block['text'][:200]}[/]")
                elif block["type"] == "tool_use":
                    assistant_content.append(block)
                    tool_uses.append(block)

            messages.append({"role": "assistant", "content": assistant_content})

            if stop_reason == "end_turn" or not tool_uses:
                final_text = " ".join(
                    b["text"] for b in assistant_content if b["type"] == "text" and b["text"].strip()
                )
                if _debug:
                    console.print(f"  [dim]🔄 Tool loop done — {iteration} iteration(s)[/]")
                return messages, final_text if final_text else None

            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_id = tool_use["id"]

                if _debug:
                    params_str = json.dumps(tool_input, ensure_ascii=False)[:80]
                    console.print(f"  [green]🔧 {tool_name}[/]([dim]{params_str}[/])")

                tool_def = TOOL_MAP.get(tool_name)
                if tool_def:
                    try:
                        result = await tool_def.execute(tool_input)
                    except Exception as exc:
                        result = f"Error: {exc}"
                else:
                    result = f"Unknown tool: {tool_name}"

                result_tokens = estimate_tokens(result)
                self.budget.add("tools", result_tokens)
                tool_results_text.append(f"[{tool_name}] {result}")

                if _debug:
                    short_result = result[:120].replace("\n", " ")
                    pct = (self.budget.tools_used / self.budget.budget_tools * 100)
                    console.print(f"     [dim]→ {short_result}...[/]")
                    console.print(f"     [dim]+{result_tokens} tok (budget {pct:.0f}%)[/]")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

            # Self-summarization trigger
            if self.budget.should_stop_tools and tool_results_text:
                if _debug:
                    console.print("  [yellow bold]🗜️  Self-summarization: compressing tool context...[/]")
                compressed = await self._self_summarize(tool_results_text)
                tool_results_text = [compressed]
                if _debug:
                    console.print(f"     [dim]Compressed to {estimate_tokens(compressed)} tokens[/]")

        if _debug:
            console.print(f"  [dim]🔄 Tool loop done — {iteration} iteration(s)[/]")
        return messages, None

    async def _self_summarize(self, tool_results: list[str]) -> str:
        combined = "\n\n".join(tool_results)
        messages = [{"role": "user", "content": combined}]

        resp = await self.client.chat_completion(
            system=SELF_SUMMARIZE_PROMPT,
            messages=messages,
            max_tokens=500,
            temperature=0.1,
        )

        text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text += block["text"]
        return text or combined[:500]

    # -------------------------------------------------------------------
    # Phase 4: ANSWER
    # -------------------------------------------------------------------

    async def _simple_answer(self, query: str, context: list[dict]) -> str:
        messages = list(context)
        messages.append({"role": "user", "content": query})

        full_answer = []
        async for token in self.client.chat_stream(
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        ):
            sys.stdout.write(token)
            sys.stdout.flush()
            full_answer.append(token)

        print()
        return "".join(full_answer)

    async def _answer(
        self, query: str, conversation: list[dict], tool_messages: list[dict]
    ) -> str:
        messages = list(tool_messages)

        if messages and messages[-1].get("role") == "user":
            messages.append({
                "role": "assistant",
                "content": "I now have all the information needed. Let me provide a comprehensive answer.",
            })
            messages.append({
                "role": "user",
                "content": "Please provide your final answer based on the tool results above.",
            })

        full_answer = []
        async for token in self.client.chat_stream(
            system=ANSWER_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        ):
            sys.stdout.write(token)
            sys.stdout.flush()
            full_answer.append(token)

        print()
        return "".join(full_answer)

    # -------------------------------------------------------------------
    # Phase 5: REFLECT
    # -------------------------------------------------------------------

    async def _reflect(
        self, query: str, answer: str, conversation: Conversation
    ) -> None:
        turn_text = f"User: {query}\n\nAssistant: {answer}"

        # Key facts extraction
        try:
            kf_resp = await self.client.chat_completion(
                system=KEY_FACTS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": turn_text}],
                max_tokens=300,
                temperature=0.0,
            )
            kf_text = ""
            content = kf_resp.get("content", [])
            if isinstance(content, str):
                kf_text = content
            else:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        kf_text += block["text"]

            kf_json = kf_text.strip()
            if "```" in kf_json:
                kf_json = kf_json.split("```")[1]
                if kf_json.startswith("json"):
                    kf_json = kf_json[4:]
                kf_json = kf_json.strip()
            parsed = json.loads(kf_json)
            new_facts = KeyFacts(
                topic=parsed.get("topic"),
                entities=parsed.get("entities", []),
                user_intent=parsed.get("user_intent"),
                key_numbers=parsed.get("key_numbers", {}),
                conclusions=parsed.get("conclusions", []),
            )
            existing = conversation.get_key_facts()
            existing.merge(new_facts)
            conversation.set_key_facts(existing)
            if _debug:
                console.print(Panel(
                    f"Topic: [bold]{existing.topic}[/]\n"
                    f"Entities: [cyan]{', '.join(existing.entities[:8])}[/]\n"
                    f"Numbers: [green]{len(existing.key_numbers)}[/] | "
                    f"Conclusions: [green]{len(existing.conclusions)}[/]",
                    title="[bold]Phase 5: REFLECT — Key Facts[/]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                ))
        except Exception as exc:
            dbg(f"⚠️  Key facts extraction failed: {exc}", "yellow")

        # Progressive summary (after turn 2+)
        if conversation.turn_count >= 2:
            try:
                summary_input = ""
                if conversation.progressive_summary:
                    summary_input = f"Previous summary:\n{conversation.progressive_summary}\n\n"
                summary_input += f"Latest turn:\n{turn_text}"

                sum_resp = await self.client.chat_completion(
                    system=SUMMARIZE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": summary_input}],
                    max_tokens=400,
                    temperature=0.1,
                )
                sum_text = ""
                for block in sum_resp.get("content", []):
                    if block.get("type") == "text":
                        sum_text += block["text"]

                if sum_text:
                    conversation.progressive_summary = sum_text
                    dbg(f"📝 Progressive summary updated ({estimate_tokens(sum_text)} tokens)")
            except Exception as exc:
                dbg(f"⚠️  Summary update failed: {exc}", "yellow")
