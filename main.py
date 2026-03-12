#!/usr/bin/env python3
"""
ReAct Agent CLI — Interactive terminal chat with tool-augmented reasoning.

Author:  Adam Baudisch
Project: Finalni domaci ukol — AI Agenti s nastroji
Note:    Tento agent je zjednodusena cast vetsiho produkcniho systemu
         (KIT AI Platform), vyvijeneho na zaklade nove nabytych znalosti
         z kurzu. Ukazuje klicove koncepty: ReAct pattern, tool loop,
         budget management, semanticka pamet a context compression.

Usage:
    python main.py                  # interactive CLI
    python main.py "your question"  # single query mode
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from llm_client import AnthropicClient
from memory import Conversation, ConversationStore
from pipeline import ReActPipeline, set_debug

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------

theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "dim": "dim",
    "user": "bold cyan",
    "agent": "bold magenta",
    "cmd": "bold green",
    "heading": "bold white",
})

console = Console(theme=theme)

# ---------------------------------------------------------------------------
# Prompt toolkit setup
# ---------------------------------------------------------------------------

COMMANDS = ["n", "l", "o", "h", "f", "s", "d", "b", "q", "?",
            "/new", "/list", "/open", "/history", "/facts",
            "/summary", "/debug", "/budget", "/quit", "/help"]

pt_style = PTStyle.from_dict({
    "prompt": "bold #00d7ff",
    "rprompt": "#666666",
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class AgentCLI:
    """Interactive CLI for the ReAct agent with rich UI."""

    def __init__(self, client: AnthropicClient):
        self.client = client
        self.store = ConversationStore()
        self.conversation: Conversation | None = None
        self.debug = True
        history_path = os.path.expanduser("~/.react_agent/input_history")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        self.session = PromptSession(
            history=FileHistory(history_path),
            completer=WordCompleter(COMMANDS, sentence=True),
            style=pt_style,
        )

    # -------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------

    def _banner(self) -> None:
        title_text = Text()
        title_text.append("ReAct Agent", style="bold cyan")
        title_text.append(" — Custom Framework", style="dim")

        info = Text()
        info.append("Adam Baudisch", style="bold")
        info.append(" — Finalni domaci ukol\n", style="dim")
        info.append("Soucasti vetsiho projektu ", style="dim")
        info.append("KIT AI Platform", style="bold cyan")
        info.append(", vyvijeneho na zaklade\n", style="dim")
        info.append("nove nabytych znalosti z kurzu.\n\n", style="dim")
        info.append("Architektura:  ", style="bold")
        info.append("Route -> Think -> Tools -> Answer -> Reflect\n", style="cyan")
        info.append("Nastroje:      ", style="bold")
        info.append("Wikipedia, Calculator, DuckDuckGo\n", style="cyan")
        info.append("Pamet:         ", style="bold")
        info.append("Key Facts + Progressive Summary + JSON persistence", style="cyan")

        console.print()
        console.print(Panel(
            info,
            title=title_text,
            border_style="cyan",
            box=box.DOUBLE_EDGE,
            padding=(1, 2),
        ))

    def _help(self) -> None:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            padding=(0, 2),
            title="Prikazy",
            title_style="bold",
        )
        table.add_column("Klavesa", style="bold green", width=8)
        table.add_column("Prikaz", style="dim", width=12)
        table.add_column("Popis", width=40)

        table.add_row("?", "/help", "Zobrazit tuto napovedu")
        table.add_row("n", "/new", "Zalozit novou konverzaci")
        table.add_row("l", "/list", "Seznam ulozenych konverzaci")
        table.add_row("o N", "/open N", "Otevrit konverzaci (cislo nebo ID)")
        table.add_row("h", "/history", "Zobrazit historii zprav")
        table.add_row("f", "/facts", "Zobrazit extrahovana fakta")
        table.add_row("s", "/summary", "Zobrazit progressive summary")
        table.add_row("d", "/debug", "Prepnout debug mode (on/off)")
        table.add_row("b", "/budget", "Info o token budgetu")
        table.add_row("q", "/quit", "Ukoncit program")
        table.add_row("", "", "")
        table.add_row("", "", "[dim]Cokoliv jineho = dotaz pro agenta[/]")

        console.print()
        console.print(table)
        console.print()

    def _status_bar(self) -> HTML:
        if self.conversation:
            sid = self.conversation.id[-8:]
            t = self.conversation.turn_count
            tok = f"{self.conversation.total_tokens:,}"
            dbg = "ON" if self.debug else "OFF"
            return HTML(
                f'<style fg="#666666">[</style>'
                f'<style fg="#00d7ff">{sid}</style>'
                f'<style fg="#666666"> | {t} turns | {tok} tok | debug:{dbg}]</style>'
            )
        dbg = "ON" if self.debug else "OFF"
        return HTML(
            f'<style fg="#666666">[nova konverzace | debug:{dbg}]</style>'
        )

    def _ensure_conversation(self) -> Conversation:
        if not self.conversation:
            self.conversation = self.store.create("New conversation")
            console.print(f"  [success]+ Nova konverzace[/] {self.conversation.id[-8:]}")
        return self.conversation

    # -------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------

    def cmd_new(self, _arg: str = "") -> None:
        self.conversation = self.store.create("New conversation")
        console.print(f"  [success]+ Nova konverzace:[/] {self.conversation.id[-8:]}")

    def cmd_list(self, _arg: str = "") -> None:
        convs = self.store.list_all()
        if not convs:
            console.print("  [dim]Zadne ulozene konverzace.[/]")
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        table.add_column("#", style="bold cyan", width=3, justify="right")
        table.add_column("ID", style="dim", width=10)
        table.add_column("Nazev", width=35)
        table.add_column("Turny", justify="center", width=5)
        table.add_column("Aktualizovano", style="dim", width=14)
        table.add_column("", width=2)

        for i, c in enumerate(convs[:15], 1):
            updated = datetime.fromtimestamp(c.updated_at).strftime("%d.%m. %H:%M")
            short_id = c.id[-8:]
            active = "[green]<-[/]" if self.conversation and c.id == self.conversation.id else ""
            table.add_row(
                str(i), short_id, c.title[:35],
                str(c.turn_count), updated, active
            )

        console.print()
        console.print(table)
        console.print("  [dim]Otevri pomoci:[/] [green]o 1[/] [dim]nebo[/] [green]o <id>[/]")
        console.print()

    def cmd_open(self, arg: str = "") -> None:
        if not arg:
            console.print("  [error]Pouziti:[/] [green]o 1[/] [dim]nebo[/] [green]o <id>[/]")
            return
        convs = self.store.list_all()

        match = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(convs):
                match = convs[idx]
        except ValueError:
            pass

        if not match:
            for c in convs:
                if c.id.endswith(arg) or c.id == arg:
                    match = c
                    break

        if match:
            self.conversation = match
            console.print(f"  [success]Otevreno:[/] {match.title}")
            recent = match.messages[-4:]
            if recent:
                console.print()
                for msg in recent:
                    content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                    short = content[:120] + ("..." if len(content) > 120 else "")
                    if msg["role"] == "user":
                        console.print(f"    [user]Vy:[/] {short}")
                    else:
                        console.print(f"    [agent]Agent:[/] {short}")
                console.print()
        else:
            console.print(f"  [error]Nenalezeno '{arg}'.[/] Zkus [green]l[/] pro seznam.")

    def cmd_history(self, _arg: str = "") -> None:
        if not self.conversation or not self.conversation.messages:
            console.print("  [dim]Zadne zpravy.[/]")
            return

        console.print()
        console.rule(f"[bold]Historie — {self.conversation.title[:40]}[/]", style="dim")
        console.print()

        for i, msg in enumerate(self.conversation.messages):
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            turn = i // 2 + 1
            if msg["role"] == "user":
                console.print(f"  [user]\\[{turn}] Vy:[/] {content}")
            else:
                console.print(Panel(
                    Markdown(content[:500]),
                    border_style="magenta",
                    title=f"[agent]Agent — Turn {turn}[/]",
                    title_align="left",
                    box=box.ROUNDED,
                    padding=(0, 1),
                    width=min(console.width - 4, 80),
                ))
            console.print()

    def cmd_facts(self, _arg: str = "") -> None:
        if not self.conversation:
            console.print("  [dim]Zadna aktivni konverzace.[/]")
            return
        kf = self.conversation.get_key_facts()
        if not any([kf.topic, kf.entities, kf.key_numbers, kf.conclusions]):
            console.print("  [dim]Zatim zadna fakta.[/]")
            return

        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 2),
            title="Key Facts (semanticka pamet)",
            title_style="bold green",
        )
        table.add_column("Pole", style="bold", width=15)
        table.add_column("Hodnota", width=50)

        if kf.topic:
            table.add_row("Tema", kf.topic)
        if kf.user_intent:
            table.add_row("Zamer", kf.user_intent)
        if kf.entities:
            table.add_row("Entity", ", ".join(kf.entities[:10]))
        if kf.key_numbers:
            nums = "\n".join(f"{k}: {v}" for k, v in list(kf.key_numbers.items())[:8])
            table.add_row("Cisla", nums)
        if kf.conclusions:
            conc = "\n".join(f"- {c}" for c in kf.conclusions[:5])
            table.add_row("Zavery", conc)

        console.print()
        console.print(table)
        console.print()

    def cmd_summary(self, _arg: str = "") -> None:
        if not self.conversation:
            console.print("  [dim]Zadna aktivni konverzace.[/]")
            return
        if self.conversation.progressive_summary:
            console.print()
            console.print(Panel(
                self.conversation.progressive_summary,
                title="[bold]Progressive Summary[/]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2),
                width=min(console.width - 4, 80),
            ))
            console.print()
        else:
            console.print("  [dim]Summary se generuje po 2+ turnech.[/]")

    def cmd_debug(self, _arg: str = "") -> None:
        self.debug = not self.debug
        set_debug(self.debug)
        if self.debug:
            console.print("  [success]Debug ON[/] — vidis routing, think, tool cally, budget, reflect")
        else:
            console.print("  [warning]Debug OFF[/] — cisty vystup, jen odpoved")

    def cmd_budget(self, _arg: str = "") -> None:
        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 2),
            title="Token Budget System",
            title_style="bold cyan",
        )
        table.add_column("", style="bold", width=20)
        table.add_column("", width=40)

        table.add_row("Kontext okno", "200,000 tokens")
        table.add_row("Tool budget", "20,000 tokens per turn")
        table.add_row("Auto-stop", "pri 80% tool budgetu")
        table.add_row("Self-summarization", "komprese tool vysledku pri prekroceni")
        table.add_row("Response reserve", "4,096 tokens")
        if self.conversation:
            table.add_row("Celkem za session", f"{self.conversation.total_tokens:,} tokens")

        console.print()
        console.print(table)
        console.print()

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    async def run(self) -> None:
        self._banner()
        self._help()
        set_debug(self.debug)

        SLASH_MAP = {
            "/quit": "q", "/exit": "q", "/q": "q",
            "/new": "n", "/n": "n",
            "/list": "l", "/l": "l",
            "/open": "o", "/o": "o", "/continue": "o",
            "/history": "h", "/h": "h",
            "/facts": "f", "/f": "f",
            "/summary": "s", "/s": "s",
            "/debug": "d", "/d": "d",
            "/budget": "b", "/b": "b",
            "/help": "?", "?": "?",
        }

        CMD_MAP = {
            "n": self.cmd_new, "l": self.cmd_list, "o": self.cmd_open,
            "h": self.cmd_history, "f": self.cmd_facts, "s": self.cmd_summary,
            "d": self.cmd_debug, "b": self.cmd_budget,
        }

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.prompt(
                        "  > ",
                        rprompt=self._status_bar(),
                    )
                )
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Cau![/]")
                break

            if not user_input:
                continue

            # Parse command
            first_word = user_input.lower().split()[0]
            arg = user_input[len(first_word):].strip()

            resolved = SLASH_MAP.get(first_word, first_word if len(first_word) == 1 else None)

            if resolved == "q":
                console.print("  [dim]Cau![/]")
                break

            if resolved == "?":
                self._help()
                continue

            if resolved in CMD_MAP:
                CMD_MAP[resolved](arg)
                continue

            if user_input.startswith("/"):
                console.print(f"  [error]Neznamy prikaz.[/] Napis [green]?[/] pro napovedu.")
                continue

            # ── Query ─────────────────────────────────────────────
            conv = self._ensure_conversation()
            if conv.turn_count == 0:
                conv.title = user_input[:50]

            console.print()

            pipeline = ReActPipeline(self.client)
            try:
                answer = await pipeline.run(user_input, conv)
            except Exception as exc:
                console.print(f"  [error]Chyba: {exc}[/]")
                continue

            # Show answer in a nice panel
            console.print()
            console.print(Panel(
                Markdown(answer),
                border_style="magenta",
                title="[bold magenta]Odpoved[/]",
                title_align="left",
                box=box.ROUNDED,
                padding=(1, 2),
                width=min(console.width - 4, 80),
            ))

            conv.messages.append({"role": "user", "content": user_input})
            conv.messages.append({"role": "assistant", "content": answer})
            conv.turn_count += 1
            conv.total_tokens += pipeline.budget.total_used
            self.store.save(conv)

            # Post-answer hint
            console.print(
                f"  [dim]Pokracuj dalsim dotazem, nebo:[/] "
                f"[green]f[/][dim]=fakta[/]  "
                f"[green]s[/][dim]=summary[/]  "
                f"[green]d[/][dim]=debug[/]  "
                f"[green]?[/][dim]=vse[/]"
            )
            console.print()


async def single_query(client: AnthropicClient, query: str) -> None:
    """Single query mode."""
    store = ConversationStore()
    conv = store.create(query[:50])
    pipeline = ReActPipeline(client)
    answer = await pipeline.run(query, conv)

    console.print()
    console.print(Panel(
        Markdown(answer),
        border_style="magenta",
        title="[bold magenta]Odpoved[/]",
        title_align="left",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    conv.messages.append({"role": "user", "content": query})
    conv.messages.append({"role": "assistant", "content": answer})
    conv.turn_count = 1
    conv.total_tokens = pipeline.budget.total_used
    store.save(conv)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("MODEL", "claude-sonnet-4-20250514")

    if not api_key:
        console.print("[error]Chyba: ANTHROPIC_API_KEY neni nastaveny.[/]")
        console.print("Zkopiruj .env.example do .env a pridej klic.")
        sys.exit(1)

    client = AnthropicClient(api_key=api_key, model=model)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query(client, query))
    else:
        asyncio.run(AgentCLI(client).run())


if __name__ == "__main__":
    main()
