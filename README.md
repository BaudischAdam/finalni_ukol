# ReAct Agent — Custom Framework

**Adam Baudisch — Finalni domaci ukol**

Soucasti vetsiho projektu **KIT AI Platform**, vyvijeneho na zaklade nove nabytych znalosti z kurzu.

---

## Co to je

Vlastni implementace ReAct (Reasoning + Acting) agenta postavena primo nad Anthropic Claude API **bez pouziti frameworku** (zadny LangChain, LlamaIndex apod.).

Agent demonstruje:
- **5-fazovy pipeline**: Route → Think → Tools → Answer → Reflect
- **Nativni tool_use**: primo Claude API `tool_use` / `tool_result` bloky
- **Token budget system**: sledovani spotřeby, auto-stop pri 80%, self-summarization
- **Semanticka pamet**: Key Facts extraction (topic, entities, key_numbers, conclusions)
- **Progressive summary**: komprese historie konverzace pro dlouhe session
- **Conversation persistence**: JSON soubory (simulace databaze)
- **Streaming odpovedi**: SSE streaming z Claude API
- **Query routing**: klasifikace dotazu (simple vs reasoning) pro efektivitu

### Nastroje

| Nastroj | Popis |
|---------|-------|
| `wikipedia_search` | Hledani clanku na Wikipedii |
| `wikipedia_get_article` | Ziskani uvodu clanku podle presneho nazvu |
| `calculator` | Bezpecne vyhodnoceni matematickych vyrazu (sandboxed) |
| `web_search` | DuckDuckGo instant answers API |

---

## Architektura

```
User query
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1: ROUTE                         │
│  LLM klasifikuje: simple / reasoning    │
│  + kontext z Key Facts & Summary        │
└──────────┬─────────────┬────────────────┘
           │             │
     simple│             │reasoning
           ▼             ▼
   ┌──────────┐  ┌─────────────────────┐
   │ Streaming │  │ Phase 2: THINK      │
   │ odpoved   │  │ LLM planuje tooly   │
   └──────────┘  └────────┬────────────┘
                          ▼
                 ┌─────────────────────┐
                 │ Phase 3: TOOL LOOP  │
                 │ Iterativni volani   │
                 │ nastroju s budget   │
                 │ + self-summarize    │
                 └────────┬────────────┘
                          ▼
                 ┌─────────────────────┐
                 │ Phase 4: ANSWER     │
                 │ Streaming odpoved   │
                 │ s celym kontextem   │
                 └────────┬────────────┘
                          ▼
┌─────────────────────────────────────────┐
│  Phase 5: REFLECT                       │
│  Key Facts extraction + Progressive     │
│  summary update                         │
└─────────────────────────────────────────┘
```

### Soubory

| Soubor | Popis |
|--------|-------|
| `main.py` | CLI rozhrani (rich + prompt_toolkit) |
| `pipeline.py` | 5-fazovy ReAct pipeline |
| `tools.py` | Definice nastroju (Wikipedia, Calculator, DuckDuckGo) |
| `memory.py` | Konverzacni persistence + Key Facts |
| `budget.py` | Token budget tracker |
| `llm_client.py` | Anthropic Claude API klient (httpx) |

---

## Instalace a spusteni

### Pozadavky

- Python 3.11+
- Anthropic API klic ([console.anthropic.com](https://console.anthropic.com))

### Setup

```bash
# 1. Vytvor virtualni prostredi
python -m venv .venv

# 2. Aktivuj
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Nainstaluj zavislosti
pip install -r requirements.txt

# 4. Nastav API klic
cp .env.example .env
# Uprav .env a vloz svuj ANTHROPIC_API_KEY
```

### Spusteni

```bash
python main.py
```

---

## Ovladani

| Klavesa | Prikaz | Popis |
|---------|--------|-------|
| `?` | `/help` | Zobrazit napovedu |
| `n` | `/new` | Nova konverzace |
| `l` | `/list` | Seznam konverzaci |
| `o N` | `/open N` | Otevrit konverzaci (cislo nebo ID) |
| `h` | `/history` | Historie zprav |
| `f` | `/facts` | Extrahovana fakta (Key Facts) |
| `s` | `/summary` | Progressive summary |
| `d` | `/debug` | Prepnout debug mode (on/off) |
| `b` | `/budget` | Info o token budgetu |
| `q` | `/quit` | Ukoncit |

Cokoliv jineho = dotaz pro agenta.

### Debug mode

Ve vychozim stavu je debug mode **zapnuty** — zobrazuje vsechny interni faze pipeline (ROUTE, THINK, tool calls, REFLECT, budget). Prepnes prikazem `d`.

---

## Ukazka pouziti

```
> najdi mi co znamená drop-shipping

╭─ Phase 1: ROUTE ─╮
│ REASONING — ...   │
╰───────────────────╯
╭─ Phase 2: THINK ─╮
│ Plan (2 steps):   │
│ 1. web_search ... │
│ 2. wikipedia ...  │
╰───────────────────╯
  🔧 web_search({"query": "drop-shipping"})
  🔧 wikipedia_get_article({"title": "Drop shipping"})
  🔄 Tool loop done — 3 iteration(s)

╭─ Agent ──────────────────────────╮
│ **Drop-shipping** je obchodní    │
│ model, při kterém prodejce ...   │
╰──────────────────────────────────╯

> l
╭───┬──────────┬──────────────────────────────┬───────╮
│ # │ ID       │ Nazev                        │ Turny │
├───┼──────────┼──────────────────────────────┼───────┤
│ 1 │ 98-15026 │ najdi mi co znamená drop-... │   1   │
╰───┴──────────┴──────────────────────────────┴───────╯
```

---

## Technicky popis klicovych konceptu

### ReAct pattern
Agent strida **reasoning** (analyza dotazu, planovani) s **acting** (volani externich nastroju). Kazda iterace tool loopu: LLM rozhodne co zavolat → tool se provede → vysledek se vlozi zpet do kontextu → LLM rozhodne dal.

### Token budget system
Sleduje spotrebu tokenu v realnem case. Pri dosazeni 80% tool budgetu (20k tokenu) automaticky zastavi tool loop a spusti **self-summarization** — LLM zkomprimuje dosavadni tool vysledky do kratkeho summary, aby se usetrilo misto pro odpoved.

### Key Facts extraction
Po kazdem turnu LLM extrahuje strukturovana fakta (topic, entities, key_numbers, conclusions). Tato fakta se predavaji do dalsich turnu jako kontext — agent si tak "pamatuje" co je dulezite bez nutnosti preposilat celou historii.

### Progressive summary
Od 2. turnu se po kazdem turnu aktualizuje komprimovany souhrn cele konverzace. Nahrazuje starsi zpravy → udrzuje kontext v rozumne velikosti i pri dlouhych konverzacich.

### Native Claude tool_use
Zadny framework — primo Anthropic Messages API:
- `tool_use` bloky: Claude rozhodne jaky tool zavolat a s jakymi parametry
- `tool_result` bloky: vysledky nastroju se vloži zpet do zprav
- Iterativni loop: az 8 iteraci, kazda muze zavolat vice toolu naraz

---

## Souvislost s KIT AI Platform

Tento agent je zjednodusena verze produkčního reasoning pipeline z projektu **KIT AI Platform** (on-premise AI server pro technickou podporu). Produkční verze navíc obsahuje:
- RAG pipeline s vektorovou databází
- Multi-tier inference (rychlý 7B / střední 14B / těžký 32B+ model)
- MCP (Model Context Protocol) integrace
- PostgreSQL persistence
- Governed Learning systém

Tento homework agent demonstruje jádro architektury — ReAct pattern, tool_use, context management — v samostatné, spustitelné podobě.

---

## Licence

Skolni projekt — pouze pro vzdelavaci ucely.