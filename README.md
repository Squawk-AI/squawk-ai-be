# Squawk AI — Automated Root Cause Analysis for Aircraft Maintenance

Squawk AI is a conversational diagnostic engine that performs **differential diagnosis** on pilot-reported aircraft issues. It conducts a structured interview with the pilot, builds and iteratively refines a **fault tree**, and produces a diagnostic report with likely root causes and recommended maintenance actions.

Think of it as a domain-specific reasoning system: the pilot describes a problem, the engine generates hypotheses, asks targeted follow-up questions to narrow them down, and delivers a structured write-up to maintenance — all grounded in aircraft manuals, historical maintenance data, and web research.

## How It Works

```
Pilot reports issue
        |
   [ RAG Research ]
   - Vector search aircraft manuals (pgvector)
   - Query past squawks from fleet history
   - Web search for known issues (Tavily)
        |
   [ Build Fault Tree ]
   LLM generates a hierarchical fault tree
   with intermediate and basic failure modes
        |
   [ Diagnostic Interview Loop ]
   - Generate prioritized follow-up questions
   - Pilot answers
   - Update fault tree confidence levels
   - LLM decides: enough info, or keep asking?
        |
   [ Generate Report ]
   - Diagnostic memo with likely root causes
   - Recommended maintenance actions with manual references
   - Email to maintenance team
```

The core loop is a **LangGraph state machine** with 11 nodes. The interview is adaptive — questions are generated based on the current fault tree state, not from a static script. The system knows when to stop (convergent confidence on root causes, or sufficient diagnostic coverage).

## Architecture

**Stack:** Python, FastAPI, LangGraph, Supabase (PostgreSQL + pgvector), OpenAI / Anthropic LLMs

```
main.py          — FastAPI server, conversation lifecycle, polling API
graph.py         — LangGraph workflow: 11 nodes, state machine, diagnostic loop
prompt.py        — All LLM prompts (fault tree, questions, updates, decisions, summary)
llm_select.py    — Model selection + custom embedding clients (OpenAI, HuggingFace)
config.py        — Supabase client initialization
db_logging.py    — Conversation and message persistence
utils.py         — Retry logic, Tavily web search, email delivery
```

### Key Design Decisions

- **Fault Tree Analysis (FTA)** as the reasoning structure, not free-form chat. The LLM builds a JSON fault tree with typed nodes (intermediate, basic, undeveloped) and confidence levels. This constrains the LLM's reasoning to produce actionable, structured output.
- **Separate decision node** to determine interview completion. Rather than letting the question-generation LLM decide when to stop, a dedicated node evaluates fault tree convergence against a diagnostic coverage checklist.
- **Multi-source RAG** combines aircraft manual embeddings, fleet-specific historical squawks (past maintenance records), and live web search into a single research bundle before hypothesis generation.
- **Interrupt-based human-in-the-loop.** LangGraph's interrupt mechanism pauses the graph at specific nodes to wait for pilot input, rather than polling or websockets. The API uses a poll-based pattern for the client.

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/create` | Start a new diagnostic session |
| `GET` | `/poll/{id}` | Get current messages and state |
| `POST` | `/reply/{id}` | Submit pilot's response |
| `POST` | `/feedback/{id}` | Thumbs up/down on the final report |
| `GET` | `/health` | Health check with component status |

Conversations are identified by UUID. The client polls for new messages and renders UI hints (`pill-select` for aircraft selection, `status` for progress updates, `running` for loading states).

## Data Model

Three Supabase projects handle different concerns:

- **Vector Store** — Aircraft manual chunks with pgvector embeddings + historical squawk embeddings. Supports cosine similarity search via a custom `execute_sql` RPC.
- **Aircraft Config** — Aircraft overviews (systems, specs, checklists), org-specific tail numbers and configurations.
- **Conversation Log** — Conversation state, individual messages, diagnostic summaries, and user feedback.

## Running Locally

```bash
# Clone and install
git clone https://github.com/natalyarockets/SquawkAI_POCnb.git
cd SquawkAI_POCnb
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in your API keys and database credentials

# Run
uvicorn main:app --reload
```

Requires Python 3.11+. See `.env.example` for all required environment variables.

## Monitoring

- **Sentry** for error tracking with conversation-level context tags
- **LangSmith** for LLM call tracing and prompt observability
- Structured logging throughout with operation-level context

## Status

This is a working proof of concept built for a real aviation maintenance use case. It's deployed and tested with actual aircraft types and maintenance workflows. Areas for future work:

- Authentication and multi-tenancy
- Streaming responses (currently poll-based)
- Parameterized queries for the vector search layer
- Expanded test coverage
