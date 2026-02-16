# Squawk AI - Root Cause Analysis Engine for Aircraft Maintenance

Squawk AI is a conversational diagnostic engine that performs differential diagnosis on pilot-reported aircraft issues. It interviews the pilot, builds and refines a fault tree, and produces a report with likely root causes and recommended maintenance actions.

The pilot describes a problem, the engine generates hypotheses, asks follow-up questions to narrow them down, and delivers a write-up to maintenance. All of this is grounded in aircraft manuals, historical maintenance data, and web research.

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

The core loop is a LangGraph state machine with 11 nodes. The interview is adaptive: questions come from the current fault tree state, not a static script. The system stops when root causes converge or diagnostic coverage is sufficient.

## Architecture

Python, FastAPI, LangGraph, Supabase (PostgreSQL + pgvector), OpenAI/Anthropic LLMs

```
main.py          - FastAPI server, conversation lifecycle, polling API
graph.py         - LangGraph workflow: 11 nodes, state machine, diagnostic loop
prompt.py        - All LLM prompts (fault tree, questions, updates, decisions, summary)
llm_select.py    - Model selection + custom embedding clients (OpenAI, HuggingFace)
config.py        - Supabase client initialization
db_logging.py    - Conversation and message persistence
utils.py         - Retry logic, Tavily web search, email delivery
```

### Design Decisions

- **Fault Tree Analysis (FTA)** as the reasoning structure instead of free-form chat. The LLM builds a JSON fault tree with typed nodes (intermediate, basic, undeveloped) and confidence levels. This constrains the LLM to produce structured, actionable output.
- **Separate decision node** for interview completion. Instead of letting the question-generation LLM decide when to stop, a dedicated node evaluates fault tree convergence against a diagnostic coverage checklist.
- **Multi-source RAG** that combines aircraft manual embeddings, fleet-specific historical squawks, and live web search into a single research bundle before building hypotheses.
- **Interrupt-based human-in-the-loop.** LangGraph's interrupt mechanism pauses the graph at specific nodes to collect pilot input. The client uses polling to pick up new messages.

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/create` | Start a new diagnostic session |
| `GET` | `/poll/{id}` | Get current messages and state |
| `POST` | `/reply/{id}` | Submit pilot's response |
| `POST` | `/feedback/{id}` | Thumbs up/down on the final report |
| `GET` | `/health` | Health check with component status |

Conversations are identified by UUID. The client polls for new messages and renders based on UI hints (`pill-select` for aircraft selection, `status` for progress updates, `running` for loading states).

## Data Model

Three Supabase projects handle different concerns:

- **Vector Store** - Aircraft manual chunks with pgvector embeddings + historical squawk embeddings. Cosine similarity search via a custom `execute_sql` RPC.
- **Aircraft Config** - Aircraft overviews (systems, specs, checklists), org-specific tail numbers and configurations.
- **Conversation Log** - Conversation state, messages, diagnostic summaries, and feedback.

## Running Locally

```bash
git clone https://github.com/Squawk-AI/squawk-ai-be.git
cd squawk-ai-be
pip install -r requirements.txt

cp .env.example .env
# fill in API keys and database credentials

uvicorn main:app --reload
```

Requires Python 3.11+. See `.env.example` for all required environment variables.

## Monitoring

- Sentry for error tracking with conversation-level context tags
- LangSmith for LLM call tracing and prompt observability
- Structured logging with operation-level context

## Status

Working proof of concept, deployed and tested with real aircraft types and maintenance workflows. Future work:

- Auth and multi-tenancy
- Streaming responses (currently poll-based)
- Parameterized queries for the vector search layer
- Test coverage
