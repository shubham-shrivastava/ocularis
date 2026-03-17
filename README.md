# Ocularis

An open-source autonomous browsing agent that can take a goal, search the web, extract and compare results, and return a structured answer.

Vision models can see and click -- but they get stuck constantly. Cookie popups, loading spinners, the same button clicked five times with zero progress. The model is not the hard part. Reliability is.

Ocularis wraps a vision model in an orchestration layer that handles supervision, recovery, planning, extraction, and reasoning -- so the agent can go from a vague goal to a useful result.

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Find best 1TB Samsung SSD", "start_url": "https://amazon.in", "browser_mode": "launch"}'
```

The agent opens a browser, searches, extracts products using an LLM, compares them, picks the best match, and returns a structured answer with links, prices, and ratings.

![Ocularis Dashboard](assets/ui.png)

---

## What it does

**Goal Planner** — Decomposes high-level goals into concrete sub-steps (search, scroll, extract, compare). Each sub-step has a postcondition that is verified before moving on.

**Context-Aware Browsing** — Understands intent (comparison, summarization, extraction), classifies page types (search results, product listing, article), and extracts structured data. Persists comparison state across steps.

**LLM-Powered Extraction** — Raw page blocks are extracted via JavaScript, then an LLM identifies actual products, filters out navigation links, ads, and irrelevant brands, and returns structured data with title, price, and rating.

**Text Reasoning** — A dedicated reasoning layer compares extracted candidates, selects the best match based on the user's criteria, and generates a human-readable answer with confidence scores.

**Reliability Supervisor** — Hashes screenshots before and after every action. MD5 for speed, SSIM for perceptual accuracy. Stuck three times? Auto-recover with a cycling strategy (scroll, refresh, go back, offset click). Exhausted all strategies? Escalate to a human.

**Episodic Memory** — Embeds every step as a text summary in pgvector. Before acting, asks "what worked last time on a screen like this?" Recall is similarity-based, not keyword-based.

**Critic Loop** — After every action, sends before/after screenshots back to the model: "did the page actually change toward the goal?" Structured result: did it progress, what blocked it, what to try next.

**Goal Evaluation** — A dedicated evaluator asks the model "is the task done?" after every successful step. Terminates the loop with high confidence or escalates to a human when uncertain.

**Human-in-the-Loop** — Dashboard with pause/resume controls. When the circuit breaker fires or a sensitive URL pattern is matched, the run pauses and waits for a human action before continuing.

---

## Architecture

```
POST /run  →  RunSession (per-run isolation)
                 │
                 ├── BrowserManager (Playwright, launch or connect mode)
                 ├── GoalPlanner (decomposes goal → sub-steps with postconditions)
                 ├── Supervisor (MD5/SSIM stuck detection + circuit breaker)
                 ├── ModelClient (actor → critic → goal check)
                 ├── ContentExtractor (JS block extraction → LLM product identification)
                 ├── ReasoningExecutor (compare candidates → pick best → structured answer)
                 ├── GoalEvaluator
                 └── Memory (NullMemory → EpisodicMemory with pgvector)
```

Every run is isolated in its own `RunSession`. No state leaks across concurrent runs.

### Data flow (single run)

```mermaid
flowchart TD
  Goal["POST /run"] --> CreateRun["Create RunRecord status=queued"]
  CreateRun --> Session["Create RunSession"]
  Session --> Loop["Orchestration Loop"]
  Loop --> GetState["BrowserManager.get_state"]
  GetState --> Recall["Memory.recall"]
  Recall --> AskModel["ModelClient.predict"]
  AskModel --> PreCheck["Supervisor.pre_action_check"]
  PreCheck -->|stuck| Recover["Recovery strategy"]
  PreCheck -->|healthy| Execute["BrowserManager.execute_action"]
  Recover --> Execute
  Execute --> PostState["BrowserManager.get_state"]
  PostState --> Verify["Supervisor.post_action_verify + Critic"]
  Verify -->|progress| EvalGoal["GoalEvaluator.evaluate"]
  Verify -->|no progress| CircuitBreaker{"Retries exceeded?"}
  CircuitBreaker -->|yes| Halt["waiting_for_human"]
  CircuitBreaker -->|no| Loop
  EvalGoal -->|achieved| Complete["completed"]
  EvalGoal -->|not achieved| MaxCheck{"max_steps?"}
  MaxCheck -->|yes| Exceeded["max_steps_exceeded"]
  MaxCheck -->|no| Loop
```

---

## Getting started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Docker](https://docs.docker.com/get-docker/) (for PostgreSQL + pgvector)
- Node.js 18+ (only for the dashboard)

### Quick start

```bash
# 1. Start the database
docker compose up -d

# 2. Install Python dependencies
uv sync
uv run playwright install chromium

# 3. Start the API server
uv run uvicorn api.main:app --reload
```

API at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Start the dashboard (optional)

```bash
cd web/dashboard
npm install
npm run dev
```

Dashboard at `http://localhost:3000`.

---

## Running with a real model

Ocularis starts in **development mode** (MockModelClient) by default. To use a real vision model, you need an OpenAI-compatible vision endpoint. UI-TARS by ByteDance is the recommended model.

### Choosing a model

| Model | HuggingFace ID | Memory | Notes |
|---|---|---|---|
| **UI-TARS 1.5 7B** | `ByteDance-Seed/UI-TARS-1.5-7B` | ~16 GB | **Recommended.** Latest publicly available. Better grounding than v1. |
| UI-TARS 7B DPO | `ByteDance-Seed/UI-TARS-7B-DPO` | ~16 GB | Original v1. Superseded by 1.5. |
| UI-TARS 72B DPO | `ByteDance-Seed/UI-TARS-72B-DPO` | ~140 GB | Best v1 accuracy. Needs multi-GPU. |
| UI-TARS 2B SFT | `ByteDance-Seed/UI-TARS-2B-SFT` | ~6 GB | Smallest. Limited accuracy. |

UI-TARS 2 was announced (Sep 2025) and outperforms all of the above, but is not yet released on HuggingFace.

### Option A: Ollama (easiest, works on Mac)

Best option for Apple Silicon Macs. Ollama serves an OpenAI-compatible API locally.

```bash
# Install Ollama: https://ollama.com
ollama pull 0000/ui-tars-1.5-7b
ollama serve
```

The model is ~15 GB. On an M4 Pro with 48 GB unified memory, it runs comfortably with room to spare.

Then configure Ocularis:

```yaml
# config.yaml
model:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
  model_name: "0000/ui-tars-1.5-7b"
```

### Option B: vLLM on NVIDIA GPU (best throughput)

vLLM downloads the model automatically on first launch and serves an OpenAI-compatible API.

```bash
pip install vllm

vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --served-model-name ui-tars \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --limit-mm-per-prompt image=5

# For 72B on multi-GPU:
# vllm serve ByteDance-Seed/UI-TARS-72B-DPO \
#   --served-model-name ui-tars \
#   --host 0.0.0.0 --port 8001 \
#   --trust-remote-code --limit-mm-per-prompt image=5 \
#   -tp 4
```

### Option C: vLLM on Apple Silicon (via vllm-metal)

If you prefer vLLM over Ollama on Mac, use the `vllm-metal` plugin which runs on MLX:

```bash
# See https://github.com/vllm-project/vllm-metal for install
vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --served-model-name ui-tars \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code
```

Note: vllm-metal vision model support may still be experimental. Ollama is the safer bet on Mac for now.

### Configuring Ocularis for a local model

For vLLM (Options B and C):

```yaml
# config.yaml
model:
  base_url: "http://localhost:8001/v1"
  api_key: "not-needed-for-local"
  model_name: "ui-tars"
```

If you want to pre-download weights (e.g. for offline use):

```bash
pip install huggingface_hub
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./models/UI-TARS-1.5-7B
```

### Option D: Remote API (any OpenAI-compatible vision endpoint)

If you're running the model on a remote server or using a cloud provider, set the endpoint and key in `config.yaml`:

```yaml
model:
  base_url: "https://your-model-api.example.com/v1"
  api_key: "sk-..."
  model_name: "your-model"
```

Or via environment variables:

```bash
export OCULARIS_MODEL__BASE_URL="https://your-model-api.example.com/v1"
export OCULARIS_MODEL__API_KEY="sk-..."
export OCULARIS_MODEL__MODEL_NAME="your-model"
```

The API server automatically uses `APIModelClient` when `model.api_key` is set. When no key is configured, it falls back to `MockModelClient` for local development.

---

## Environment variables

API keys are auto-resolved from two environment variables based on each service's `base_url`:

| Variable | Used for | When |
|---|---|---|
| `OPENROUTER_API_KEY` | Vision model, text model | When `base_url` points to `openrouter.ai` |
| `OPENAI_API_KEY` | Planner, episodic memory embeddings | When `base_url` points to `api.openai.com` |

Set them in a `.env` file (gitignored) or export them directly. No need to configure separate keys per service.

All other settings from `config.yaml` can be overridden via environment variables using the prefix `OCULARIS_` with double underscores for nesting:

| Variable | Description | Default |
|---|---|---|
| `OCULARIS_MODEL__BASE_URL` | Vision model API endpoint | `https://openrouter.ai/api/v1` |
| `OCULARIS_MODEL__MODEL_NAME` | Model name to use | `bytedance/ui-tars-1.5-7b` |
| `OCULARIS_DB__URL` | PostgreSQL connection string | `postgresql+asyncpg://ocularis:ocularis@localhost:5432/ocularis` |

---

## Browser modes

**Launch mode** (default) — Spawns a fresh Chromium instance via Playwright. Isolated, clean, no user data.

**Connect mode** — Attaches to an existing Chrome browser via CDP. Gives access to user profiles, cookies, and logged-in sessions.

```bash
# Start Chrome with remote debugging (helper script included)
./scripts/start_chrome_debug.sh

# Then use connect mode in your run request:
curl -X POST http://localhost:8000/run -H 'Content-Type: application/json' -d '{
  "goal": "Check my email",
  "start_url": "https://mail.google.com",
  "browser_mode": "connect",
  "cdp_url": "http://127.0.0.1:9222"
}'
```

---

## Security

- **Domain allowlist** enforced at `BrowserContext` level — covers all pages, popups, new tabs, and iframes. Empty list = allow all.
- **Sensitive URL patterns** (`confirm_patterns`) pause the run and require human approval before the action executes.
- **Password field redaction** — bounding boxes of `input[type=password]` are blacked out before the screenshot reaches the model or storage.
- **CDP binds to `127.0.0.1` only** — no remote debugging exposure.
- **Ephemeral mode** — no screenshots written to disk, no embeddings stored. Use for sensitive sessions.
- **Local model inference** recommended for tasks involving sensitive data (see vLLM setup above).

---

## Configuration

Key sections in `config.yaml`:

```yaml
browser:
  headless: false             # true for server deployments
  viewport_width: 1280
  viewport_height: 800

model:
  base_url: "http://localhost:8000/v1"
  api_key: ""                 # set to enable real model
  model_name: "ui-tars"

supervisor:
  stuck_threshold: 3          # identical hashes before declaring stuck
  ssim_similarity_floor: 0.98 # SSIM score above this = visually identical
  max_retries: 3              # before circuit breaker fires

security:
  allowed_domains: []         # empty = allow all; ["example.com"] to restrict
  confirm_patterns: []        # URL patterns requiring human approval
  block_password_fields: true

goal_evaluator:
  confidence_threshold: 0.8   # min model confidence to accept "achieved"

memory:
  enabled: false              # set true + OPENAI_API_KEY for episodic memory
  embedding_model: "text-embedding-3-small"
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/run` | Start a new agent run (returns `run_id` + `queued` status) |
| `GET` | `/runs` | List all runs (optional `?status=` filter) |
| `GET` | `/runs/{id}` | Get run details + all steps |
| `GET` | `/runs/{id}/replay` | Replay a completed run |
| `POST` | `/runs/{id}/pause` | Pause an active run |
| `POST` | `/runs/{id}/resume` | Resume a paused run |
| `POST` | `/runs/{id}/intervene` | Send a manual action to a waiting run |
| `GET` | `/runs/{id}/steps/{n}/screenshot?phase=pre\|post` | Serve step screenshot |
| `WS` | `/ws/runs/{id}` | Live stream of step traces |
| `GET` | `/health` | Health check + active run count |

---

## Folder structure

```
ocularis/
  core/
    browser_manager.py    # Playwright wrapper (launch + connect modes)
    model_client.py       # ModelClientProtocol + APIModelClient + MockModelClient
    run_session.py        # Per-run state container + orchestration loop + RunRegistry
    schemas.py            # All Pydantic models (internal + wire types)
    settings.py           # pydantic-settings loader from config.yaml + env key resolution
    logging_setup.py      # Loguru structured JSON config
  logic/
    supervisor.py         # MD5/SSIM stuck detection, recovery, circuit breaker
    goal_evaluator.py     # Goal completion checker
    goal_planner.py       # Decomposes goals into sub-steps with postconditions
    intent_resolver.py    # Classifies user intent (compare, summarize, extract)
    page_classifier.py    # Identifies page type (search results, product listing, article)
    content_extractor.py  # JS block extraction + LLM-based product identification
    reasoning_executor.py # Compares candidates, selects best match, generates answer
    text_reasoner.py      # LLM abstraction for text/JSON generation
    memory.py             # MemoryProtocol + NullMemory + EpisodicMemory
    prompts.py            # All prompt templates
  api/
    main.py               # FastAPI app, all endpoints, WebSocket
    dependencies.py       # FastAPI dependency injection
  db/
    models.py             # SQLAlchemy + pgvector ORM models
    repository.py         # CRUD + vector similarity search
    engine.py             # Async engine init + session factory
  scripts/
    start_chrome_debug.sh # Safe CDP launcher for connect mode
  web/
    dashboard/            # Next.js 14 dashboard with live view + result display
  tests/
    test_supervisor.py
    test_goal_evaluator.py
    test_run_session.py
    test_browser_manager.py
    test_browser_manager_s2.py
    test_context_aware.py
    test_memory.py
    test_orchestration.py
    test_api_endpoints.py
  config.yaml
  docker-compose.yml
  pyproject.toml
```

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Browser | Playwright (Chromium) |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` |
| Frontend | Next.js 14 App Router + Tailwind |
| Package management | uv |
| Logging | Loguru (structured JSON) |
| Validation | Pydantic v2 |

---

## Run the tests

```bash
uv run pytest
```

All tests run without a browser, database, or model server. `MockModelClient` is used throughout.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details. Contributions welcome.
