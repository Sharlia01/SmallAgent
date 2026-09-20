# RAGSys Agent

RAGSys Agent is a full-stack, agent-driven question-answering system for personal knowledge bases. It decides when to use tools, searches private documents or the live web, and produces streaming answers grounded in conversation history and retrieved evidence.

Retrieval-Augmented Generation (RAG) is one of the Agent's core tools rather than the complete definition of the system.

## Features

- User registration, authentication, and session-level data isolation
- Document upload, parsing, retrieval, and deletion
- Agent-directed selection between private knowledge-base retrieval and live web search
- Combined use of knowledge-base and web evidence in a single task
- Streaming answers with traceable citations
- Bounded multi-turn conversation context
- Support for PDF, DOCX, TXT, Excel, PowerPoint, HTML, and Markdown
- An offline RAG evaluation dataset and retrieval metrics

## Architecture

```text
User question + conversation history + session document
                           │
                           ▼
                   Agent Planner / Router
                           │
                 ┌─────────┼─────────┐
                 │         │         │
                 ▼         ▼         ▼
        Knowledge-base RAG  Web search  No tool
                 │         │         │
                 └─────────┴─────────┘
                           │
                           ▼
                 Normalized evidence
                           │
                           ▼
                     Answer model
                           │
                           ▼
               Streaming answer + citations
```

The Agent is responsible for retrieval decisions and tool execution. A separate answer model synthesizes the final response. This planner-executor-synthesizer design allows a smaller model with reliable function calling to handle routing while a more capable model focuses on answer quality.

## Model configuration

| Responsibility | Configuration | Current default |
| --- | --- | --- |
| Agent routing and tool calls | `AGENT_MODEL` | `qwen3.7-flash-2026-07-15` |
| Retrieval query normalization | `QUERY_REWRITE_MODEL` | `qwen3.7-flash-2026-07-15` |
| Evidence sufficiency check | `RAG_EVIDENCE_SUFFICIENCY_MODEL` | `qwen3.7-flash-2026-07-15` |
| Final answer generation | `CHAT_MODEL` | `deepseek-v4-pro` |
| Live web search | `WEB_SEARCH_MODEL` | `qwen-plus` |
| Suggested questions and session titles | Configured in code | `qwen3.7-flash-2026-07-15` |
| Text embeddings | Local model directory | `bge-small-zh-v1.5` |
| Retrieval reranking | Local model directory | `bge-reranker-v2-m3` |

Both `AGENT_MODEL` and `CHAT_MODEL` are currently set to `deepseek-v4-pro` in `.env.example`. To use a smaller model for routing and a larger model for answer synthesis, configure them separately in `backend/.env`:

```env
AGENT_MODEL=<function-calling router model>
CHAT_MODEL=deepseek-v4-pro
```

## Technology stack

- **Frontend:** React, TypeScript, Vite, and Ant Design
- **Backend:** FastAPI and Python
- **Data services:** PostgreSQL, Elasticsearch, and Redis
- **Agent:** An OpenAI-compatible function-calling loop
- **Embeddings:** Local `bge-small-zh-v1.5`
- **Reranking:** Local `bge-reranker-v2-m3`
- **LLM:** Alibaba Cloud DashScope

## Getting started

### Prerequisites

- Docker and Docker Compose
- Node.js and npm (only when using the optional frontend)
- A local `bge-small-zh-v1.5` model directory
- A local `bge-reranker-v2-m3` model directory
- A DashScope API key

### 1. Start the backend

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env`, add your DashScope API key, and replace the example PostgreSQL, Elasticsearch, and JWT secrets:

```env
DASHSCOPE_API_KEY="your-api-key"
BGE_MODEL_HOST_PATH=../../models/bge-small-zh-v1.5
RERANKER_MODEL_HOST_PATH=../../models/bge-reranker-v2-m3
POSTGRES_PASSWORD="replace-with-a-strong-password"
ELASTIC_PASSWORD="replace-with-a-strong-password"
JWT_SECRET_KEY="replace-with-a-long-random-string"
```

The model host paths are resolved relative to `backend/docker-compose.yml`.
Their defaults point to `Projects/models/bge-small-zh-v1.5` and
`Projects/models/bge-reranker-v2-m3`. Switching the reranker does not require
rebuilding the existing 512-dimensional embedding index.

After reranking, the final Top chunks are checked for complete evidence coverage.
The checker requires every requested metric, time range, qualifier, and sub-question
to be directly supported. Insufficient evidence is removed before answer generation.
Set `RAG_EVIDENCE_SUFFICIENCY_ENABLED=false` to disable this gate. Provider failures
fail closed by default; set `RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN=true` only when
availability is more important than evidence guarantees.

Start the API and its supporting services:

```bash
docker compose up -d --build
```

API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Use the terminal CLI (no frontend required)

The CLI calls the same FastAPI endpoints as the web application, so it uses the
same authentication, sessions, retrieval/reranking, citation, and refusal flow.
Run it inside the already-started API container:

```bash
cd backend

# Register once; the password is entered with hidden input.
docker compose exec LS_api python rag_cli.py --username <username> --register

# Log in and enter the interactive question-answer loop.
docker compose exec LS_api python rag_cli.py --username <username>
```

Useful interactive commands include `/new`, `/sessions`, `/use <session_id>`,
`/history`, `/upload <file...>`, `/quick <file>`, `/files`, `/sources`, and
`/thinking on|off`. Enter `/help` in the client for the complete list.

If the backend Python dependencies are installed on the host, it can also run
outside Docker:

```bash
cd backend
python app/rag_cli.py --username <username>
```

The API address defaults to `http://localhost:8000`. Override it with
`--base-url` or `RAG_API_URL`. For a single non-interactive question, add
`--question "your question"`.

CLI requests skip recommended-question generation. Session titles are derived
locally from the first 30 characters of the first question, so neither feature
adds another model call to the CLI response path.

### 3. Start the frontend (optional)

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend: [http://localhost:5181](http://localhost:5181)

## Testing

Backend tests run against an isolated PostgreSQL test container:

```bash
cd backend

# Run the complete test suite
./run-tests.sh

# Run a specific test category
./run-tests.sh unit
./run-tests.sh api
./run-tests.sh integration
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## Offline RAG evaluation

The evaluation dataset and runner are located in `backend/evals`. Validate the dataset without accessing Elasticsearch or model services:

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --validate-only
```

Run a retrieval baseline:

```bash
docker compose exec LS_api \
  python /app/evals/run_retrieval_eval.py \
  --index-name <user-id> \
  --run-name retrieval_baseline
```

See [`backend/evals/README.md`](backend/evals/README.md) for dataset details, metrics, and additional commands.

## Useful commands

```bash
# View backend logs
docker logs -f LS_api

# View service status
cd backend
docker compose ps

# Stop backend services
docker compose down
```
