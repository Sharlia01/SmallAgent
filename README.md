# RAGSys

RAGSys is a full-stack document question-answering system powered by Retrieval-Augmented Generation (RAG). Users can upload documents, organize a personal knowledge base, and receive streaming answers grounded in the content of those documents.

## Features

- User registration and authentication
- Document upload, parsing, search, and deletion
- RAG-based question answering with streaming responses
- Conversation sessions and message history
- Support for common document formats, including PDF, DOCX, TXT, Excel, PowerPoint, HTML, and Markdown

## Tech Stack

- **Frontend:** React, TypeScript, Vite, and Ant Design
- **Backend:** FastAPI and Python
- **Data services:** PostgreSQL, Elasticsearch, and Redis
- **Embedding:** Local `bge-small-zh-v1.5`
- **LLM and rerank service:** Alibaba Cloud DashScope

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js and npm
- A local `bge-small-zh-v1.5` model directory
- A DashScope API key

### 1. Start the backend

```bash
cd backend
cp .env.example .env
```

Open `backend/.env`, add your DashScope API key, and replace the example passwords and JWT secret with secure values:

```env
DASHSCOPE_API_KEY="your-api-key"
BGE_MODEL_HOST_PATH=../../models/bge-small-zh-v1.5
```

`BGE_MODEL_HOST_PATH` is resolved relative to `backend/docker-compose.yml`.
The default matches a model stored at `Projects/models/bge-small-zh-v1.5`.

Then start the API and its supporting services:

```bash
docker compose up -d --build
```

The API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5181](http://localhost:5181) in your browser.

## Useful Commands

```bash
# View backend logs
docker logs -f swxy_api

# Stop all backend services
cd backend
docker compose down
```
