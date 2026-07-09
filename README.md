# AI Medical Consultant

Python-first AI medical consultation prototype. The current version uses only the Python standard library for the backend and a small native JavaScript frontend.

## Features

- Multi-turn consultation sessions
- Local multimodal RAG pipeline with query expansion, BM25, text dense retrieval, image dense retrieval, medical term retrieval, RRF fusion, and context assembly
- LangChain-style agent decision orchestration: rule precheck, RAG retrieval, prompt assembly, LLM generation, JSON parsing, and safety merge
- Symptom extraction and urgency scoring
- Department recommendation
- Source knowledge display
- Safety disclaimer and urgent-care detection

## Run

Install Python dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
npm start
```

Open `http://127.0.0.1:3000`.

To run the previous dependency-light standard-library backend:

```bash
npm run legacy
```

Embedding provider defaults to the Qwen adapter:

```powershell
$env:EMBEDDING_PROVIDER="qwen"
$env:QWEN_ENABLE_LOCAL="0"
npm.cmd start
```

With `QWEN_ENABLE_LOCAL=0`, the app keeps a dependency-free fallback so it still runs before local Qwen model dependencies are installed.

## Backend Architecture

The default runtime is still dependency-light:

```text
run.py
  -> app.main.run()
  -> ThreadingHTTPServer
  -> app.services.runtime.consultation_service
  -> MedicalAgent
  -> RAGService + optional LLM workflow
```

The default runtime now uses the FastAPI entrypoint at `app/fastapi_main.py`.
The app reuses the same `ConsultationService`, Agent, and RAG pipeline as the
legacy server, so the migration path does not fork business logic.

Equivalent FastAPI command:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.fastapi_main:app --host 0.0.0.0 --port 3000 --reload
```

FastAPI routes are versioned under `/api/v1/*`, with legacy `/api/*`
compatibility routes kept for the current frontend. `npm run legacy` still runs
the previous `ThreadingHTTPServer` implementation.

## Agent LLM Orchestration

The agent first runs the deterministic triage/RAG pipeline, then can optionally call an OpenAI-compatible LLM through a LangChain LCEL chain (`ChatPromptTemplate | RunnableLambda`). The LLM is disabled by default so tests remain offline.

To test with DeepSeek:

```powershell
$env:AGENT_LLM_ENABLED="1"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
$env:DEEPSEEK_BASE_URL="<your base_url, for example https://.../v1>"
$env:DEEPSEEK_API_KEY="<your api_key>"
npm.cmd start
```

Supported LLM parameters:

```powershell
$env:LLM_TEMPERATURE="0.3"
$env:LLM_TOP_P="0.9"
$env:LLM_MAX_TOKENS="1400"
$env:LLM_TIMEOUT_SECONDS="30"
```

The safety merge keeps urgent-care detection from being downgraded by the model, appends a medical disclaimer when needed, and falls back to the rule-based answer if the provider is unavailable.

Keep the terminal running while using the app. If port `3000` is unavailable on Windows PowerShell:

```powershell
$env:PORT=3001
npm.cmd start
```

## Test

```bash
npm test
```

## Vectorize Documents

Put `.md`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.webp`, or `.bmp` medical knowledge files in `knowledge_base/`, then build the local vector index:

```bash
npm run vectorize
```

The generated index is saved to `storage/vector_index.json`. The app loads this path by default through `RAG_INDEX_PATH`.

Document files can include front matter:

```markdown
---
id: chest-pain-red-flags
title: 胸痛急症危险信号
category: emergency
department: 急诊科 / 心内科
severity_hint: 4
keywords: 胸痛,胸闷,呼吸困难,大汗
aliases: 胸口疼,心口疼,喘不上气
red_flags: 胸痛伴呼吸困难,大汗,晕厥
---

正文医学知识...
```

Image files can include sidecar metadata with the same stem, for example `skin_rash_photo.json`:

```json
{
  "id": "skin-rash-image",
  "title": "皮疹照片",
  "category": "image",
  "department": "皮肤科",
  "severity_hint": 1,
  "keywords": "皮疹,过敏,skin,rash",
  "image_type": "clinical photo",
  "body_part": "skin",
  "description": "患者皮肤红色皮疹照片"
}
```

Images are indexed as multimodal chunks with:

- `modality=image`
- generated `caption`
- original `image_path`
- text vector from caption/metadata
- image vector from image bytes/path

## API

Current stable frontend-compatible routes:

- `GET /api/health`
- `GET /api/knowledge?q=头痛`
- `POST /api/consultations`
- `POST /api/consultations/{id}/messages`
- `GET /api/consultations/{id}`

Optional FastAPI v1 routes:

- `GET /api/v1/health`
- `GET /api/v1/knowledge?q=...&top_k=5`
- `GET /api/v1/agent/status`
- `POST /api/v1/consultations`
- `POST /api/v1/consultations/{id}/messages`
- `GET /api/v1/consultations/{id}`
- `WS /api/ws/chat`

## Next Steps

- Replace the standard-library hash multimodal retriever with Qwen-VL/Qwen Embedding, BioMedCLIP, SentenceTransformer + FAISS, or ChromaDB.
- Add persistent storage behind `ConsultationService`, starting with SQLite for local development and PostgreSQL for deployment.
- Add persistent storage with PostgreSQL.
- Expand the LLM provider layer with streaming SSE/WebSocket output.

## RAG Pipeline

The current RAG implementation lives in `app/services/rag_service.py`:

1. Offline vectorization: load text/image documents, split text chunks, caption images, embed text and images, save `storage/vector_index.json`.
2. Query expansion: synonym replacement, symptom normalization, and medical term expansion.
3. Sparse retrieval: BM25-style keyword matching.
4. Dense text retrieval: dependency-free hash-vector cosine similarity.
5. Dense image retrieval: image-vector similarity plus caption matching.
6. Medical term retrieval: disease, symptom, red-flag, and department term matching.
7. Reciprocal Rank Fusion: combines all retrieval channels.
8. Deduplication and Top-K selection.
9. Context assembly: system prompt, patient context, conversation history, retrieved docs, and user question.

The embedding provider interface lives in `app/services/embedding_provider.py`. The default provider is `QwenEmbeddingProvider`, which falls back to dependency-free hash vectors until local Qwen dependencies and model weights are installed:

```python
class EmbeddingProvider:
    def embed_text(text) -> list[float]: ...
    def embed_image(image_path) -> list[float]: ...
    def caption_image(image_path, metadata) -> str: ...
```

Planned local Qwen runtime dependencies, pending confirmation before adding production dependencies:

```text
sentence-transformers
transformers
torch
accelerate
pillow
qwen-vl-utils
```
