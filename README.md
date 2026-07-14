# OllamaRAG Chatbot

A fully local AI chatbot with **Retrieval-Augmented Generation (RAG)**, **image generation**, and **PDF/code generation** — powered by [Ollama](https://ollama.com) and a FastAPI backend.

```
┌─────────────────────────────────────────────────────────┐
│                    OllamaRAG Stack                       │
│                                                          │
│  React UI ──► FastAPI  ──► Ollama (LLM + embeddings)    │
│                  │                                       │
│                  ├──► ChromaDB  (vector store)           │
│                  ├──► pdfplumber / python-docx (loaders) │
│                  ├──► ReportLab  (PDF export)            │
│                  └──► Stable Diffusion A1111 (images)    │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.11 | python.org |
| Ollama | latest | [ollama.com](https://ollama.com) |
| Node.js | ≥ 18 (optional, for UI dev) | nodejs.org |
| Stable Diffusion WebUI | optional | [A1111 GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |

---

## Quick Start

### 1 — Install Ollama & pull models

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2          # chat model
ollama pull nomic-embed-text  # embedding model (required for RAG)
ollama pull mistral            # optional alternative
```

### 2 — Clone & install Python deps

```bash
git clone <repo-url> ollama-rag-chatbot
cd ollama-rag-chatbot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Start the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API is now running at `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### 4 — Open the frontend

Open `index.html` in your browser, or serve it:

```bash
python -m http.server 3000
```

Then visit `http://localhost:3000`.

---

## Features

### 💬 RAG Chat
Upload any PDF, TXT, Markdown, CSV, or DOCX file. The backend:
1. Extracts text and chunks it (500-char windows, 100-char overlap)
2. Embeds chunks via `nomic-embed-text` running in Ollama
3. Stores vectors in ChromaDB (persisted on disk)
4. On each query, retrieves top-k chunks by cosine similarity
5. Injects retrieved context into the LLM's system prompt

### 🎨 Image Generation
Sends prompts to Stable Diffusion WebUI (A1111) at `localhost:7860`.
Requires A1111 running with `--api` flag:
```bash
./webui.sh --api   # Linux/macOS
webui-user.bat --api   # Windows
```

### 📄 PDF Generation
The LLM writes each section independently, then ReportLab compiles a
styled A4 PDF with custom typography, section headings, and a title page.

### 💻 Code Generation
Sends a structured prompt to the LLM and returns a clean code block.
Supports any language (Python, TypeScript, Rust, Go, etc.)

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | RAG-augmented chat |
| POST | `/upload` | Upload & index a document |
| GET | `/kb` | List knowledge base |
| DELETE | `/kb/{doc_id}` | Remove document |
| GET | `/models` | List Ollama models |
| POST | `/generate/image` | Image via Stable Diffusion |
| POST | `/generate/pdf` | AI-written PDF |
| POST | `/generate/code` | Code generation |

### Example: chat with RAG

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the key findings",
    "model": "llama3.2",
    "use_rag": true,
    "top_k": 3
  }'
```

### Example: upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_paper.pdf"
```

### Example: generate a PDF report

```bash
curl -X POST http://localhost:8000/generate/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Renewable Energy Trends in 2025",
    "sections": ["Introduction", "Solar Power", "Wind Energy", "Conclusion"],
    "model": "llama3.2"
  }' \
  --output report.pdf
```

---

## Project Structure

```
ollama-rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI routes
│   ├── rag.py           ← Chunking, embedding, retrieval (ChromaDB)
│   ├── llm.py           ← Ollama chat/completion client
│   └── generators.py    ← Image (SD) + PDF (ReportLab) generation
├── static/
│   └── index.html       ← Frontend UI
├── uploads/             ← Uploaded files (auto-created)
├── chroma_db/           ← Persistent vector store (auto-created)
├── generated/           ← Generated images & PDFs (auto-created)
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `app/rag.py` to change:
- `EMBED_MODEL` — embedding model (default: `nomic-embed-text`)
- `CHUNK_SIZE` — characters per chunk (default: 500)
- `CHUNK_OVERLAP` — overlap between chunks (default: 100)

Edit `app/llm.py` to change:
- `BASE_URL` — Ollama server address (default: `http://localhost:11434`)

Edit `app/generators.py` to change:
- `A1111_URL` — Stable Diffusion server (default: `http://localhost:7860`)

---

## Supported Models

Any model available in your local Ollama instance works. Recommended:

| Use case | Model |
|----------|-------|
| Fast chat | `llama3.2` or `phi3:mini` |
| Reasoning | `deepseek-r1:7b` |
| Code | `qwen2.5-coder:7b` |
| Embeddings | `nomic-embed-text` (required) |
| Multilingual | `qwen2.5:7b` |

---

## License

MIT — use freely for personal and commercial projects.
