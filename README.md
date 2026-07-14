# ⬡ Quattro AI

A locally-hosted AI chatbot built on **FastAPI** and **Ollama** — runs entirely on your own machine, with no data ever sent to the cloud. It combines streaming chat, retrieval-augmented generation (RAG) over your documents, vision, code generation, image generation, and PDF report generation in one interface.

## Features

- **Streaming chat** — token-by-token responses over Server-Sent Events (`/chat/stream`), with a stop button to cancel generation mid-response.
- **RAG over your documents** — upload PDF, DOCX, TXT, MD, JSON, CSV, XLSX, XLS, PPTX, or PPT files. They're chunked, embedded with `nomic-embed-text`, stored in a local ChromaDB vector store, and retrieved automatically to ground answers (with cited sources). Adjustable top-k retrieval count.
- **Vision** — attach images (drag-and-drop, file picker, or paste) and ask questions about them using a vision-capable Ollama model. RAG context is automatically skipped when an image is attached so the model focuses on what it sees.
- **Code generation** — describe what you want in plain English and get back syntax-highlighted code, either as a concise block or with a line-by-line explanation of what each line does and why. Auto-detects the target language from your description.
- **Image generation** — generates images via a local Stable Diffusion install (AUTOMATIC1111 WebUI API by default; swappable for ComfyUI).
- **PDF report generation** — give it a topic and it writes each section with the LLM, then compiles a styled, multi-section PDF with ReportLab.
- **Code file attachments** — attach source files directly to a chat turn (sent verbatim to the model, never RAG-indexed) for review, debugging, or Q&A.
- **Auto intent detection** — a plain chat message is automatically routed to the right pipeline (chat / image / PDF / code) based on what you asked for, no mode-switching required.
- **Chat history** — sessions are saved locally in the browser (`localStorage`), with a sidebar to revisit past conversations.
- **Model picker** — model list is pulled live from your local Ollama instance; your last-used model is remembered.
- **Quality-of-life extras** — copy/regenerate buttons on responses, export chat to `.txt`, keyboard shortcuts (new chat, focus input, export, attach, stop generation), and drag-and-drop file attachment.

## Architecture

```
┌───────────────────────┐        ┌────────────────────────────────┐
│  static/index.html      │ ─────▶ │  FastAPI (app/main.py)           │
│  (vanilla HTML/CSS/JS)   │        │  /chat, /chat/stream, /upload      │
└───────────────────────┘        │  /kb, /generate/*, /models          │
                                  └────────────┬──────────┬───────────┘
                                               │          │
                            ┌──────────────────┘          └──────────────────┐
                            ▼                                                ▼
                  ┌────────────────────┐                          ┌────────────────────┐
                  │  app/rag.py           │                          │  app/llm.py            │
                  │  ChromaDB +             │                          │  Ollama client            │
                  │  nomic-embed-text         │                          │  (chat/stream/vision)       │
                  └────────────────────┘                          └────────────────────┘
                                                                             │
                                                                             ▼
                                                                   ┌────────────────────┐
                                                                   │  app/generators.py      │
                                                                   │  Stable Diffusion          │
                                                                   │  (A1111) + ReportLab          │
                                                                   └────────────────────┘
```

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, request/response models, all HTTP + SSE endpoints |
| `app/rag.py` | Document loading (multi-format), chunking, embedding, ChromaDB storage/retrieval, knowledge-base management |
| `app/llm.py` | Ollama chat/completion/streaming client, system-prompt composition (RAG context + attached code + vision), model listing |
| `app/generators.py` | Stable Diffusion image generation and LLM-driven, ReportLab-compiled PDF report generation |
| `static/index.html` | Full frontend — chat UI, model picker, RAG toggle, attachments, session history, shortcuts |

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running locally
- *(Optional)* **[AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** running with `--api` enabled, if you want image generation — the app falls back to a placeholder image if it isn't running

### Models used

Pull whichever models fit your hardware — the model dropdown populates automatically from whatever you have installed in Ollama. These are the ones this project has been built and tested against:

```bash
ollama pull qwen3:14b          # default chat model
ollama pull llama3.2           # lighter-weight fallback
ollama pull nomic-embed-text   # required — powers RAG embeddings
ollama pull gemma4:e4b         # vision, lightweight
ollama pull gemma4:26b         # vision, higher quality
```

`nomic-embed-text` is required for RAG to work. The rest are interchangeable — swap in any chat or vision model your machine can comfortably run, then select it from the model dropdown in the UI.

## Setup

```bash
git clone https://github.com/Skyborg141/Quattro-AI.git
cd Quattro-AI

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Make sure Ollama is running (`ollama serve`, or the desktop app), then pull the models you plan to use (see above).

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` , `/chat-ui` | Serves the frontend |
| `GET` | `/models` | Lists models available in the local Ollama instance |
| `POST` | `/chat` | Single-shot chat completion (RAG, images, code context) |
| `POST` | `/chat/stream` | Streaming chat via Server-Sent Events |
| `POST` | `/upload` | Upload and ingest a document into the RAG knowledge base |
| `GET` | `/kb` | List all documents currently in the knowledge base |
| `DELETE` | `/kb/{doc_id}` | Remove a document from the knowledge base |
| `POST` | `/generate/image` | Generate an image via Stable Diffusion |
| `POST` | `/generate/pdf` | Generate a multi-section PDF report |
| `POST` | `/generate/code` | Generate code, optionally with a line-by-line explanation |

## Project Structure

```
Quattro-AI/
├── app/
│   ├── main.py           # FastAPI app + endpoints
│   ├── rag.py             # RAG pipeline (ChromaDB + document loaders)
│   ├── llm.py              # Ollama client
│   └── generators.py         # Image + PDF generation
├── static/
│   └── index.html            # Frontend
├── chroma_db/                  # Vector store (created at runtime)
├── uploads/                      # Ingested source documents
├── generated/                      # Generated images/PDFs
├── requirements.txt
├── Commands.txt                # Quick-start command reference
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
