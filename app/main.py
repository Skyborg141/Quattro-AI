import os, uuid, time, logging, json, re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.rag import RAGPipeline
from app.llm import OllamaClient
from app.generators import ImageGenerator, PDFGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OllamaRAG Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag     = RAGPipeline()
llm     = OllamaClient()
img_gen = ImageGenerator()
pdf_gen = PDFGenerator()

class CodeFile(BaseModel):
    filename: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3.2"
    use_rag: bool = True
    top_k: int = 3
    history: list[dict] = []
    images: list[str] = []   # base64-encoded image strings, no data:image/... prefix
    code_files: list[CodeFile] = []   # plain-text source files attached directly to chat — never RAG-indexed

class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
    chunks: list[dict] = []
    model: str
    elapsed_ms: int

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0

class PDFRequest(BaseModel):
    topic: str
    model: str = "llama3.2"
    sections: list[str] = ["Introduction", "Analysis", "Conclusion"]
    style: str = "professional"

class CodeRequest(BaseModel):
    description: str
    language: str = "python"
    model: str = "llama3.2"
    detail: str = "line-by-line"   # "line-by-line" | "concise"

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=Path("static/index.html").read_text(encoding="utf-8"))

@app.get("/chat-ui", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=Path("static/index.html").read_text(encoding="utf-8"))

@app.get("/models")
async def list_models():
    return await llm.list_models()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.perf_counter()
    context_chunks, sources = [], []
    if req.use_rag:
        results = await rag.query(req.message, k=req.top_k)
        context_chunks = results["chunks"]
        sources = results["sources"]

    code_context = None
    if req.code_files:
        code_context = "\n\n".join(
            f"--- FILE: {cf.filename} ---\n{cf.content}"
            for cf in req.code_files
        )

    reply = await llm.chat(
        message=req.message,
        model=req.model,
        history=req.history,
        context_chunks=context_chunks,
        images=req.images or None,
        code_context=code_context,
    )
    elapsed = int((time.perf_counter() - t0) * 1000)
    return ChatResponse(reply=reply, sources=sources, chunks=context_chunks, model=req.model, elapsed_ms=elapsed)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Server-Sent Events endpoint for streaming chat responses.
    Yields tokens as 'data: <token>\\n\\n' lines.
    Sends '[SOURCES]:{"sources":[...]}' before '[DONE]' when RAG is active.
    """
    context_chunks, sources = [], []
    if req.use_rag:
        results = await rag.query(req.message, k=req.top_k)
        context_chunks = results["chunks"]
        sources = results["sources"]

    code_context = None
    if req.code_files:
        code_context = "\n\n".join(
            f"--- FILE: {cf.filename} ---\n{cf.content}"
            for cf in req.code_files
        )

    import json as _json

    async def event_generator():
        try:
            async for token in llm.chat_stream(
                message=req.message,
                model=req.model,
                history=req.history,
                context_chunks=context_chunks,
                images=req.images or None,
                code_context=code_context,
            ):
                # Escape newlines so each SSE message stays on one data: line
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            if sources:
                payload = _json.dumps({"sources": sources})
                yield f"data: [SOURCES]:{payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx proxy buffering
        },
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".json", ".csv", ".docx", ".xlsx", ".xls", ".pptx", ".ppt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    content = await file.read()
    doc_id = str(uuid.uuid4())
    save_path = Path("uploads") / f"{doc_id}{ext}"
    save_path.parent.mkdir(exist_ok=True)
    save_path.write_bytes(content)
    result = await rag.ingest(file_path=str(save_path), doc_id=doc_id, filename=file.filename)
    return {"doc_id": doc_id, "filename": file.filename, **result}

@app.get("/kb")
async def list_kb():
    return await rag.list_documents()

@app.delete("/kb/{doc_id}")
async def delete_kb(doc_id: str):
    await rag.delete_document(doc_id)
    return {"deleted": doc_id}

@app.post("/generate/image")
async def generate_image(req: ImageRequest):
    return await img_gen.generate(prompt=req.prompt, width=req.width, height=req.height, steps=req.steps, cfg_scale=req.cfg_scale)

@app.post("/generate/pdf")
async def generate_pdf(req: PDFRequest):
    result = await pdf_gen.generate(topic=req.topic, sections=req.sections, model=req.model, llm=llm, style=req.style)
    return FileResponse(result["path"], media_type="application/pdf", filename=result["filename"])

@app.post("/generate/code")
async def generate_code(req: CodeRequest):
    if req.detail == "concise":
        prompt = (
            f"Write precise, well-commented, production-quality {req.language} code for:\n\n"
            f"{req.description}\n\nReturn ONLY the code, no explanation."
        )
        code = await llm.complete(prompt=prompt, model=req.model)
        code = _strip_code_fence(code)
        return {"language": req.language, "code": code, "model": req.model, "lines": []}

    # ── line-by-line detail mode ────────────────────────────────────────────
    prompt = (
        f"Write precise, correct, production-quality {req.language} code for the "
        f"following request:\n\n{req.description}\n\n"
        "Respond with ONLY a single JSON object — no markdown fences, no prose "
        "before or after — in EXACTLY this shape:\n\n"
        '{\n'
        '  "code": "the complete code as a single string, with \\n for newlines",\n'
        '  "explanations": [\n'
        '    {"line": 1, "text": "what this exact line does and why"},\n'
        '    {"line": 2, "text": "..."}\n'
        '  ],\n'
        '  "summary": "2-3 sentence overview of the overall approach"\n'
        "}\n\n"
        "Rules:\n"
        "- Number every non-blank line of code starting at 1, matching the code field exactly.\n"
        "- Each explanation must be specific to that line (variables, logic, why it's needed) "
        "— never generic filler like 'this line does something'.\n"
        "- Cover EVERY line, including imports, blank structural lines you can skip, "
        "braces/indentation lines you can skip, but cover every line with real logic.\n"
        "- Do not include markdown code fences inside the \"code\" string.\n"
        "- Output must be valid JSON — escape quotes and newlines properly."
    )
    raw = await llm.complete(prompt=prompt, model=req.model)
    parsed = _parse_code_json(raw)

    if parsed is None:
        # Model didn't return valid JSON — fall back to plain code, no explanations,
        # rather than failing the request outright.
        logger.warning("Code-gen JSON parse failed, falling back to raw code only")
        return {
            "language": req.language,
            "code": _strip_code_fence(raw),
            "model": req.model,
            "lines": [],
            "summary": "",
        }

    return {
        "language": req.language,
        "code": parsed.get("code", ""),
        "model": req.model,
        "lines": parsed.get("explanations", []),
        "summary": parsed.get("summary", ""),
    }


def _strip_code_fence(text: str) -> str:
    """Remove ```lang ... ``` wrapping if the model added it despite instructions."""
    text = text.strip()
    m = re.match(r"^```[a-zA-Z0-9]*\n(.*)\n```$", text, re.DOTALL)
    return m.group(1) if m else text


def _parse_code_json(raw: str) -> Optional[dict]:
    """Extract and parse the JSON object the model was asked to return,
    tolerating markdown fences or stray text around it."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` wrapper if present
    fence = re.match(r"^```[a-zA-Z0-9]*\n(.*)\n```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the outermost {...} block and try again
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None

# Mount static files last
app.mount("/static", StaticFiles(directory="static"), name="static")
