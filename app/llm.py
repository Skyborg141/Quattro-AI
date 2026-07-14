"""
app/llm.py — Ollama LLM Client
================================
Wraps Ollama's chat and completion endpoints with context injection for RAG.
"""

import asyncio, logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import ollama as _sdk
    _SDK = True
except ImportError:
    _SDK = False


class OllamaClient:
    BASE_URL = "http://localhost:11434"

    # ── Public methods ─────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        model: str = "qwen3:14b",   # updated default — June 2026
        history: list[dict] = None,
        context_chunks: list[dict] = None,
        images: list[str] = None,
        code_context: str = None,
    ) -> str:
        """
        Send a chat turn with optional RAG context, attached code files, and/or
        images, all injected as a single composed system message.
        context_chunks: list of {"text": ..., "source": ..., "score": ...}
        images: list of base64-encoded image strings (no data:image/... prefix)
                to attach to the current user message, for vision models.
        code_context: pre-formatted string of attached source file contents
                (filename + content blocks), built by main.py — never RAG-indexed,
                always sent verbatim so the model sees the exact file contents.
        """
        messages = []

        # Compose the system message from whichever context sources are active.
        # These are independent and can combine (e.g. RAG docs + an attached .py file).
        sys_parts = []
        if context_chunks:
            ctx_text = "\n\n---\n\n".join(
                f"[Source: {c['source']} | score: {c['score']}]\n{c['text']}"
                for c in context_chunks
            )
            sys_parts.append(
                "Use the following retrieved context passages to answer the "
                "user's question accurately. Cite sources when relevant.\n\n"
                f"=== RETRIEVED CONTEXT ===\n{ctx_text}\n=== END CONTEXT ==="
            )
        if code_context:
            sys_parts.append(
                "The user has attached the following source file(s) directly to "
                "this conversation. Read them carefully and use them to answer "
                "questions, review, explain, or debug as asked. Refer to files by "
                "name when relevant.\n\n"
                f"=== ATTACHED CODE FILES ===\n{code_context}\n=== END ATTACHED CODE FILES ==="
            )
        if images and not sys_parts:
            sys_parts.append(
                "You have vision capabilities. An image has been attached to the "
                "user's message below — look at it carefully and answer based on "
                "what you actually see."
            )

        base_identity = "You are a helpful AI assistant running locally via Ollama."
        if sys_parts:
            sys_content = base_identity + "\n\n" + "\n\n".join(sys_parts)
        else:
            sys_content = base_identity
        messages.append({"role": "system", "content": sys_content})

        # Add conversation history
        if history:
            messages.extend(history[-10:])   # keep last 10 turns

        # Add current user message (with images attached, if any)
        user_msg = {"role": "user", "content": message}
        has_images = bool(images)
        if has_images:
            user_msg["images"] = images
            logger.info(
                "Sending %d image(s) to model '%s', first image b64 length=%d",
                len(images), model, len(images[0]) if images else 0,
            )
        messages.append(user_msg)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._chat_sync, model, messages, has_images
        )

    async def chat_stream(
        self,
        message: str,
        model: str = "qwen3:14b",
        history: list[dict] = None,
        context_chunks: list[dict] = None,
        images: list[str] = None,
        code_context: str = None,
    ):
        """
        Async generator that yields text tokens one at a time from Ollama's
        streaming endpoint.  Build the message list exactly the same way as
        chat() so the system prompt, RAG context, and image handling are
        identical between the streaming and non-streaming paths.
        """
        import httpx

        # ── Build message list (mirrors chat()) ────────────────────────────
        messages = []
        sys_parts = []
        if context_chunks:
            ctx_text = "\n\n---\n\n".join(
                f"[Source: {c['source']} | score: {c['score']}]\n{c['text']}"
                for c in context_chunks
            )
            sys_parts.append(
                "Use the following retrieved context passages to answer the "
                "user's question accurately. Cite sources when relevant.\n\n"
                f"=== RETRIEVED CONTEXT ===\n{ctx_text}\n=== END CONTEXT ==="
            )
        if code_context:
            sys_parts.append(
                "The user has attached the following source file(s) directly to "
                "this conversation. Read them carefully and use them to answer "
                "questions, review, explain, or debug as asked. Refer to files by "
                "name when relevant.\n\n"
                f"=== ATTACHED CODE FILES ===\n{code_context}\n=== END ATTACHED CODE FILES ==="
            )
        if images and not sys_parts:
            sys_parts.append(
                "You have vision capabilities. An image has been attached to the "
                "user's message below — look at it carefully and answer based on "
                "what you actually see."
            )
        base_identity = "You are a helpful AI assistant running locally via Ollama."
        sys_content = base_identity + ("\n\n" + "\n\n".join(sys_parts) if sys_parts else "")
        messages.append({"role": "system", "content": sys_content})

        if history:
            messages.extend(history[-10:])

        user_msg: dict = {"role": "user", "content": message}
        has_images = bool(images)
        if has_images:
            user_msg["images"] = images
            logger.info(
                "Stream: sending %d image(s) to model '%s'",
                len(images), model,
            )
        messages.append(user_msg)

        # ── Stream from Ollama HTTP API ────────────────────────────────────
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    try:
                        import json as _json
                        chunk = _json.loads(raw_line)
                    except Exception:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    async def complete(self, prompt: str, model: str = "llama3.2") -> str:
        """Single-turn completion (no history, no RAG)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._complete_sync, model, prompt
        )

    async def list_models(self) -> dict:
        """Return models available in the local Ollama instance."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_models_sync)

    # ── Sync helpers (run in thread pool) ─────────────────────────────────────

    def _chat_sync(self, model: str, messages: list[dict], has_images: bool = False) -> str:
        # The Ollama Python SDK's typed Message/Image validation can silently
        # mishandle plain base64 strings passed inside a dict's "images" key
        # (see ollama-python issue #375). The raw HTTP endpoint accepts the
        # same base64 array with no validation surprises, so use it directly
        # whenever an image is attached — regardless of SDK availability.
        if has_images:
            return self._http_chat(model, messages)
        if _SDK:
            resp = _sdk.chat(model=model, messages=messages)
            return resp["message"]["content"]
        return self._http_chat(model, messages)

    def _complete_sync(self, model: str, prompt: str) -> str:
        if _SDK:
            resp = _sdk.generate(model=model, prompt=prompt)
            return resp["response"]
        return self._http_generate(model, prompt)

    def _list_models_sync(self) -> dict:
        if _SDK:
            models = _sdk.list()
            return {"models": [m.model for m in models.models]}
        import httpx
        r = httpx.get(f"{self.BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return {"models": [m["name"] for m in r.json()["models"]]}

    def _http_chat(self, model: str, messages: list[dict]) -> str:
        import httpx
        r = httpx.post(
            f"{self.BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _http_generate(self, model: str, prompt: str) -> str:
        import httpx
        r = httpx.post(
            f"{self.BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"]
