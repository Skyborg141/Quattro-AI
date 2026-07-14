"""
app/generators.py — Image & PDF Generation
============================================

ImageGenerator  → Stable Diffusion via A1111 REST API (or ComfyUI)
PDFGenerator    → LLM-written content compiled to PDF via ReportLab
"""

import asyncio, base64, io, logging, uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm import OllamaClient

logger = logging.getLogger(__name__)

# ── PDF deps ───────────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, TableOfContents
    )
    _PDF_OK = True
except ImportError:
    _PDF_OK = False
    logger.warning("reportlab not installed. PDF generation disabled.")


OUTPUT_DIR = Path("./generated")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Image Generator
# ──────────────────────────────────────────────────────────────────────────────

class ImageGenerator:
    """
    Calls Automatic1111 Stable Diffusion WebUI at http://localhost:7860.
    Install: https://github.com/AUTOMATIC1111/stable-diffusion-webui
    Alternatively, swap _call_a1111 for _call_comfyui if you prefer ComfyUI.
    """
    A1111_URL = "http://localhost:7860"

    async def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        negative_prompt: str = "blurry, low quality, ugly",
    ) -> dict:
        loop = asyncio.get_event_loop()
        try:
            b64, info = await loop.run_in_executor(
                None, self._call_a1111, prompt, negative_prompt,
                width, height, steps, cfg_scale
            )
        except Exception as e:
            logger.warning("SD unavailable (%s), returning placeholder", e)
            b64 = self._placeholder_b64()
            info = {"note": "Stable Diffusion not running. Start A1111 or ComfyUI."}

        # Save to disk
        img_bytes = base64.b64decode(b64)
        fname = f"img_{uuid.uuid4().hex[:8]}.png"
        fpath = OUTPUT_DIR / fname
        fpath.write_bytes(img_bytes)

        return {
            "filename": fname,
            "url": f"/static/generated/{fname}",
            "b64": b64,
            "info": info,
        }

    def _call_a1111(self, prompt, neg, w, h, steps, cfg) -> tuple[str, dict]:
        import httpx
        payload = {
            "prompt": prompt,
            "negative_prompt": neg,
            "width": w,
            "height": h,
            "steps": steps,
            "cfg_scale": cfg,
            "sampler_name": "DPM++ 2M Karras",
        }
        r = httpx.post(f"{self.A1111_URL}/sdapi/v1/txt2img", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["images"][0], data.get("info", {})

    def _placeholder_b64(self) -> str:
        """1×1 grey PNG as placeholder when SD is offline."""
        PNG_1x1 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YGBgAAAABAABJjAHggAAAABJRU5ErkJggg=="
        )
        return PNG_1x1


# ──────────────────────────────────────────────────────────────────────────────
# PDF Generator
# ──────────────────────────────────────────────────────────────────────────────

class PDFGenerator:
    """
    Generates a multi-section PDF:
    1. Ask the LLM to write each section.
    2. Compile with ReportLab into a styled A4 PDF.
    """

    async def generate(
        self,
        topic: str,
        sections: list[str],
        model: str,
        llm: "OllamaClient",
        style: str = "professional",
    ) -> dict:
        # Generate section content concurrently
        tasks = [
            llm.complete(
                prompt=(
                    f"Write a {style} section titled '{sec}' for a report about: {topic}.\n"
                    f"Length: 2-4 paragraphs. Plain text only, no markdown."
                ),
                model=model,
            )
            for sec in sections
        ]
        contents = await asyncio.gather(*tasks)

        fname = f"report_{uuid.uuid4().hex[:8]}.pdf"
        fpath = OUTPUT_DIR / fname

        if _PDF_OK:
            self._build_pdf(fpath, topic, sections, contents)
        else:
            fpath.write_text(
                f"PDF generation requires reportlab.\n"
                f"pip install reportlab\n\n--- {topic} ---\n\n"
                + "\n\n".join(f"== {s} ==\n{c}" for s, c in zip(sections, contents))
            )

        return {"path": str(fpath), "filename": fname}

    def _build_pdf(
        self,
        path: Path,
        title: str,
        sections: list[str],
        contents: list[str],
    ):
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=2.5 * cm,
            leftMargin=2.5 * cm,
            topMargin=3 * cm,
            bottomMargin=2.5 * cm,
        )
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=22, leading=28, spaceAfter=12,
            textColor=colors.HexColor("#1a1a2e"),
        )
        heading_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=14, leading=18, spaceBefore=20, spaceAfter=6,
            textColor=colors.HexColor("#4a4aff"),
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=11, leading=16, spaceAfter=8,
            textColor=colors.HexColor("#222222"),
        )

        story = [
            Paragraph(title, title_style),
            HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4a4aff")),
            Spacer(1, 12),
        ]

        for sec, content in zip(sections, contents):
            story.append(Paragraph(sec, heading_style))
            for para in content.split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 8))

        doc.build(story)
        logger.info("PDF written to %s", path)
