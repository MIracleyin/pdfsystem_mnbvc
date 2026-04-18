"""Gradio demo for the pdfsys-mnbvc MVP pipeline.

What this demonstrates (matching the code that actually exists in the
repo today, not the aspirational PRD):

* Stage-A XGBoost router — decides text-ok vs needs-ocr from 124
  PyMuPDF-derived features.
* MuPDF fast path — extracts Markdown-ready segments when the router
  picks ``Backend.MUPDF``. Overlaid on the first page as colored bboxes.
* ModernBERT OCR quality scorer — optional, heavy (~800 MB download,
  3–5 s per doc on CPU). Off by default to keep the demo snappy.

PIPELINE / VLM / DEFERRED backends are surfaced through the router
decision but are still stubs in ``packages/pdfsys-parser-*``; the UI
just reports the routing choice in that case and skips extraction.

Runs locally (``python demo/app.py``) and as a Hugging Face Space (see
the repo-root ``README.md`` frontmatter and ``demo/README.md``).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

# Allow ``python demo/app.py`` without installing the workspace by falling
# back to the in-tree sources. When running under HF Spaces / uv sync the
# packages are already on sys.path and these inserts become no-ops.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for pkg in ("pdfsys-core", "pdfsys-router", "pdfsys-parser-mupdf", "pdfsys-bench"):
    src = _REPO_ROOT / "packages" / pkg / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

from pipeline import (  # noqa: E402 — must come after sys.path surgery
    PipelineResult,
    pick_curated_features,
    render_first_page_with_bboxes,
    run_pipeline,
)


# ------------------------------------------------------------------ constants

DESCRIPTION = """\
# PDFSystem-MNBVC · Pipeline Demo

**FinePDFs-inspired PB-scale PDF → pretraining-data pipeline**, adapted
for the Chinese MNBVC corpus. This demo shows the MVP closed loop that
is actually implemented in the repo today:

**Router (XGBoost, 124 features)** → **MuPDF fast path** → **OCR Quality Scorer (ModernBERT)**

The router decides whether a PDF is cheap to parse with PyMuPDF alone,
or whether it needs to go to the (still-stubbed) OCR / VLM backends.
Roughly 90% of a typical PDF corpus takes the green fast-path lane.
"""

PIPELINE_DIAGRAM_MD = """\
### Pipeline

```
               ┌────────────────┐
   PDF ───────►│  Stage-A       │  XGBoost · ~10 ms/PDF
               │  Router        │  124 PyMuPDF features
               └────────┬───────┘
                        │  ocr_prob
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       MUPDF         PIPELINE        VLM / DEFERRED
       (text-ok)     (OCR, stub)     (VLM, stub)
          │
          ▼
     PyMuPDF blocks ─► Markdown + Segments (with bboxes)
          │
          ▼
     ModernBERT-large OCR quality regressor ─► score ∈ [0, 3]
```

**Backend color legend on page preview**

- 🟢 `mupdf` — text-ok fast path (implemented)
- 🟠 `pipeline` — OCR lane (stub, routing only)
- 🟣 `vlm` — VLM lane (stub, routing only)
- ⚪ `deferred` — held back until VLM workers online
"""


def _safe(val, default=""):
    """Coerce NaN / None for Gradio components that don't like them."""
    if val is None:
        return default
    try:
        import math

        if isinstance(val, float) and math.isnan(val):
            return default
    except Exception:
        pass
    return val


# ------------------------------------------------------------------ handlers


def process_pdf(
    pdf_file: str | None,
    run_quality: bool,
    ocr_threshold: float,
    progress: gr.Progress = gr.Progress(),
):
    """Main Gradio callback. Returns one value per output component."""
    empty_segments = [[0, 0, "-", "-", 0, ""]]
    empty_features = [["(no PDF uploaded)", ""]]
    empty_summary = "Upload a PDF to get started."

    if not pdf_file:
        return (
            empty_summary,
            "", 0.0, 0, "", 0.0,
            None,
            "_No markdown yet._",
            empty_segments,
            empty_features,
            "{}",
        )

    pdf_path = Path(pdf_file)

    try:
        progress(0.1, desc="Routing (XGBoost)…")
        result: PipelineResult = run_pipeline(
            pdf_path,
            run_quality=run_quality,
            ocr_threshold=ocr_threshold,
        )

        progress(0.7, desc="Rendering first page…")
        preview = render_first_page_with_bboxes(pdf_path, result, page_index=0)

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        err_json = {"error": str(e), "traceback": tb.splitlines()[-6:]}
        return (
            f"**Failed:** `{e}`",
            "", 0.0, 0, "", 0.0,
            None,
            f"```\n{tb}\n```",
            empty_segments,
            empty_features,
            json.dumps(err_json, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------- summary
    lines = [
        f"**File:** `{pdf_path.name}` ({pdf_path.stat().st_size / 1024:.1f} KB)",
        f"**Routed to:** `{result.backend}` &nbsp;·&nbsp; "
        f"P(ocr) = **{result.ocr_prob:.3f}** &nbsp;·&nbsp; {result.num_pages} page(s)",
    ]
    flags = []
    if result.is_form:
        flags.append("is_form")
    if result.is_encrypted:
        flags.append("encrypted")
    if result.needs_password:
        flags.append("password-protected")
    if result.garbled_text_ratio > 0.01:
        flags.append(f"garbled_text_ratio={result.garbled_text_ratio:.2%}")
    if flags:
        lines.append("**Flags:** " + ", ".join(f"`{f}`" for f in flags))
    if result.router_error:
        lines.append(f"**Router error:** `{result.router_error}`")
    if result.extract_error:
        lines.append(f"**Extract error:** `{result.extract_error}`")
    if result.quality_error:
        lines.append(f"**Quality error:** `{result.quality_error}`")

    if result.backend == "mupdf" and not result.extract_error:
        stats = result.extract_stats
        lines.append(
            f"**Extracted:** {stats.get('segment_count', 0)} segments, "
            f"{stats.get('char_count', 0):,} chars "
            f"(pages {stats.get('pages_extracted', 0)}/{stats.get('page_count', 0)})"
        )
    else:
        lines.append(
            "_MuPDF extraction skipped — backend is not `mupdf`. "
            "PIPELINE/VLM backends are still stubs in this repo._"
        )

    if result.quality_score is not None:
        lines.append(
            f"**OCR quality:** **{result.quality_score:.2f}** / 3.0 "
            f"({result.quality_num_tokens} tokens, `{result.quality_model}`)"
        )

    lines.append(
        f"**Timing (ms):** router **{result.wall_ms_router:.0f}** · "
        f"extract **{result.wall_ms_extract:.0f}** · "
        f"quality **{result.wall_ms_quality:.0f}**"
    )
    summary_md = "\n\n".join(lines)

    # ------------------------------------------------------------- markdown
    md_text = result.markdown.strip() or "_No markdown — this PDF was not routed to MuPDF._"
    if len(md_text) > 20_000:
        md_text = md_text[:20_000] + "\n\n…\n\n**[truncated for UI — full Markdown in the JSON tab]**"

    # ------------------------------------------------------------- segments
    seg_rows = [
        [s["index"], s["page"], s["type"], str(s["bbox_norm"]), s["chars"], s["preview"]]
        for s in result.segments
    ] or empty_segments

    # ------------------------------------------------------------- features
    feat_rows = pick_curated_features(result.router_features) or empty_features

    # ------------------------------------------------------------- raw JSON
    raw = result.to_record()
    raw["router_features_full"] = result.router_features
    raw["segments_full"] = result.segments
    raw_json_str = json.dumps(raw, indent=2, ensure_ascii=False, default=str)

    return (
        summary_md,
        result.backend,
        float(result.ocr_prob) if result.ocr_prob == result.ocr_prob else 0.0,
        int(result.num_pages),
        ("-" if result.quality_score is None else f"{result.quality_score:.2f} / 3.0"),
        float(result.wall_ms_router + result.wall_ms_extract + result.wall_ms_quality),
        preview,
        md_text,
        seg_rows,
        feat_rows,
        raw_json_str,
    )


# ---------------------------------------------------------------------- UI

CSS = """
.small-num input { font-weight: 600; font-size: 1.1rem; }
footer { display: none !important; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="PDFSystem-MNBVC Demo") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # -------------------- left column: controls + diagram
            with gr.Column(scale=1, min_width=320):
                pdf_input = gr.File(
                    label="Upload a PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                with gr.Accordion("Options", open=True):
                    ocr_threshold = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="OCR probability threshold",
                        info="ocr_prob ≥ threshold ⇒ route off the MuPDF fast path",
                    )
                    run_quality = gr.Checkbox(
                        label="Run ModernBERT quality scorer",
                        value=False,
                        info="~3–5 s on CPU. First run downloads ~800 MB.",
                    )
                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
                gr.Markdown(PIPELINE_DIAGRAM_MD)

            # -------------------- right column: outputs
            with gr.Column(scale=2, min_width=520):
                summary_md = gr.Markdown(
                    "Upload a PDF and click **Run Pipeline**.",
                )

                with gr.Row():
                    backend_out = gr.Textbox(
                        label="Backend", interactive=False, elem_classes=["small-num"]
                    )
                    ocr_prob_out = gr.Number(
                        label="P(OCR)", interactive=False, precision=3,
                        elem_classes=["small-num"],
                    )
                    pages_out = gr.Number(
                        label="Pages", interactive=False,
                        elem_classes=["small-num"],
                    )
                    quality_out = gr.Textbox(
                        label="Quality", interactive=False,
                        elem_classes=["small-num"],
                    )
                    wall_ms_out = gr.Number(
                        label="Total ms", interactive=False, precision=0,
                        elem_classes=["small-num"],
                    )

                with gr.Tabs():
                    with gr.Tab("Page preview"):
                        preview_img = gr.Image(
                            label="First page with extracted bboxes",
                            type="pil",
                            interactive=False,
                            height=720,
                        )
                    with gr.Tab("Markdown"):
                        md_out = gr.Markdown()
                    with gr.Tab("Segments"):
                        seg_df = gr.Dataframe(
                            headers=["idx", "page", "type", "bbox_norm", "chars", "preview"],
                            datatype=["number", "number", "str", "str", "number", "str"],
                            wrap=True,
                            label="Extracted segments (one row per block)",
                        )
                    with gr.Tab("Router features"):
                        feat_df = gr.Dataframe(
                            headers=["feature", "value"],
                            datatype=["str", "str"],
                            label="Curated subset (full 124-dim vector in Raw JSON)",
                        )
                    with gr.Tab("Raw JSON"):
                        raw_json = gr.Code(label="All pipeline outputs", language="json")

        # ----------------------------------------------------------- wiring
        outputs = [
            summary_md,
            backend_out, ocr_prob_out, pages_out, quality_out, wall_ms_out,
            preview_img,
            md_out,
            seg_df,
            feat_df,
            raw_json,
        ]
        run_btn.click(
            process_pdf,
            inputs=[pdf_input, run_quality, ocr_threshold],
            outputs=outputs,
        )
        # Auto-run on file upload (with quality off for snappiness).
        pdf_input.upload(
            lambda f, t: process_pdf(f, False, t),
            inputs=[pdf_input, ocr_threshold],
            outputs=outputs,
        )

        gr.Markdown(
            "---\n"
            "Repo: [pdfsystem_mnbvc](https://github.com/) · "
            "Architecture: [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) · "
            "Router weights: FinePDFs upstream (Apache-2.0) · "
            "Quality model: `HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn`"
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    # Sensible defaults for both local dev and HF Spaces.
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue(max_size=8).launch(
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CSS,
    )
