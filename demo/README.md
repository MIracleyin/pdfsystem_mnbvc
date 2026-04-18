# pdfsys-mnbvc · Gradio Demo

A small self-contained Gradio app that runs the **actually-implemented** MVP
path of the pdfsys-mnbvc pipeline on a single PDF you upload.

It exercises the same three components the bench harness does:

1. **Stage-A XGBoost router** (`pdfsys_router.Router`) — 124 PyMuPDF features → `ocr_prob` → one of `mupdf / pipeline / vlm / deferred`.
2. **MuPDF fast path** (`pdfsys_parser_mupdf.extract_doc`) — runs only when the router picks `mupdf`. Emits `Segment[]` with normalized bboxes + a merged Markdown blob.
3. **ModernBERT OCR quality scorer** (`pdfsys_bench.quality.OcrQualityScorer`) — optional; heavy; gated behind a checkbox.

PIPELINE / VLM / DEFERRED backends are currently stubs in the repo, so the
demo surfaces the router decision and skips extraction for them.

## UI

```
┌─────────────────┬──────────────────────────────────────────────────┐
│  upload PDF     │  Summary · backend · P(ocr) · pages · timing     │
│  threshold      ├──────────────────────────────────────────────────┤
│  ☐ quality      │  [ Page preview │ Markdown │ Segments │          │
│  [Run Pipeline] │    Router features │ Raw JSON ]                  │
│                 │                                                   │
│  pipeline       │  Page preview draws extracted bboxes (color =     │
│  diagram        │  chosen backend) directly on the first page.      │
└─────────────────┴──────────────────────────────────────────────────┘
```

## Run locally

```bash
# option A — full workspace install (recommended)
uv sync                                           # installs all packages + deps
python -m pdfsys_router.download_weights          # one-time: XGBoost weights (257 KB)
python app.py                                     # http://localhost:7860

# option B — plain pip (matches HF Spaces)
pip install -r requirements.txt
python -m pdfsys_router.download_weights
python app.py
```

First run of the quality scorer pulls `HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn`
(~800 MB) from the HF Hub. Set `HF_HOME=/path/to/cache` to control where it lands.

## Deploy to Hugging Face Spaces

The root `README.md` already contains the required [Spaces YAML config](https://huggingface.co/docs/hub/spaces-config-reference):

```yaml
---
title: PDFSystem MNBVC Demo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
license: apache-2.0
---
```

### Option 1 · One-click from GitHub (recommended)

1. Push this repo to GitHub.
2. Go to <https://huggingface.co/new-space>.
3. Pick **Gradio** SDK, hardware **CPU basic** is enough for the MVP loop.
4. In **Files** → **Create Space from an existing GitHub repo**, paste the repo URL.

HF Spaces will clone the whole repo, read the YAML header in the root
`README.md`, install `requirements.txt`, and launch `app.py`. The router's
XGBoost weights are downloaded automatically on first request (~257 KB, inline
in the Space container).

### Option 2 · Manual push

```bash
git clone https://huggingface.co/spaces/<you>/pdfsys-mnbvc-demo
cd pdfsys-mnbvc-demo
# copy repo contents into this dir (the four workspace packages must come
# along — they are installed editable by requirements.txt)
cp -r /path/to/pdfsystem_mnbvc/{app.py,requirements.txt,README.md,packages,demo} .
git add . && git commit -m "Initial deploy" && git push
```

### Resource notes (HF Spaces free tier: CPU, 16 GB RAM)

- Router: ~50–100 ms per PDF; effectively free.
- MuPDF extraction: ~10 ms per page.
- Quality scorer (ModernBERT-large): ~3–5 s per PDF at bf16; fits in RAM.
  Disabled by default in the UI. **Keep it off** unless you want to wait.
- GPU Spaces aren't required; the MVP path is CPU-only. A GPU Space becomes
  useful once the Pipeline / VLM parsers land.

## Files

| Path | Role |
| ---- | ---- |
| `demo/app.py` | Gradio `Blocks` definition + event handlers. |
| `demo/pipeline.py` | Pure-Python wrapper around `Router` + `extract_doc` + `OcrQualityScorer`. Rendering helpers live here too. |
| `app.py` (repo root) | Thin HF-Spaces entry; imports `demo.app`. |
| `requirements.txt` (repo root) | Pin-friendly deps for `pip install -r`. Installs the four workspace packages in editable mode. |

The demo imports the real pipeline modules — if you change `pdfsys-router`
or `pdfsys-parser-mupdf`, the demo picks it up on the next launch.
