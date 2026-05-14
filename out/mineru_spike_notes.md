# mineru 3.x spike notes — 2026-05-14

## Installed version

`mineru==3.1.12` (PyPI). `mineru.__version__` is unset; the actual version comes from `import importlib.metadata; importlib.metadata.version("mineru")`.

Installed via `uv pip install --index https://pypi.org/simple "mineru>=3.1,<4.0"` — 30 packages. Plus `accelerate==1.13.0` was needed (one extra `uv pip install accelerate`) for the VLM transformers backend's `device_map` handling. With just bare `mineru`, you get `ValueError: ... requires accelerate`.

## Module layout

```
mineru/
├── backend/
│   ├── pipeline/    (traditional CV backend — not used by us)
│   └── vlm/         (VLM backend — what we use)
├── cli/             (do_parse, aio_do_parse high-level API)
├── data/            (data readers/writers, DataWriter base class)
├── model/           (sub-model implementations)
├── utils/           (config_reader, get_device, env vars)
└── version
```

## Public entry point (verified)

```python
from mineru.cli.common import do_parse

do_parse(
    output_dir: str,                # mineru writes artifacts here
    pdf_file_names: list[str],      # logical names, e.g. ["doc"]
    pdf_bytes_list: list[bytes],    # raw PDF bytes per item (parallel to file_names)
    p_lang_list: list[str],         # e.g. ["ch"] or ["en"]
    backend: str = "pipeline",      # use "vlm-transformers" for VLM
    parse_method: str = "auto",
    formula_enable: bool = True,    # MUST set True (defaults True; magic-pdf 1.x had it off)
    table_enable: bool = True,      # MUST set True
    f_dump_md: bool = True,         # writes <name>.md
    f_dump_content_list: bool = True,   # writes <name>_content_list.json
    f_dump_middle_json: bool = True,
    f_dump_model_output: bool = True,
    f_dump_orig_pdf: bool = True,
    f_draw_layout_bbox: bool = True,
    f_draw_span_bbox: bool = True,
    f_make_md_mode: str = "mm_markdown",
    start_page_id: int = 0,
    end_page_id: int | None = None,  # process slice; None = all pages
    image_analysis: bool = True,
    **kwargs,
) -> None
```

Returns `None`. Reads artifacts back from `output_dir`:

```
<output_dir>/
└── <name>/
    └── vlm/
        ├── images/*.jpg            # cropped region images
        ├── <name>_content_list.json
        └── <name>.md
```

Verified one-PDF run (`1_pg19.pdf`, scanned math, 21 KB):
- wall time ~30 s on MPS after models cached
- 14 content_list items, unique types: `['equation', 'header', 'page_number', 'text']`
- 1305-byte markdown with valid LaTeX formulas: `$\frac{x^2}{(x^2 + 1)(x^2 + 4)} = \frac{A}{(x^2 + 1)} + \frac{B}{(x^2 + 4)}$`, `$$x^{2} = \mathrm{A}(x^{2}+4) + \mathrm{B}(x^{2}+1).$$`

For VLM-only call (lower-level), `mineru.backend.vlm.vlm_analyze.doc_analyze` exists but returns `(middle_json, infer_result)` — needs additional middle_json → content_list conversion. **Higher-level `do_parse` is simpler and writes everything to disk; use it.**

## Env vars (verified)

| Name | Default | What it does |
|---|---|---|
| `MINERU_DEVICE_MODE` | autodetect (`cuda` → `mps` → `npu` → ... → `cpu`) | Forces device. We set `mps` on Mac. |
| `MINERU_MODEL_SOURCE` | `huggingface` | `modelscope` is alternative. **ModelScope throttled large files (16 kB/s) on this Mac; HF was 600-900 kB/s and got the 2.15 GB model in ~30 min**. |
| `MINERU_TOOLS_CONFIG_JSON` | `mineru.json` in home | Path to JSON config (not used by us — env vars suffice). |
| `MINERU_FORMULA_ENABLE` | unset | Overrides the `formula_enable` param if set. |
| `MINERU_TABLE_ENABLE` | unset | Overrides the `table_enable` param if set. |
| `MINERU_VLM_FORMULA_ENABLE` | `True` | VLM-specific formula toggle. |
| `MINERU_VLM_TABLE_ENABLE` | `True` | VLM-specific table toggle. |
| `MINERU_VIRTUAL_VRAM_SIZE` | autodetect | GPU memory ceiling. |

`get_device()` reads `MINERU_DEVICE_MODE` first; if unset falls back to cuda > mps > ... > cpu. So setting `mps` explicitly is correct.

**Note: `~/magic-pdf.json` from the old engine is NOT read by mineru.** The legacy config file is harmless and can stay; `_vlm_config.py` still moves it to `.bak` for cleanliness.

## Output shape — content_list item

Each item in `<name>_content_list.json` is a flat dict. Schema verified:

```json
{
  "type": "text" | "equation" | "table" | "image" | "header" | "page_number" | ...,
  "text": "...",          // content body. For LaTeX equations: "$$...$$" or "$...$". For HTML tables: presumed "<table>...</table>".
  "text_format": "latex" | "html" | absent,  // only present for equation / table
  "bbox": [x0, y0, x1, y1], // PIXEL coordinates (not normalized) at rendering DPI
  "page_idx": 0,            // 0-based page index — note: page_idx, NOT page_index
  "img_path": "images/...jpg" // only present for image items (not seen in this spike)
}
```

Key differences from magic-pdf 1.x's content_list:
- magic-pdf used `bbox` + `page_width` + `page_height` to derive normalized [0,1] coords. mineru does NOT include `page_width` / `page_height` in content_list items — they're in `middle_json` only.
- magic-pdf had separate `interline_equation` and `inline_equation` types. mineru uses `equation` for both; the `$$...$$` vs `$...$` distinction lives in the `text` field's wrapping characters.
- mineru introduces `header`, `footer`, `page_number` as distinct types (magic-pdf had `header`/`footer` but no `page_number`).

## Content-type → RegionType map

| mineru type | RegionType | content field | notes |
|---|---|---|---|
| `text` | `TEXT` | `text` | |
| `title` | `TEXT` | `text` | not seen in spike, expected |
| `equation` | `FORMULA` | `text` (already LaTeX-wrapped) | both inline and display |
| `inline_equation` | `FORMULA` | `text` | legacy compat — may not appear in mineru 3.x |
| `interline_equation` | `FORMULA` | `text` | legacy compat |
| `table` | `TABLE` | `text` (HTML or LaTeX per `text_format`) | not seen in spike |
| `image` / `figure` | `IMAGE` | `img_path` (relative to output_dir) | not seen in spike |
| `figure_caption` / `table_caption` / `table_footnote` | `TEXT` | `text` | |
| `header` | `TEXT` | `text` | |
| `footer` | `TEXT` | `text` | |
| `page_number` | `TEXT` | `text` | new in mineru — could also skip these |
| `reference` | `TEXT` | `text` | |

## bbox normalization

mineru returns `bbox` in **pixel coordinates at rendering DPI** (which mineru chose internally — appears to be ~200 DPI per the spike, based on bbox magnitudes ~700 for a US letter page). To get [0, 1] normalized coords for `pdfsys_core.BBox`, we need page dimensions.

Two options:
1. Render the page via PyMuPDF and read `page.rect` to get true page dims, then divide bbox by them. Most reliable.
2. Read `middle_json` (`f_dump_middle_json=True`) which contains per-page width/height. Heavier but doesn't require re-rendering.

**Recommendation for Task 7:** Use option 1 (PyMuPDF page.rect). The rendering step is already done by `pdfsys-parser-mupdf` / `pdfsys-layout-analyser`. Caching one `pymupdf.Document` open per PDF is cheap.

If the bbox math gets fragile, **fall back to bbox=None on Segments** — the spec doesn't require bboxes on VLM output, and downstream parquet schema treats `bbox` as nullable.

## MPS path

Worked. `MINERU_DEVICE_MODE=mps` was honored; the Qwen2VL VLM ran on MPS for inference. ~30 s per page at ~1.5 s/it for 16 predict iterations.

No fallback to CPU was needed. (The spike script had 3 false starts: ModelScope throttle, missing `accelerate`, missing `if __name__ == '__main__'` guard — none were MPS-specific.)

## Model cache

Downloaded to `~/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/` (~2.15 GB safetensors + tokenizer + 11 small config files = ~2.2 GB total).

ModelScope path is `~/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-Pro-2604-1___2B/` (note the underscored escape).

## Quirks / lessons

1. **`if __name__ == '__main__':` is mandatory** on macOS. mineru spawns subprocess workers via `concurrent.futures.ProcessPoolExecutor` for PDF rendering. Without the main-guard, the subprocesses re-import the module, re-run side effects, and the executor dies with `BrokenProcessPool`.
2. **`accelerate` is a soft requirement.** The base `mineru` package doesn't list it, but `transformers` needs it when models use `device_map`. Adding `accelerate>=1.0` to `pdfsys-parser-vlm`'s pyproject.toml prevents the surprise.
3. **ModelScope can be slow even from China.** On this Mac, the 2.15 GB safetensors throttled to 16 kB/s on MS (small files were fine at 600-800 kB/s — feels like server-side QoS for large blobs). HF Hub was consistently 600-900 kB/s. Setting `MINERU_MODEL_SOURCE=huggingface` is the safer default for our use case; users in network-restricted environments can switch with one env var.
4. **mineru writes large output trees.** Even with `f_dump_orig_pdf=False, f_dump_middle_json=False, f_dump_model_output=False, f_draw_layout_bbox=False, f_draw_span_bbox=False`, it still writes a `<name>/vlm/images/` subdir with cropped region JPGs. We can ignore these (we only read `*_content_list.json` and `*.md`), but be aware the temp dir gets several MB per PDF. Using `tempfile.TemporaryDirectory()` (auto-cleanup) is the right pattern.
5. **mineru's `content_list` items use `page_idx`, not `page_index`.** Both old-magic-pdf-style fallback (`item.get("page_idx", item.get("page_index", 0))`) is still safe.

## Code template for Task 7's `_invoke_mineru`

Drop-in replacement for the old `_invoke_magic_pdf` code:

```python
def _invoke_mineru(
    self, pdf_bytes: bytes
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Call mineru's VLM entry and return (content_list, markdown, stats)."""
    # Lazy import — mineru pulls torch + transformers + many deps.
    from mineru.cli.common import do_parse  # noqa: PLC0415
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="pdfsys_vlm_") as tmpdir:
        do_parse(
            output_dir=tmpdir,
            pdf_file_names=["doc"],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],  # could be configurable via VlmConfig later
            backend="vlm-transformers",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_dump_md=True,
            f_dump_content_list=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
        )

        td = _Path(tmpdir)
        content_list: list[dict[str, Any]] = []
        for cand in td.rglob("*_content_list.json"):
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    content_list = [it for it in data if isinstance(it, dict)]
                break
            except (json.JSONDecodeError, OSError):
                continue

        md_content = ""
        for cand in td.rglob("*.md"):
            try:
                md_content = cand.read_text(encoding="utf-8")
                break
            except OSError:
                continue

    stats: dict[str, Any] = {
        "api": "mineru_v3",
        "backend": "vlm-transformers",
    }
    return content_list, md_content, stats
```

The `_MINERU_TYPE_MAP` can stay close to the magic-pdf 1.x version with one addition: `"page_number": RegionType.TEXT` (or drop entirely if we don't want page numbers in our markdown — magic-pdf didn't emit them).

For tables and equations, the `content` field already comes wrapped in `$$...$$` / `$...$` / `<table>...</table>` — no extra wrapping needed in `_content_list_to_segments`. The existing pull-from-field logic (`item.get("html", "") or item.get("latex", "") or item.get("text", "")`) works as-is since mineru puts everything in `text`.
