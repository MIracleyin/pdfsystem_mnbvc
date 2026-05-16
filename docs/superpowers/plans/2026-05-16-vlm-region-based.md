# VLM Lane Region-Based Parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `pdfsys-parser-vlm/extract.py` so the VLM lane crops each layout region from the PDF and recognizes it via `MinerUClient.batch_content_extract`, instead of feeding whole pages to `do_parse`. DocLayout-YOLO becomes the single layout source; mineru becomes a per-region recognizer.

**Architecture:** Spike-first. Task 1 verifies `MinerUClient` can be constructed directly and `batch_content_extract([crop], ['table'])` returns sensible content. Task 2 rewrites `extract.py` against the verified API. Tasks 3-5 extend the viz CLI + HTML to surface per-region results. Tasks 6-7 re-run the 150-PDF dataset + verify acceptance and commit a post-build note.

**Tech Stack:** Python 3.11, `mineru>=3.1` (already installed), `mineru_vl_utils.MinerUClient`, PyMuPDF for page rendering, PIL for cropping.

**Source spec:** `docs/superpowers/specs/2026-05-16-vlm-region-based-design.md` — read first for §4 region-type map, §7 batching shape, §13 acceptance queries.

**User constraint:** No unit tests, consistent with prior iterations. Verification is by spike + end-to-end run + browser inspection.

**Phasing note:** Task 1 is a discovery spike whose findings (`out/mineru_region_spike_notes.md`) inform Task 2's rewrite. Controller should read the spike notes between Tasks 1 and 2.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `out/mineru_region_spike_notes.md` | create (spike) | Verified `MinerUClient.batch_content_extract` signature + crop pipeline + ModelSingleton coexistence |
| `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` | rewrite | Region-based VLM extraction (~300 LOC) |
| `packages/pdfsys-cli/src/pdfsys_cli/viz.py` | modify | Extend row schema with `extract.regions_extracted[]` + `extract.region_failures` |
| `packages/pdfsys-bench/viz/index.html` | modify | Routing trace text + new Per-region extraction card |
| `out/e2e_full_mineru3_regional/` | regenerate | Output of the full 150-PDF re-run with the new code |
| `out/viz_regional/` | regenerate | Viz bundle for the re-run |
| `docs/superpowers/specs/2026-05-16-vlm-region-based-design.md` | modify (§16) | Post-build note |

Untouched: `pdfsys-core`, `pdfsys-router`, `pdfsys-layout-analyser`, `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline`, `pdfsys-bench/annotation/*`, `pdfsys-cli/runner.py`, `pdfsys-cli/_vlm_config.py`, `pdfsys-cli/parquet_writer.py`, all YAMLs.

---

## Task 1: Spike — verify `MinerUClient` region-based API on one PDF

**Files:**
- Create: `out/mineru_region_spike_notes.md`
- Create (throwaway): `/tmp/mineru_region_spike.py`

This task does NOT modify any package code. It verifies the `MinerUClient.batch_content_extract` signature and behavior on one VLM-routed PDF, so Task 2 implements against verified ground truth.

### Why this spike

mineru's `MinerUClient` is documented but the relationship between `batch_content_extract`, `content_extract`, `two_step_extract` and the underlying model loading is not obvious from the API listing. We need to confirm:

1. Can we construct `MinerUClient(backend='transformers')` directly without going through `do_parse`?
2. Does `batch_content_extract([pil_image], ['equation'])` return a usable result?
3. Does this interfere with `ModelSingleton` if `do_parse` was called earlier in the same process?
4. What's the wall-time per region (informs whether batching matters)?

- [ ] **Step 1: Set up the env vars (mineru reads at construction)**

```bash
export MINERU_DEVICE_MODE=mps
export MINERU_MODEL_SOURCE=huggingface
```

- [ ] **Step 2: Pick a known VLM-routed PDF + write the spike script**

```bash
.venv/bin/python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet('out/e2e_full_mineru3/dataset.parquet')
vlm = df[df['extract_backend'] == 'vlm']
print('VLM PDFs:')
for _, row in vlm.head(3).iterrows():
    name = Path(row['pdf_path']).name
    for cand in Path('packages/pdfsys-bench').rglob(name):
        print(f'  {cand}')
        break
"
```

Pick `packages/pdfsys-bench/olmocr_bench_50/pdfs/old_scans_math/1_pg19.pdf` (used in the previous spike — known to have formula regions).

Write `/tmp/mineru_region_spike.py`:

```python
"""Region-based mineru spike — verify MinerUClient.batch_content_extract.

Crop layout regions from one VLM-routed PDF, feed each region to
mineru with the appropriate type, print the result. Confirms the API
shape that Task 2 will commit to.

Usage:
    .venv/bin/python /tmp/mineru_region_spike.py 2>&1 | tee /tmp/mineru_region_spike.log
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MINERU_DEVICE_MODE", "mps")
os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")


def main() -> None:
    import json
    from pathlib import Path
    import pymupdf
    from PIL import Image

    PDF = "packages/pdfsys-bench/olmocr_bench_50/pdfs/old_scans_math/1_pg19.pdf"
    print(f"[spike] PDF: {PDF}")

    # 1. Get the layout this PDF produced in the prior full run.
    import pandas as pd
    df = pd.read_parquet("out/e2e_full_mineru3/dataset.parquet")
    row = df[df["pdf_path"].str.contains("1_pg19.pdf")].iloc[0]
    sha = row["sha256"]
    layout_path = Path("out/e2e_full_mineru3/.cache/layout") / f"{sha}.json"
    # The actual filename pattern may have a suffix — glob it.
    candidates = list(Path("out/e2e_full_mineru3/.cache/layout").glob(f"{sha}*"))
    if not candidates:
        raise SystemExit(f"no layout cache for sha={sha}")
    layout_data = json.loads(candidates[0].read_text())
    pages = layout_data.get("pages") or []
    print(f"[spike] layout pages: {len(pages)}")
    total_regions = sum(len(p.get("regions", [])) for p in pages)
    print(f"[spike] total regions: {total_regions}")
    types_seen = sorted({r["type"] for p in pages for r in p.get("regions", [])})
    print(f"[spike] region types: {types_seen}")

    # 2. Render the first complex page to PIL.
    doc = pymupdf.open(PDF)
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    print(f"[spike] page image: {page_img.size}")
    doc.close()

    # 3. Crop each region on page 0.
    page_regions = next(p for p in pages if p.get("index", 0) == 0)
    REGION_TYPE_TO_MINERU = {
        "TEXT": "text", "TITLE": "text", "HEADER": "text", "FOOTER": "text",
        "REFERENCE": "text", "PAGE_NUMBER": "text",
        "TABLE": "table",
        "FORMULA": "equation",
        "IMAGE": "image", "FIGURE": "image",
    }
    crops = []
    types = []
    region_meta = []
    W, H = page_img.size
    for r in page_regions.get("regions", []):
        bbox = r.get("bbox", {})
        box = (
            int(bbox.get("x0", 0) * W),
            int(bbox.get("y0", 0) * H),
            int(bbox.get("x1", 0) * W),
            int(bbox.get("y1", 0) * H),
        )
        crop = page_img.crop(box)
        if crop.size[0] < 10 or crop.size[1] < 10:
            print(f"[spike]  skip region (too small): {r['type']} {crop.size}")
            continue
        crops.append(crop)
        types.append(REGION_TYPE_TO_MINERU.get(r["type"], "text"))
        region_meta.append((r["type"], box, crop.size))

    print(f"[spike] {len(crops)} crops prepared")

    # 4. Construct MinerUClient.
    from mineru_vl_utils import MinerUClient
    t0 = time.time()
    client = MinerUClient(backend="transformers")
    print(f"[spike] MinerUClient ready in {time.time()-t0:.1f}s")

    # 5. Batch extract.
    t0 = time.time()
    results = client.batch_content_extract(crops, types)
    print(f"[spike] batch_content_extract returned {len(results)} results in {time.time()-t0:.1f}s")

    # 6. Inspect.
    for i, (meta, res) in enumerate(zip(region_meta, results)):
        rtype, _box, crop_size = meta
        text = (res.text if res else "") or ""
        print(f"\n[spike] region {i}: type={rtype} crop={crop_size} mineru_type={types[i]}")
        print(f"  result repr: {type(res).__name__}; text len: {len(text)}")
        print(f"  text excerpt: {text[:200]!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the spike**

```bash
.venv/bin/python /tmp/mineru_region_spike.py 2>&1 | tee /tmp/mineru_region_spike.log
```

Wall time expectation: model load 5-20s (warm from prior run), then ~1-5s per region × ~14 regions = roughly 30-90 s total. If it stalls > 5 minutes without log output, kill and report BLOCKED.

- [ ] **Step 4: Write `out/mineru_region_spike_notes.md`**

Required content:

```markdown
# mineru region-based spike notes — 2026-05-16

## MinerUClient construction
- Direct construction works: `MinerUClient(backend='transformers')` returns a usable client.
- Model load wall time: <X> seconds (model already cached at ~/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/).
- Coexistence with ModelSingleton: <verified by re-running spike after a do_parse / not tested>.

## batch_content_extract behavior
- Signature: `batch_content_extract(images: list[PIL.Image], types: Sequence[str]) -> list[ExtractStr | None]`
- Per-region wall time: <X> seconds avg.
- Returns `ExtractStr` objects; `.text` field is the recognized content.

## Region type → mineru type mapping (verified)
| pdfsys RegionType | mineru type | sample output |
|---|---|---|
| TEXT | text | plain text |
| FORMULA | equation | `$$...$$` LaTeX |
| TABLE | table | `<table>...</table>` HTML (if present in this PDF) |
| IMAGE | image | caption text (if present in this PDF) |

## Quirks
- (any errors / surprises)

## Code template for Task 2's _extract_page_regions
(short snippet showing the verified call shape)
```

- [ ] **Step 5: Commit**

```bash
git add -f out/mineru_region_spike_notes.md
git commit -m "docs(spike): region-based mineru API surface verified"
```

---

## Task 2: Rewrite `extract.py` against verified API

**Files:**
- Rewrite: `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py`

**Required input:** `out/mineru_region_spike_notes.md` (from Task 1). The implementer subagent MUST be given the spike notes contents in their prompt — they cannot infer the API from training data.

- [ ] **Step 1: Read the spike notes**

```bash
cat out/mineru_region_spike_notes.md
```

Pin in mind: verified import path, `batch_content_extract` signature, model load behavior, and the region type → mineru type map.

- [ ] **Step 2: Read the existing extract.py to understand the contract**

```bash
.venv/bin/python -c "
import inspect
from pdfsys_parser_vlm import extract
print(inspect.getsource(extract.VlmParser))
" 2>&1 | head -120
```

Three public methods MUST be preserved unchanged externally:
- `extract(pdf_path, sha256=None) -> ExtractedDoc`
- `extract_bytes(pdf_bytes, sha256=None) -> ExtractedDoc`
- `extract_complex_pages(pdf_path, layout, sha256=None) -> ExtractedDoc`

Plus module-level `extract_doc` / `extract_doc_from_layout` convenience functions.

- [ ] **Step 3: Rewrite the file**

Replace `packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` entirely with:

```python
"""VLM extraction backend — region-based via mineru 3.x MinerUClient.

Replaces the prior do_parse-based implementation (2026-05-14). Layout
analyser (DocLayout-YOLO) is the single source of truth for which
regions exist on each page; mineru's job narrows to "recognize this
cropped region as type X".

Pipeline:
    1. Filter to complex pages (any TABLE / FORMULA region).
    2. For each complex page:
       a. Render the page to PIL via PyMuPDF (DPI 200).
       b. Crop each region's bbox from the page image.
       c. Build (crops, types) lists and call
          MinerUClient.batch_content_extract(crops, types).
       d. Each result.text becomes a Segment.
    3. merge_segments_to_markdown handles reading order.

Heavy dependencies (mineru, mineru_vl_utils, pymupdf, PIL) are imported
lazily inside the methods that need them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pdfsys_core import (
    Backend,
    BBox,
    ExtractedDoc,
    LayoutDocument,
    RegionType,
    Segment,
    VlmConfig,
    merge_segments_to_markdown,
)

# pdfsys RegionType → mineru type string for batch_content_extract.
# Verified against MinerUClient prompts dict (see 2026-05-14 spike).
_REGIONTYPE_TO_MINERU: dict[RegionType, str] = {
    RegionType.TEXT: "text",
    RegionType.TITLE: "text",
    RegionType.HEADER: "text",
    RegionType.FOOTER: "text",
    RegionType.REFERENCE: "text",
    RegionType.PAGE_NUMBER: "text",
    RegionType.TABLE: "table",
    RegionType.FORMULA: "equation",
    RegionType.IMAGE: "image",
    RegionType.FIGURE: "image",
}

# Minimum crop size; mineru VL rejects tiny images (verified Task-1 spike).
_MIN_CROP_PX = 10

# DPI for PDF page → image rendering. Matches pdfsys-layout-analyser default.
_DEFAULT_RENDER_DPI = 200


class VlmParser:
    """Region-based VLM extraction parser using mineru 3.x."""

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()
        self._client = None
        self._render_dpi = _DEFAULT_RENDER_DPI

    # ------------------------------------------------------------------ api

    def extract(
        self, pdf_path: str | Path, sha256: str | None = None
    ) -> ExtractedDoc:
        """Process an entire PDF — every page treated as 'complex' so all
        regions get extracted. Requires a precomputed LayoutDocument via
        extract_with_layout(); without one, raises an explicit error.
        """
        raise NotImplementedError(
            "Region-based VlmParser requires a LayoutDocument. "
            "Use extract_complex_pages(pdf_path, layout)."
        )

    def extract_bytes(
        self, pdf_bytes: bytes, sha256: str | None = None
    ) -> ExtractedDoc:
        """Same constraint as extract()."""
        raise NotImplementedError(
            "Region-based VlmParser requires a LayoutDocument. "
            "Use extract_complex_pages(pdf_path, layout)."
        )

    def extract_complex_pages(
        self,
        pdf_path: str | Path,
        layout: LayoutDocument,
        sha256: str | None = None,
    ) -> ExtractedDoc:
        """Extract every region on every page that has TABLE or FORMULA content.

        The runner only calls this method when Stage-B routes the PDF to
        VLM (which happens iff layout.has_complex_content is true). Within
        such pages, ALL region types (TEXT included) are extracted via
        mineru VL — keeping the lane single-engine.
        """
        path = Path(pdf_path)
        sha = sha256 or layout.sha256 or _sha256_of_file(path)

        complex_pages = {
            lp.index
            for lp in layout.pages
            if any(r.type in (RegionType.TABLE, RegionType.FORMULA) for r in lp.regions)
        }

        if not complex_pages:
            return ExtractedDoc(
                sha256=sha,
                backend=Backend.VLM,
                segments=(),
                markdown="",
                stats={
                    "vlm_engine": "mineru-3.x region-based",
                    "complex_pages": 0,
                    "reason": "no_complex_content",
                },
            )

        segments, page_failures, region_failures, region_type_counts = (
            self._run_vlm_per_region(path, layout, complex_pages)
        )

        markdown = merge_segments_to_markdown(tuple(segments))

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.VLM,
            segments=tuple(segments),
            markdown=markdown,
            stats={
                "vlm_engine": "mineru-3.x region-based",
                "vlm_model": self.config.model,
                "complex_pages": len(complex_pages),
                "complex_page_indices": sorted(complex_pages),
                "region_count": len(segments),
                "region_failures": region_failures,
                "region_type_counts": region_type_counts,
                "page_failures": page_failures,
                "char_count": len(markdown),
                "segment_count": len(segments),
            },
        )

    # --------------------------------------------------------------- internal

    def _ensure_client(self) -> Any:
        """Lazily construct MinerUClient. One per VlmParser instance."""
        if self._client is None:
            from mineru_vl_utils import MinerUClient  # noqa: PLC0415

            self._client = MinerUClient(backend="transformers", image_analysis=True)
        return self._client

    def _run_vlm_per_region(
        self, pdf_path: Path, layout: LayoutDocument, complex_pages: set[int]
    ) -> tuple[list[Segment], list[dict[str, Any]], int, dict[str, int]]:
        """Crop layout regions from complex pages, batch-extract via mineru.

        Returns (segments, page_failures, region_failures, region_type_counts).
        """
        import pymupdf  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        client = self._ensure_client()

        segments: list[Segment] = []
        page_failures: list[dict[str, Any]] = []
        region_failures = 0
        region_type_counts: dict[str, int] = {}

        doc = pymupdf.open(str(pdf_path))
        try:
            for layout_page in layout.pages:
                if layout_page.index not in complex_pages:
                    continue
                if layout_page.index >= len(doc):
                    page_failures.append({
                        "page": layout_page.index,
                        "error": "page index out of range",
                    })
                    continue

                page = doc[layout_page.index]
                pix = page.get_pixmap(dpi=self._render_dpi)
                page_img = Image.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples
                )
                W, H = page_img.size

                crops: list[Any] = []
                types: list[str] = []
                region_refs: list[Any] = []

                for region in layout_page.regions:
                    mineru_type = _REGIONTYPE_TO_MINERU.get(region.type, "text")
                    bbox = region.bbox
                    box = (
                        int(bbox.x0 * W),
                        int(bbox.y0 * H),
                        int(bbox.x1 * W),
                        int(bbox.y1 * H),
                    )
                    crop = page_img.crop(box)
                    if crop.size[0] < _MIN_CROP_PX or crop.size[1] < _MIN_CROP_PX:
                        region_failures += 1
                        continue
                    crops.append(crop)
                    types.append(mineru_type)
                    region_refs.append(region)

                if not crops:
                    continue

                try:
                    results = client.batch_content_extract(crops, types)
                except Exception as e:  # noqa: BLE001 — one page fails, others continue
                    page_failures.append({
                        "page": layout_page.index,
                        "error": f"{type(e).__name__}: {e}"[:200],
                    })
                    continue

                for region, result in zip(region_refs, results):
                    text = (getattr(result, "text", None) or "") if result else ""
                    text = text.strip()
                    if not text:
                        region_failures += 1
                        # Still emit a segment so the page's region list is
                        # complete in viz; just with empty content.
                    region_type_counts[region.type.value] = (
                        region_type_counts.get(region.type.value, 0) + 1
                    )
                    segments.append(
                        Segment(
                            index=len(segments),
                            backend=Backend.VLM,
                            page_index=region.page_idx if hasattr(region, "page_idx")
                                       else layout_page.index,
                            type=region.type,
                            content=text,
                            bbox=region.bbox,
                            source_region_id=getattr(region, "region_id", None),
                        )
                    )
        finally:
            doc.close()

        return segments, page_failures, region_failures, region_type_counts


# ---------------------------------------------------------------- convenience

def extract_doc(
    pdf_path: str | Path,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """Region-based VlmParser cannot extract without a LayoutDocument; raises."""
    parser = VlmParser(config=config)
    return parser.extract(pdf_path, sha256=sha256)


def extract_doc_from_layout(
    pdf_path: str | Path,
    layout: LayoutDocument,
    config: VlmConfig | None = None,
    sha256: str | None = None,
) -> ExtractedDoc:
    """One-shot convenience: extract complex pages via region-based mineru."""
    parser = VlmParser(config=config)
    return parser.extract_complex_pages(pdf_path, layout, sha256=sha256)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Verify the module imports cleanly + contract preserved**

```bash
.venv/bin/python -c "
from pdfsys_parser_vlm import VlmParser, extract_doc, extract_doc_from_layout
import pdfsys_parser_vlm.extract as ex
assert hasattr(ex.VlmParser, 'extract')
assert hasattr(ex.VlmParser, 'extract_bytes')
assert hasattr(ex.VlmParser, 'extract_complex_pages')
import inspect
src = inspect.getsource(ex)
assert 'do_parse' not in src, 'do_parse reference still present'
assert 'batch_content_extract' in src, 'batch_content_extract not used'
print('module surface OK, do_parse removed, batch_content_extract present')
"
```

Expected: prints `module surface OK, do_parse removed, batch_content_extract present`.

- [ ] **Step 5: Single-PDF smoke against the new code**

```bash
.venv/bin/python <<'PY'
from pdfsys_cli._vlm_config import ensure_mineru_env
ensure_mineru_env('mps')

def main():
    from pathlib import Path
    import json
    import pandas as pd
    from pdfsys_parser_vlm import VlmParser
    from pdfsys_core import VlmConfig, LayoutCache

    df = pd.read_parquet('out/e2e_full_mineru3/dataset.parquet')
    vlm = df[df['extract_backend'] == 'vlm'].iloc[0]
    name = Path(vlm['pdf_path']).name
    src = next(Path('packages/pdfsys-bench').rglob(name))
    sha = vlm['sha256']
    print(f'testing on: {src}, sha={sha[:12]}...')

    # Reload layout from cache.
    cache = LayoutCache(Path('out/e2e_full_mineru3/.cache/layout'))
    # Find the layout file (suffix unknown).
    matches = list(Path('out/e2e_full_mineru3/.cache/layout').glob(f'{sha}*'))
    if not matches:
        raise SystemExit(f'no layout cache for sha={sha}')
    layout = cache.load_from_path(matches[0]) if hasattr(cache, 'load_from_path') else None
    if layout is None:
        # Fallback: reconstruct via from_dict
        from pdfsys_core import LayoutDocument
        layout = LayoutDocument.from_dict(json.loads(matches[0].read_text()))

    parser = VlmParser(config=VlmConfig(model='mineru-2.5'))
    doc = parser.extract_complex_pages(src, layout)
    print(f'segments: {len(doc.segments)}')
    print(f'markdown chars: {len(doc.markdown)}')
    print(f'backend: {doc.backend.value}')
    print(f'stats keys: {list(doc.stats.keys())}')
    print(f"region_count: {doc.stats.get('region_count')}")
    print(f"region_failures: {doc.stats.get('region_failures')}")
    print(f"region_type_counts: {doc.stats.get('region_type_counts')}")
    print('--- markdown preview ---')
    print(doc.markdown[:500])

if __name__ == '__main__':
    main()
PY
```

Expected:
- `segments > 0`
- `markdown chars > 200`
- `region_count` matches the layout's region count for complex pages
- `region_failures` low (single digits typical)
- `region_type_counts` includes at least one of `TABLE` / `FORMULA`
- Markdown preview shows some recognized content (LaTeX or text)

If `markdown chars` is < 50 or segments == 0, the API binding is wrong — go back to the spike notes and fix `_REGIONTYPE_TO_MINERU` or the `batch_content_extract` call shape.

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py
git commit -m "feat(parser-vlm): rewrite extract.py for region-based parsing via MinerUClient"
```

---

## Task 3: Extend `viz.py` row schema for per-region data

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/viz.py`

- [ ] **Step 1: Add the `regions_extracted` and `region_failures` fields to row schema**

In `packages/pdfsys-cli/src/pdfsys_cli/viz.py`, locate the `rows.append({...})` block inside `_load_rows`. Currently the `extract` sub-dict has 4 keys. Replace that block (currently around line 95-110) with:

```python
        # Extract per-region content for VLM rows only (others have it null).
        regions_extracted = None
        region_failures = None
        if raw.get("extract_backend") == "vlm":
            regions_extracted = []
            # The runner serializes Segment.* to dataset.parquet's `markdown`
            # column but doesn't currently expose per-segment fields. Read
            # segment data from the layout cache + reconstruct from markdown
            # is not possible; instead, fall back to reading the run's
            # per-PDF segments from the JSONL summary if present. For now,
            # leave the list empty when raw doesn't carry it.
            # Future: add a `segments_excerpt` parquet column to expose this.
            seg_data = raw.get("segments_excerpt") or []
            for seg in seg_data:
                regions_extracted.append({
                    "page_idx": seg.get("page_index"),
                    "type": seg.get("type"),
                    "bbox": seg.get("bbox"),
                    "content_excerpt": (seg.get("content") or "")[:200],
                })
            region_failures = raw.get("region_failures")

        rows.append({
            "id": short_id,
            "sha256": sha,
            "pdf_path": pdf_path,
            "pdf_basename": basename,
            "source": _source_from_path(pdf_path),
            "num_pages": raw.get("num_pages") or 0,
            "backend": raw.get("backend"),
            "stage_b_backend": raw.get("stage_b_backend"),
            "ocr_prob": raw.get("ocr_prob"),
            "is_form": bool(raw.get("is_form")),
            "is_encrypted": bool(raw.get("is_encrypted")),
            "garbled_text_ratio": raw.get("garbled_text_ratio") or 0.0,
            "layout": layout_block,
            "extract": {
                "backend": raw.get("extract_backend"),
                "markdown_chars": raw.get("markdown_chars") or 0,
                "markdown_excerpt": md_excerpt,
                "markdown_path": md_path,
                "regions_extracted": regions_extracted,
                "region_failures": region_failures,
            },
            "quality": {
                "score": raw.get("quality_score"),
                "model": raw.get("quality_model"),
                "kept": bool(raw.get("kept")),
            },
            "error_class": raw.get("error_class"),
            "error_message": raw.get("error_message"),
            "preview_key": _preview_key_from_path(pdf_path),
            "wall_ms_total": raw.get("wall_ms_total") or 0.0,
        })
```

(Two new keys: `regions_extracted` and `region_failures` inside the `extract` dict.)

**Note for the implementer:** `raw.get("segments_excerpt")` and `raw.get("region_failures")` will return `None` on data produced by the CURRENT runner (which doesn't write those parquet columns). This is intentional fallback — the row schema is forward-compatible. The actual per-region data populates correctly when Task 4 lands its parquet schema extension.

For this task, just commit the schema change. Task 4 wires up the parquet side.

- [ ] **Step 2: Verify module still imports**

```bash
.venv/bin/python -c "from pdfsys_cli import viz; print('OK')"
.venv/bin/python -m pdfsys_cli.viz --help
```

Expected: `OK` plus the existing help text.

- [ ] **Step 3: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/viz.py
git commit -m "feat(viz): row schema adds extract.regions_extracted + region_failures"
```

---

## Task 4: Expose per-segment content + region_failures in the parquet output

**Files:**
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/runner.py`
- Modify: `packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py`

The viz schema (Task 3) reads `raw.get("segments_excerpt")` and `raw.get("region_failures")` from parquet rows. Those columns don't exist yet. Add them.

- [ ] **Step 1: Add fields to `DocResult` in `runner.py`**

Locate the `DocResult` dataclass (currently around lines 22-60). After the `quality_model` field add:

```python
    # extract internals (new for region-based VLM viz)
    segments_excerpt: list[dict] = field(default_factory=list)
    region_failures: int | None = None
```

- [ ] **Step 2: Populate the fields after extract stage**

Locate the function `_stage_extract` in `runner.py` (currently around line 282-355). After the section that sets `row.extract_stats = dict(extracted.stats)` in each branch (mupdf, pipeline, vlm), populate the two new fields. Specifically, in the `vlm` branch (around line 327-333), after `row.extract_stats = dict(extracted.stats)`, add:

```python
            row.region_failures = extracted.stats.get("region_failures")
            row.segments_excerpt = [
                {
                    "page_index": s.page_index,
                    "type": s.type.value,
                    "bbox": [s.bbox.x0, s.bbox.y0, s.bbox.x1, s.bbox.y1] if s.bbox else None,
                    "content": (s.content or "")[:200],
                }
                for s in extracted.segments
            ]
```

- [ ] **Step 3: Add the parquet columns**

In `packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py`, locate the `SCHEMA` constant (currently 21 columns). Add two new columns at the end (before `kept`/`wall_ms_total` — keep those last):

```python
SCHEMA = pa.schema(
    [
        ("pdf_path", pa.string()),
        ("sha256", pa.string()),
        ("backend", pa.string()),
        ("stage_b_backend", pa.string()),
        ("ocr_prob", pa.float64()),
        ("num_pages", pa.int32()),
        ("is_form", pa.bool_()),
        ("garbled_text_ratio", pa.float64()),
        ("is_encrypted", pa.bool_()),
        ("layout_model", pa.string()),
        ("layout_num_regions", pa.int32()),
        ("layout_has_complex", pa.bool_()),
        ("extract_backend", pa.string()),
        ("markdown", pa.string()),
        ("markdown_chars", pa.int64()),
        ("quality_score", pa.float64()),
        ("quality_model", pa.string()),
        ("error_class", pa.string()),
        ("error_message", pa.string()),
        ("segments_excerpt", pa.string()),  # JSON-encoded list[dict]
        ("region_failures", pa.int32()),
        ("kept", pa.bool_()),
        ("wall_ms_total", pa.float64()),
    ]
)
```

Two new columns: `segments_excerpt` (JSON-encoded string to avoid nested parquet types) and `region_failures` (int32, nullable).

- [ ] **Step 4: Populate the new columns in `write_row`**

In `parquet_writer.py`, locate the `write_row` method's `data` dict (currently ~25 keys). Add two entries:

```python
        import json as _json  # noqa: PLC0415

        data: dict[str, Any] = {
            "pdf_path": row.pdf_path,
            "sha256": row.sha256,
            "backend": row.backend,
            "stage_b_backend": row.stage_b_backend,
            "ocr_prob": row.ocr_prob,
            "num_pages": row.num_pages,
            "is_form": row.is_form,
            "garbled_text_ratio": row.garbled_text_ratio,
            "is_encrypted": row.is_encrypted,
            "layout_model": row.layout_model,
            "layout_num_regions": row.layout_num_regions,
            "layout_has_complex": row.layout_has_complex,
            "extract_backend": row.extract_backend,
            "markdown": md_value,
            "markdown_chars": row.markdown_chars,
            "quality_score": row.quality_score,
            "quality_model": row.quality_model,
            "error_class": row.error_class,
            "error_message": row.error_message,
            "segments_excerpt": (
                _json.dumps(row.segments_excerpt, ensure_ascii=False)
                if row.segments_excerpt else None
            ),
            "region_failures": row.region_failures,
            "kept": kept,
            "wall_ms_total": wall_total,
        }
```

- [ ] **Step 5: Update `viz.py` to decode `segments_excerpt`**

In `viz.py`'s `_load_rows`, the row currently reads `raw.get("segments_excerpt") or []`. After Task 4's parquet change, the value is a JSON string. Decode it:

In `packages/pdfsys-cli/src/pdfsys_cli/viz.py`, find the lines added in Task 3:

```python
        if raw.get("extract_backend") == "vlm":
            regions_extracted = []
            seg_data = raw.get("segments_excerpt") or []
            for seg in seg_data:
                ...
```

Replace `seg_data = raw.get("segments_excerpt") or []` with:

```python
            raw_seg = raw.get("segments_excerpt")
            if isinstance(raw_seg, str) and raw_seg:
                try:
                    seg_data = json.loads(raw_seg)
                except json.JSONDecodeError:
                    seg_data = []
            else:
                seg_data = raw_seg or []
```

- [ ] **Step 6: Verify all 3 modules still import**

```bash
.venv/bin/python -c "
from pdfsys_cli.runner import DocResult, run
from pdfsys_cli.parquet_writer import SCHEMA, ParquetSink
from pdfsys_cli import viz
print('runner OK, parquet OK, viz OK')
print('schema cols:', len(SCHEMA))
print('DocResult has segments_excerpt:', 'segments_excerpt' in [f.name for f in __import__('dataclasses').fields(DocResult)])
print('DocResult has region_failures:', 'region_failures' in [f.name for f in __import__('dataclasses').fields(DocResult)])
"
```

Expected:
- `runner OK, parquet OK, viz OK`
- `schema cols: 23`
- Both flags true.

- [ ] **Step 7: Smoke regression — mupdf-only run (parquet schema still works for non-VLM)**

```bash
rm -rf /tmp/_pdfsys_t4_reg
.venv/bin/pdfsys run --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
  --out-dir /tmp/_pdfsys_t4_reg \
  --stages router,extract,parquet \
  --limit 2 \
  --no-quality
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('/tmp/_pdfsys_t4_reg/dataset.parquet')
print(df[['pdf_path', 'extract_backend', 'segments_excerpt', 'region_failures']].to_string(index=False))
print('columns:', list(df.columns))
"
```

Expected:
- 2 rows, `extract_backend = mupdf` for both
- `segments_excerpt = None` (not populated for mupdf rows)
- `region_failures = None`
- 23 columns total.

- [ ] **Step 8: Commit**

```bash
git add packages/pdfsys-cli/src/pdfsys_cli/runner.py \
        packages/pdfsys-cli/src/pdfsys_cli/parquet_writer.py \
        packages/pdfsys-cli/src/pdfsys_cli/viz.py
git commit -m "feat(cli): parquet now carries segments_excerpt + region_failures for VLM rows"
```

---

## Task 5: Update viz HTML — routing trace text + per-region card

**Files:**
- Modify: `packages/pdfsys-bench/viz/index.html`

- [ ] **Step 1: Update Stage-B routing-trace text**

In `packages/pdfsys-bench/viz/index.html`, locate the `routingTrace` function. Inside it, find the line:

```javascript
        branch(hasComplex === true && vlmOn, 'has_complex = true &amp; vlm_enabled = true', `<strong>VLM</strong> (mineru)`),
```

Replace with:

```javascript
        branch(hasComplex === true && vlmOn, 'has_complex = true &amp; vlm_enabled = true', `<strong>VLM</strong> (mineru, region-based via batch_content_extract)`),
```

- [ ] **Step 2: Add the Per-region extraction card**

In the same file, locate `async function renderStages(row) {`. After the existing `Extract (...)` stage card push (the one with `<div class="md" id="md-host">loading markdown…</div>`), add:

```javascript
    if (row.extract?.regions_extracted && row.extract.regions_extracted.length > 0) {
      const regions = row.extract.regions_extracted;
      const typeCounts = {};
      for (const r of regions) typeCounts[r.type] = (typeCounts[r.type] || 0) + 1;
      const typeSummary = Object.entries(typeCounts)
        .map(([t, n]) => `${t}:${n}`).join(' · ');
      const failures = row.extract.region_failures || 0;

      parts.push(`
        <div class="stage-card"><h3>Per-region extraction</h3>
          <div class="kv"><span class="k">regions extracted</span><span class="v">${regions.length} (${typeSummary})</span></div>
          <div class="kv"><span class="k">region_failures</span><span class="v">${failures}</span></div>
          <div style="margin-top:8px;max-height:400px;overflow-y:auto">
            <table class="region-table"><thead>
              <tr><th>#</th><th>page</th><th>type</th><th>bbox</th><th>content excerpt</th></tr>
            </thead><tbody>
              ${regions.map((r, i) => `
                <tr>
                  <td>${i}</td>
                  <td>${r.page_idx}</td>
                  <td>${r.type}</td>
                  <td style="font-family:monospace;font-size:10px">${r.bbox ? r.bbox.map(v => v.toFixed(3)).join(', ') : '—'}</td>
                  <td style="font-family:monospace;font-size:11px;max-width:300px;word-break:break-all">${(r.content_excerpt || '').slice(0, 200) || '<em style="color:var(--dim)">empty</em>'}</td>
                </tr>
              `).join('')}
            </tbody></table>
          </div>
        </div>`);
    }
```

This block goes between the `Extract` card push and the `Quality` card push.

- [ ] **Step 3: Verify HTML brace + paren balance**

```bash
.venv/bin/python -c "
html = open('packages/pdfsys-bench/viz/index.html').read()
import re
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
b_open = sum(s.count('{') for s in scripts)
b_close = sum(s.count('}') for s in scripts)
p_open = sum(s.count('(') for s in scripts)
p_close = sum(s.count(')') for s in scripts)
print(f'braces: {b_open}/{b_close}, parens: {p_open}/{p_close}, lines: {html.count(chr(10))+1}')
assert b_open == b_close and p_open == p_close, 'unbalanced'
print('balanced')
"
```

Expected: balanced.

- [ ] **Step 4: Commit**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): per-region extraction card + region-based routing trace text"
```

---

## Task 6: Re-run full 150 PDFs with the new region-based VLM

**Files:**
- Output: `out/e2e_full_mineru3_regional/`

- [ ] **Step 1: Clear stale output**

```bash
rm -rf out/e2e_full_mineru3_regional
```

- [ ] **Step 2: Kick off the full run**

```bash
nohup .venv/bin/pdfsys run -c pdfsys.full.yaml \
  --out-dir out/e2e_full_mineru3_regional \
  > out/_e2e_full_mineru3_regional.log 2>&1 &
echo $! > /tmp/_pdfsys_regional.pid
echo "PID $(cat /tmp/_pdfsys_regional.pid) at $(date +%H:%M:%S)"
```

Wall time expectation: **30-90 minutes** (10 VLM PDFs × ~3-9 min each at region-by-region rates vs. previous ~30s/PDF whole-page). Pipeline + MUPDF paths unchanged → ~104 + 36 PDFs take ~3 min as before.

- [ ] **Step 3: Monitor progress**

Run a `Monitor` loop watching `out/_e2e_full_mineru3_regional.log` and `out/e2e_full_mineru3_regional/results.jsonl` — emit on each PDF processed:

```bash
PID=$(cat /tmp/_pdfsys_regional.pid)
prev=0
while ps -p "$PID" > /dev/null 2>&1; do
  cur=$(wc -l < out/e2e_full_mineru3_regional/results.jsonl 2>/dev/null | tr -d ' ')
  cur=${cur:-0}
  if [ "$cur" != "$prev" ]; then
    echo "[$(date +%H:%M:%S)] processed=${cur}/150"
    prev=$cur
  fi
  grep -E "Traceback|RuntimeError|OOM|FAILED" out/_e2e_full_mineru3_regional.log 2>/dev/null | tail -1 | while read line; do
    echo "[ERR] $line"
  done
  sleep 30
done
echo "[$(date +%H:%M:%S)] EXITED"
tail -30 out/_e2e_full_mineru3_regional.log
```

Expected: 150 processed, no Traceback, exit normally.

- [ ] **Step 4: Final summary**

```bash
cat out/e2e_full_mineru3_regional/results.summary.json
```

Expected: 150 PDFs, `by_backend = {mupdf: 104, pipeline: 36, vlm: 10}`, `num_errors ≤ 2`.

- [ ] **Step 5: No commit yet** — Task 7 commits after verification.

---

## Task 7: Verify acceptance + viz + post-build note

**Files:**
- Modify: `docs/superpowers/specs/2026-05-16-vlm-region-based-design.md` (§16 post-build note)
- Output: `out/viz_regional/`

- [ ] **Step 1: Run the 5 spec §13 acceptance queries**

```bash
.venv/bin/python <<'PY'
import pandas as pd
df = pd.read_parquet('out/e2e_full_mineru3_regional/dataset.parquet')
prev = pd.read_parquet('out/e2e_full_mineru3/dataset.parquet')

print('=== Gate 1: row count ===')
print(f'rows: {len(df)} (expect 150)')

print('\n=== Gate 2: errors ===')
errs = df['error_class'].notnull().sum()
print(f'errors: {errs} (expect ≤ 2)')
if errs:
    print(df[df['error_class'].notnull()][['pdf_path','error_class','error_message']].to_string(index=False))

print('\n=== Gate 3: backend split ===')
print(df['extract_backend'].fillna('NULL').value_counts().to_string())

print('\n=== Gate 4: VLM markdown distribution vs previous ===')
vlm_new = df[df['extract_backend'] == 'vlm']
vlm_old = prev[prev['extract_backend'] == 'vlm']
print('new vlm md_chars: min={} med={} max={}'.format(
    int(vlm_new['markdown_chars'].min()),
    int(vlm_new['markdown_chars'].median()),
    int(vlm_new['markdown_chars'].max()),
))
print('old vlm md_chars: min={} med={} max={}'.format(
    int(vlm_old['markdown_chars'].min()),
    int(vlm_old['markdown_chars'].median()),
    int(vlm_old['markdown_chars'].max()),
))

print('\n=== Gate 5: VLM rows with LaTeX or HTML table ===')
def has_complex_content(s):
    if not isinstance(s, str): return False
    return ('\\' in s) or ('$' in s) or ('<table' in s)
new_hit = sum(1 for v in vlm_new['markdown'] if has_complex_content(v))
old_hit = sum(1 for v in vlm_old['markdown'] if has_complex_content(v))
print(f'new: {new_hit}/10 ; old: {old_hit}/10')
print(f'(expect new ≥ {max(8, old_hit)})')

print('\n=== region-based specific ===')
# Decode segments_excerpt JSON to count regions per VLM row
import json
total_regions = 0
total_failures = 0
for _, row in vlm_new.iterrows():
    se = row.get('segments_excerpt')
    if isinstance(se, str) and se:
        try:
            total_regions += len(json.loads(se))
        except json.JSONDecodeError:
            pass
    rf = row.get('region_failures')
    if rf is not None and not pd.isna(rf):
        total_failures += int(rf)
print(f'total regions extracted: {total_regions}')
print(f'total region failures: {total_failures}')
print(f'failure rate: {total_failures/max(1,total_regions):.1%}')
print(f'(expect < 10%)')
PY
```

Pass criteria:
- Gate 1: 150 ✓
- Gate 2: errors ≤ 2 ✓
- Gate 3: mupdf=104 / pipeline=36 / vlm=10 ✓
- Gate 4: VLM median ≥ 1500 (or ≥ previous run's median)
- Gate 5: new ≥ max(8, old_hit) hits
- Region failure rate < 10%

If any gate fails, STOP and report — likely a binding issue in Task 2's code that needs a fix subagent.

- [ ] **Step 2: Generate viz bundle**

```bash
rm -rf out/viz_regional
.venv/bin/pdfsys visualize -r out/e2e_full_mineru3_regional -o out/viz_regional
ls -la out/viz_regional/
du -h out/viz_regional/viz_data.json out/viz_regional/page1
```

Expected: bundle generated, ~20 MB total.

- [ ] **Step 3: Browser sanity (manual or curl + grep)**

If a server is running on 8765 (from a prior task), kill and restart on the new bundle:

```bash
lsof -ti:8765 2>/dev/null | xargs -r kill 2>/dev/null
cd out/viz_regional && nohup python3 -m http.server 8765 > /tmp/_viz_server.log 2>&1 &
echo $! > /tmp/_viz_server.pid
cd -
sleep 1
curl -fsS http://localhost:8765/index.html -o /dev/null -w "viz server: HTTP %{http_code}\n"
```

Manual browser checks (open http://localhost:8765/):
- VLM row's detail view shows a "Per-region extraction" card with N regions listed.
- Routing trace's Stage-B step reads "VLM (mineru, region-based via batch_content_extract)".
- At least one region's `content excerpt` contains recognizable LaTeX (`\frac`, `\int`, etc.) or HTML (`<table>`).

- [ ] **Step 4: Append §16 post-build note to the spec**

Edit `docs/superpowers/specs/2026-05-16-vlm-region-based-design.md`. Replace the `## 16 · Post-build note (to be filled in)` line with:

```markdown
## 16 · Post-build note · 2026-05-16

### Engine
- MinerUClient constructed directly (no do_parse).
- Wall time per VLM PDF: <X> seconds (vs ~30s prior whole-page).
- Total full-run wall time: <T> minutes (vs 15 min prior).

### Acceptance gates
| Gate | Value | Result |
|---|---|---|
| 1 — Row count | 150 | PASS |
| 2 — Errors | <N> | PASS |
| 3 — Backend split | mupdf=104 / pipeline=36 / vlm=10 | PASS |
| 4 — VLM markdown_chars (median) | new=<Z>, old=<Z'> | PASS |
| 5 — VLM rows with LaTeX/HTML | new=<H>/10, old=<H'>/10 | PASS |
| Region failure rate | <R>% | PASS (< 10%) |

### Region extraction stats
- Total regions extracted across 10 VLM PDFs: <N>
- Region failures (empty / too-small): <M>
- Region type distribution: <{TEXT: ..., FORMULA: ..., TABLE: ...}>

### Notable observations
- (any surprises, e.g. one PDF with disproportionate region count)
- (any regions producing empty content)

### Open follow-ups
- ...
```

Fill in the actual numbers from Tasks 6 and 7.

- [ ] **Step 5: Final commit**

```bash
git add docs/superpowers/specs/2026-05-16-vlm-region-based-design.md
git commit -m "docs(spec): VLM region-based post-build note"
```

---

## Self-review notes

**Spec coverage:**
- §1 goal (region-based VLM) → Tasks 1 + 2
- §3 architecture (new flow) → Task 2 `_run_vlm_per_region`
- §4 region-type map → Task 2 `_REGIONTYPE_TO_MINERU`
- §5 MinerUClient lifecycle → Task 2 `_ensure_client`
- §6 region cropping → Task 2 (inline crop with min-size guard)
- §7 page-level batching → Task 2 (loop per page)
- §8 reading order via merge_segments_to_markdown → already in Task 2 flow
- §9 stats additions → Task 2 (`stats` dict in `extract_complex_pages`)
- §10 error handling → Task 2 (try/except around `batch_content_extract`)
- §11 viz updates → Tasks 3 + 5
- §12 new output dir → Task 6
- §13 acceptance → Task 7 step 1
- §14 files touched → matches file map above
- §15 definition of done → Task 7 satisfies all bullets

**Placeholders:**
- §16 post-build note has `<X>`, `<N>` fill-ins — these are intentional template placeholders to be replaced with actuals at Task 7 step 4.
- No "TBD" or "implement later" anywhere else.

**Type consistency:**
- `region_failures: int | None` consistent: declared in `DocResult` (Task 4 Step 1), populated in `_stage_extract` (Task 4 Step 2), serialized in parquet (Task 4 Step 3 + 4), decoded in viz (Task 4 Step 5), rendered in HTML (Task 5 Step 2).
- `segments_excerpt`: `list[dict]` on `DocResult`, JSON-string in parquet, decoded back to list in viz, rendered in HTML. Encoding boundary explicit in Task 4 Steps 4 and 5.
- `MinerUClient.batch_content_extract([crops], [types])` shape matches Task 2's verified spike output.

**Soft assumptions:**
- Task 1 spike will discover the precise `MinerUClient` construction args. If the spike finds that `image_analysis=True` is wrong or model_path needs to be passed explicitly, Task 2 must adapt — left flexible.
- Task 6's wall-time estimate (30-90 min) is a guess based on per-region inference being slower than per-page. If actual is much longer, may need to interrupt and reduce DPI or skip TEXT regions.
