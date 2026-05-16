# VLM Lane — Region-Based Parsing Design

**Date:** 2026-05-16
**Builds on:** `docs/superpowers/specs/2026-05-14-vlm-mineru-2x-migration-design.md` (mineru 3.x already integrated)
**Scope:** Rewrite `pdfsys-parser-vlm/extract.py` to feed mineru VL with **layout-region crops** rather than whole PDF pages. Layout analyser (DocLayout-YOLO) becomes the single source of truth for which regions exist; mineru's job narrows to "recognize this cropped region as type X". All other components (router, layout, pipeline lane / RapidOCR, runner, ParquetSink, configs) stay untouched.
**Out of scope:** pipeline lane changes (RapidOCR stays), batching/streaming optimizations, error retry, reading-order post-processing.

---

## 1 · Goal

For VLM-routed PDFs:

1. Stop calling `mineru.cli.common.do_parse` (which does its own internal layout + recognition on whole pages).
2. Instead, use the layout already produced by `pdfsys-layout-analyser` to crop each region from the PDF, and call `MinerUClient.batch_content_extract(images, types)` once per page to recognize all regions on that page in a single batch.
3. Assemble per-region results back into the existing `ExtractedDoc` / `Segment` contract.

This makes DocLayout-YOLO the authoritative layout signal across the whole VLM lane, and lets us see/diagnose exactly what mineru produced for each region in the viz UI.

## 2 · Non-goals

- Replace pipeline lane (RapidOCR stays — simple layouts are fast enough as-is).
- Hybrid per-region routing inside the VLM lane (every region on a complex page goes through mineru — the simpler P1 from brainstorm).
- Change Stage-B routing logic.
- Optimize mineru model loading or process management.
- Add error-class-specific retry; one failed region produces empty content, page continues.

## 3 · Architecture comparison

### Current (post 2026-05-14 mineru-2x migration)

```
ExtractedDoc = VlmParser.extract_complex_pages(pdf, layout):
  └─ _run_mineru(pdf_bytes, complex_pages):
       └─ do_parse(output_dir, [pdf_bytes], backend='vlm-transformers', ...)
            └─ mineru runs ITS OWN layout + content extraction on WHOLE pages,
               writes content_list.json to tmpdir
       └─ Read content_list.json, filter to complex_pages, build Segments
```

DocLayout-YOLO's layout output is used **only** for the Stage-B decision (complex / not complex). The actual region cropping is mineru's internal job.

### Target (this spec)

```
ExtractedDoc = VlmParser.extract_complex_pages(pdf, layout):
  └─ For each page in complex_pages:
       └─ page_image = render_pdf_page(pdf, page_idx, dpi=200)  # PIL.Image
       └─ For each region on the page:
            crop = page_image.crop(denormalize(region.bbox, page_w, page_h))
            mineru_type = REGION_TYPE_TO_MINERU[region.type]
            (collect crop + mineru_type)
       └─ results = client.batch_content_extract(crops, types)  # batch GPU pass
       └─ For each (region, result): build Segment with content=result.text
  └─ markdown = merge_segments_to_markdown(segments)
```

DocLayout-YOLO produces the canonical region list. mineru handles **only** per-region recognition.

## 4 · Region-type → mineru type mapping

Verified against `MinerUClient.prompts` (default dict, see spike notes 2026-05-14):

| pdfsys `RegionType` | mineru `type` arg | Result format |
|---|---|---|
| `TEXT` | `'text'` | plain text |
| `TITLE` | `'text'` | plain text (mineru has no distinct title type) |
| `HEADER` | `'text'` | plain text |
| `FOOTER` | `'text'` | plain text |
| `TABLE` | `'table'` | HTML (`<table>...</table>`) |
| `FORMULA` | `'equation'` | LaTeX wrapped in `$$...$$` |
| `IMAGE` | `'image'` | natural-language caption |
| `FIGURE` (alias) | `'image'` | natural-language caption |
| `REFERENCE` | `'text'` | plain text |
| `PAGE_NUMBER` | `'text'` | plain text |

Unknown types default to `'text'`. The map is encoded as `_REGIONTYPE_TO_MINERU` constant in extract.py.

## 5 · `MinerUClient` lifecycle

```python
class VlmParser:
    def __init__(self, config=None):
        self.config = config or VlmConfig()
        self._client = None
        self._render_dpi = 200  # matches layout-analyser default

    def _ensure_client(self):
        if self._client is None:
            from mineru_vl_utils import MinerUClient  # lazy import
            self._client = MinerUClient(
                backend='transformers',
                # model_path=None → MinerUClient pulls from HF cache via ModelSingleton
                image_analysis=True,
            )
        return self._client
```

One `MinerUClient` per `VlmParser` instance (which is one per `run()` call). Model weights load once.

**Risk:** `mineru.backend.vlm.vlm_analyze.doc_analyze` also uses a `ModelSingleton` to cache the model. If a process called `do_parse` previously and now constructs a direct `MinerUClient`, there's potential for double-load. Spike at start of implementation to confirm; fall back to keeping `ModelSingleton` if needed.

## 6 · Region cropping

```python
def _crop_region(page_image: PIL.Image, bbox: BBox) -> PIL.Image | None:
    """Crop bbox (normalized [0,1]) from page_image. Returns None if too small."""
    W, H = page_image.size
    px = (
        int(bbox.x0 * W),
        int(bbox.y0 * H),
        int(bbox.x1 * W),
        int(bbox.y1 * H),
    )
    crop = page_image.crop(px)
    if crop.size[0] < 10 or crop.size[1] < 10:
        return None  # mineru rejects tiny images; skip silently
    return crop
```

DPI choice: 200 to match `pdfsys-layout-analyser`'s default render DPI. Matches mineru VL's expected input scale (the underlying model is Qwen2-VL, comfortable with 224-1500 px region images).

Cropping uses PIL.Image.crop, no resizing. mineru VL handles resizing internally.

## 7 · Page-level batching

Per page (not per PDF) batch:

```python
for page_idx in complex_pages:
    page_img = _render_pdf_page(pdf_path, page_idx, dpi=self._render_dpi)
    crops, types, region_refs = [], [], []
    for region in regions_on_page:
        crop = _crop_region(page_img, region.bbox)
        if crop is None:
            stats['region_failures'] += 1
            continue
        crops.append(crop)
        types.append(_REGIONTYPE_TO_MINERU.get(region.type, 'text'))
        region_refs.append(region)

    if not crops:
        continue

    results = client.batch_content_extract(crops, types)
    for region, result in zip(region_refs, results):
        text = result.text if result else ''
        segments.append(Segment(
            index=len(segments),
            backend=Backend.VLM,
            page_index=region.page_idx,
            type=region.type,
            content=text.strip(),
            bbox=region.bbox,
            source_region_id=region.region_id,
        ))
```

Why per-page (not per-PDF) batch: simpler memory accounting, lets one bad page fail in isolation without losing the rest. Mineru's batch dispatch happens GPU-side anyway; we don't lose throughput in practice for the sizes here (typical complex page: 5-30 regions).

## 8 · Reading order

`pdfsys_core.merge_segments_to_markdown(segments)` already sorts segments by `(page_index, bbox.y0, bbox.x0)` before joining. We do not need to repeat that work. The list of segments we produce from the loop above is in DocLayout-YOLO's emit order (which may not be reading order); we trust `merge_segments_to_markdown` to fix it.

## 9 · `ExtractedDoc.stats` additions

Added keys (existing keys stay):

```python
stats['vlm_engine'] = 'mineru-3.x region-based'   # was 'mineru-3.x'
stats['region_count'] = len(segments)             # how many regions we actually extracted
stats['region_failures'] = N                      # crops that were too small / mineru returned empty
stats['region_type_counts'] = {'TEXT': 8, 'TABLE': 2, 'FORMULA': 4}  # by type, for diagnostics
```

These show up in viz under the Extract card.

## 10 · Error handling

| Failure mode | Behavior |
|---|---|
| Single region crop too small | Skip region, increment `region_failures`, continue |
| Single region's `batch_content_extract` returns None | Empty content for that segment, increment `region_failures`, continue |
| `batch_content_extract` raises on a whole batch (page) | Page contributes 0 segments; append `{"page": idx, "error": str(e)[:200]}` to `stats['page_failures']` list; OTHER pages continue |
| `MinerUClient` construction fails | `ExtractedDoc` not produced; runner records `error_class='extract_vlm'`, `error_message=str(e)` truncated 500 chars (existing path) |
| PDF page render fails | Same as `MinerUClient` failure — `error_class='extract_vlm'` |

The existing runner-level error capture (split into `error_class` / `error_message` from the prior spec) is unchanged.

## 11 · Viz updates

### 11.1 `viz.py` CLI

Extend the row schema (only for VLM rows; other rows leave `regions_extracted: null`):

```json
"extract": {
  "backend": "vlm",
  "markdown_chars": 1514,
  "markdown_excerpt": "...",
  "markdown_path": "markdown/<sha>.md",
  "regions_extracted": [
    {"page_idx": 0, "type": "TEXT", "bbox": [0.1, 0.2, 0.9, 0.3], "content_excerpt": "Let frac{x^2}..."},
    {"page_idx": 0, "type": "FORMULA", "bbox": [0.3, 0.4, 0.7, 0.5], "content_excerpt": "$$x^2 = A(...)$$"},
    ...
  ],
  "region_failures": 0
}
```

`content_excerpt` is the first 200 chars of `Segment.content` so viz_data.json doesn't bloat.

### 11.2 `index.html`

- Routing trace step 3: when path goes VLM, change description to
  `"has_complex = true & vlm_enabled = true → VLM (mineru, region-based via batch_content_extract)"`
- Detail view: when `regions_extracted` is present, render a new card "**Per-region extraction (N regions)**" with a table of `# | page | type | bbox | content_excerpt`. Each row also shows the region's bbox highlighted on the preview when hovered (stretch goal — drop if it complicates).

## 12 · Output directory

New full run goes to `out/e2e_full_mineru3_regional/`. The prior `out/e2e_full_mineru3/` (current production output) is preserved so we can diff later.

## 13 · Acceptance — re-run 150 PDFs

```sql
SELECT count(*) FROM 'out/e2e_full_mineru3_regional/dataset.parquet';
-- expected: 150

SELECT count(*) FROM ... WHERE error_class IS NOT NULL;
-- expected: ≤ 2 (allow a couple of region-failure-dominated rows; previous run had 0)

SELECT extract_backend, count(*) FROM ... GROUP BY extract_backend;
-- expected: mupdf=104, pipeline=36, vlm=10

SELECT extract_backend, min(markdown_chars), avg(markdown_chars), max(markdown_chars)
FROM ... GROUP BY extract_backend;
-- expected for vlm: median ≥ 1500 (previous run baseline); max ≥ 2300 (previous run baseline)

SELECT count(*) FROM ... WHERE extract_backend = 'vlm' AND (markdown LIKE '%\\\\%' OR markdown LIKE '%$%' OR markdown LIKE '%<table%');
-- expected: ≥ 8 / 10 (matches or exceeds prior run)
```

Plus new acceptance specific to region mode:

```sql
-- Every VLM row has at least one region extracted (else something silently broke)
SELECT count(*) FROM ... WHERE extract_backend = 'vlm' AND markdown_chars > 0;
-- expected: ≥ 9 / 10 (the 14-char outlier may still produce 1 region)

-- region failures bounded
-- (Reading from results.summary.json: total_region_failures ≤ 10% of total_regions)
```

## 14 · Files touched

```
MODIFIED
packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py    rewritten ~300 LOC
packages/pdfsys-cli/src/pdfsys_cli/viz.py                       add regions_extracted to row schema
packages/pdfsys-bench/viz/index.html                            routing trace text + new per-region card

NEW
docs/superpowers/specs/2026-05-16-vlm-region-based-design.md    this file
```

Untouched: `pdfsys-core`, `pdfsys-router`, `pdfsys-layout-analyser`, `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline`, `pdfsys-bench/annotation/*`, `pdfsys-cli/runner.py`, `pdfsys-cli/_vlm_config.py`, `pdfsys-cli/parquet_writer.py`, all YAML configs.

## 15 · Definition of done

- `pdfsys run -c pdfsys.full.yaml --out-dir out/e2e_full_mineru3_regional` completes with `errors ≤ 2`.
- All 5 §13 acceptance queries pass.
- `pdfsys visualize -r out/e2e_full_mineru3_regional -o out/viz_regional` produces a working bundle.
- VLM detail view shows the new Per-region extraction card with non-empty `content_excerpt` for at least one TABLE or FORMULA row.
- Routing trace's Stage-B step mentions "region-based" so future readers know this isn't whole-page VLM.
- §16 post-build note appended with actual stats.

## 16 · Post-build note (to be filled in)

Reserved.
