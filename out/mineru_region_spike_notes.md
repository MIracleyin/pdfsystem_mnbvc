# mineru region-based spike notes — 2026-05-16

## Verified

`MinerUClient.batch_content_extract([crops], [types])` works end-to-end on
Mac/MPS with our DocLayout-YOLO-cropped regions. Six regions of one PDF
(`1_pg19.pdf`, scanned math) processed in 13.9s on a warm model (one
TEXT region was 0.2s; the four FORMULA regions averaged 2.3s). Predictor
init when not cached: 3-4s.

Sample successful output for a FORMULA region:
```
'Let \(\frac{x^2}{(x^2 + 1)(x^2 + 4)} = \frac{\mathbf{A}}{(x^2 + 1)} + ...'
```

Sample successful output for a TEXT region:
```
'(5.) \( du = \frac{(3x^2 + x - 2)dx}{(x - 1)^3(x^2 + 1)}. \)'
```

LaTeX math is delivered wrapped in `\(...\)` inline / `\[...\]` display
when the call uses `type='text'`. Using `type='equation'` yields
`$$...$$` block style instead (the brainstorm Q1.A semantics). Both are
readable in viz's KaTeX renderer; we'll use the proper `type` per
region to stay consistent.

## API surface — verified call shape

### 1. Construct the client via mineru's `ModelSingleton`, NOT `MinerUClient(...)` directly

`MinerUClient(backend='transformers')` raises:
```
ValueError: model_path must be provided when model or processor is None.
```

The correct pattern is what mineru's own `vlm_doc_analyze` does:

```python
from mineru.backend.vlm.vlm_analyze import ModelSingleton

client = ModelSingleton().get_model(
    backend="transformers",
    model_path=None,      # ModelSingleton auto-downloads via auto_download_and_get_model_root_path
    server_url=None,
)
```

`ModelSingleton` is process-wide cached, so this is also the right shape
for our `VlmParser._ensure_client()` — `do_parse` and our direct path
will share the same loaded model with no double-load risk.

Init wall time when model is in HF cache: **3.2 s** (one-time per process).

### 2. `batch_content_extract` signature

```python
results: list[ExtractStr | None] = client.batch_content_extract(
    images: list[PIL.Image.Image],
    types: Sequence[str],   # one per image
)
```

- `ExtractStr` IS a subclass of `str` (verified via `inspect`). Use
  `str(result)` or treat as plain string. Do NOT access `.text` —
  that attribute does not exist.
- `None` is possible per-result (treat as "extraction failed" → empty
  content + bump `region_failures`).
- Single batch processed 6 images in 13.9 s on MPS warm.

### 3. Type strings

mineru accepts these `type` values (per `MinerUClient.prompts` dict from
2026-05-14 spike): `text | table | equation | image | chart`.

Map from `pdfsys_core.RegionType` (only 4 values in the enum):

| pdfsys `RegionType` | mineru `type` |
|---|---|
| `RegionType.TEXT` | `'text'` |
| `RegionType.IMAGE` | `'image'` |
| `RegionType.TABLE` | `'table'` |
| `RegionType.FORMULA` | `'equation'` |

Unknown values default to `'text'`. The earlier plan listing TITLE /
HEADER / FOOTER / REFERENCE / PAGE_NUMBER / FIGURE in the map is wrong —
those types do not exist in pdfsys-core's enum. The map shrinks to 4
entries.

## Layout cache file path is 2-level sharded

`pdfsys-core.LayoutCache` writes files under
`<cache_dir>/<sha[:2]>/<sha[2:4]>/<sha>.<model_tag>@<version>.json`.

Example for the spike PDF (sha `ce7b17...`):
```
out/e2e_full_mineru3/.cache/layout/ce/7b/ce7b17b543cd004773190457a038efbc41e0f98e178f0e1277bd793d16e3b6ee.doclayout-yolo-docstructbench@1.0.json
```

Implication for our `viz.py::_load_layout_block`: the existing
`glob(f"{sha}*")` (not `rglob`) was looking in the wrong directory level
— it found 0 files for every row, which is why VLM rows in the previous
viz showed empty layout blocks. **Should be `rglob(f"{sha}*")`** (or
`<cache_dir>/<sha[:2]>/<sha[2:4]>/<sha>*` directly).

This is independent of region-based parsing but worth fixing in viz.

## Layout JSON region types are lowercase strings (`'text'`, `'formula'`)

When reading the layout cache JSON directly (as the spike does), region
`type` is a string. When using `pdfsys_core.LayoutDocument.from_dict`,
those strings are auto-converted to `RegionType` enum.

`extract_complex_pages` in production receives an already-deserialized
`LayoutDocument`, so `region.type` is an enum. Inside extract.py we can
write the map keyed by enum directly (no string-coercion needed).

## Per-region wall-time profile

On a `1_pg19.pdf` page with 6 regions, MPS warm:

| Iter | Cumulative time | Per-iter |
|---|---|---|
| 1 | 2.34 s | 2.34 s (warm-up) |
| 2 | 2.65 s | 0.31 s |
| 3 | 5.23 s | 2.58 s |
| 4 | 7.92 s | 2.69 s |
| 5 | 11.00 s | 3.08 s |
| 6 | 13.85 s | 2.85 s |

Average ~2.3 s/region. For 10 VLM PDFs averaging ~14 regions = 140
regions = ~5.4 minutes inference. Plus ~30 s overhead per PDF (page
render + layout load) = ~9-10 min total VLM time. Combined with the
fixed ~3 min for the 140 non-VLM rows, **expected full run wall time:
12-15 minutes**.

## Region failure modes encountered

None on this PDF. All 6 regions returned non-empty strings. The previous
spec's worry about ExtractStr being None (per-region failure) was not
triggered; treat it as defensive only.

## Code template for Task 2's `_run_vlm_per_region`

```python
from mineru.backend.vlm.vlm_analyze import ModelSingleton

# ... inside VlmParser ...

def _ensure_client(self):
    if self._client is None:
        self._client = ModelSingleton().get_model(
            backend="transformers",
            model_path=None,
            server_url=None,
        )
    return self._client

# inside the per-page loop:
results = client.batch_content_extract(crops, types)
for region, result in zip(region_refs, results):
    text = str(result) if result is not None else ""
    ...
```

Region-type → mineru-type map (the only 4 cases needed):

```python
_REGIONTYPE_TO_MINERU = {
    RegionType.TEXT:    "text",
    RegionType.IMAGE:   "image",
    RegionType.TABLE:   "table",
    RegionType.FORMULA: "equation",
}
```

## Quirks

1. **gpu_memory: 1 GB on MPS** — mineru's batch-size detector reports
   GPU memory as 1 GB on Mac MPS (likely a stub since MPS doesn't
   expose VRAM the same way as CUDA). Default batch_size resolves to
   1, which limits parallelism to one image per forward pass. Not a
   correctness issue — region-level parallelism is happening at the
   batch_content_extract level — but means we can't speed it up by
   increasing batch_size on MPS.

2. **First-iter warmup is ~2× slower** than steady-state — caused by
   one-time CUDA/MPS kernel compilation. Not relevant for our use case
   (we re-use the same client across 10 PDFs).

3. **No `region_id` on layout regions** — `pdfsys_core.RegionType`
   regions don't appear to have a stable ID field readable as
   `region.region_id`. Our Segment.source_region_id should be None
   unless we discover otherwise during Task 2 implementation.
