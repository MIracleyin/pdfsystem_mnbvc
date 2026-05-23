# Mineru Parsers Migration — In-process (Spec #1)

**Date:** 2026-05-22
**Status:** Design — awaiting plan
**Predecessor:** `fd8b8d8 feat(parser-vlm): rewrite extract.py for region-based parsing via MinerUClient`
**Successor (out of scope):** Spec #2 — Parsers as HTTP services + Docker

## 1. Context

The current parser layer is heterogeneous:

| Parser | Today | LOC | Models |
|--------|-------|-----|--------|
| `pdfsys-parser-mupdf` | PyMuPDF text extraction (born-digital) | small | — |
| `pdfsys-parser-pipeline` | RapidOCR (rapidocr_onnxruntime) on layout-cropped regions | 360 | RapidOCR ONNX |
| `pdfsys-parser-vlm` | mineru 3.x `ModelSingleton.batch_content_extract` on layout-cropped regions | 292 | mineru VLM |

Layout-analyser (`doclayout-yolo`) runs once per PDF and emits a `LayoutDocument` to the layout cache. Stage-A router decides text-ok vs. needs-OCR. Stage-B router (in the loop, fed by layout) decides PIPELINE vs. VLM for OCR-needing PDFs. Both ocr-needing parsers consume per-region crops from the layout cache.

This design has accumulated problems:

1. **Environment fragility.** `rapidocr-onnxruntime` and `mineru` both transitively pull `opencv-python` variants. The session repeatedly hit `module 'cv2' has no attribute 'INTER_NEAREST'` because `opencv-python` and `opencv-python-headless` cannot coexist in one venv.
2. **Two divergent code paths.** Region-based VLM (custom `merge_segments_to_markdown`) is a parallel re-implementation of what mineru's `do_parse` already does end-to-end and battle-tests centrally.
3. **Mineru pipeline mode never tried.** When Stage-B decides `vlm` and `vlm_enabled=False`, the code silently falls back to the pipeline parser (RapidOCR) — but mineru also ships a *pipeline mode* (its own classical OCR pipeline) that's never been wired in. Running a bench today confirms 0/150 rows actually hit mineru, even with `--full-pipeline`.
4. **No symmetry.** Pipeline and VLM parsers have completely different internals despite occupying the same architectural slot. Maintenance grows linearly.

Mineru 3.x exposes a single entry point — `mineru.cli.common.do_parse(...)` — that takes `backend="pipeline" | "vlm-transformers" | "vlm-mlx-engine" | "vlm-vllm-engine"`. Both pipeline mode and VLM mode are end-to-end (mineru does its own layout + region extraction + markdown assembly internally), so a thin wrapper per parser is all we need.

This spec ships **Spec #1: in-process migration to mineru for both ocr-needing parsers.** Both parsers become symmetric `do_parse`-wrappers, RapidOCR and the region-based VLM custom merge are deleted. Spec #2 (separate) will then split each parser into its own HTTP service + Dockerfile.

## 2. Goals

1. `pdfsys-parser-pipeline.PipelineParser` calls `mineru.cli.common.do_parse(backend="pipeline", ...)` end-to-end.
2. `pdfsys-parser-vlm.VlmParser` calls `mineru.cli.common.do_parse(backend="vlm-transformers", ...)` end-to-end. Engine selectable via CLI flag (transformers / mlx-engine / vllm-engine).
3. Both parsers expose the same interface: `parser.extract(pdf_path: Path) -> ExtractedDoc`. No layout / regions input.
4. Mineru sidecar artifacts (`<name>_middle.json` + `<name>_content_list.json`) are kept; everything else (`_layout.pdf`, `_model.json`, `_span.pdf`, `_origin.pdf`) is disabled.
5. Bench runs `--cascade --vlm` actually hit mineru pipeline + mineru VLM (not the silent fall-back observed today).
6. Delete `pdfsys-parser-pipeline/ocr_engine.py`, `pdfsys-parser-pipeline/extract.py` (replace), and the region-based code in `pdfsys-parser-vlm/extract.py` (replace). Remove `rapidocr-onnxruntime` from `pdfsys-parser-pipeline/pyproject.toml`.

## 3. Non-Goals (explicitly Spec #2 work)

- **HTTP service wrappers** — each parser stays in-process for v1. Spec #2 splits them into HTTP services + Dockerfiles.
- **Per-parser venv isolation** — both parsers continue to live in the workspace venv. Spec #2 introduces per-parser pyproject + lockfile.
- **Page-subset routing** — Stage-B routes the whole PDF to one backend. Mixed routing ("pages 1–3 via mupdf, pages 4–5 via VLM") was already a non-goal in the prior design and stays one.
- **Backwards compatibility** — no fallback to RapidOCR, no `--ocr-engine rapidocr` switch, no preserving the region-based VLM `merge_segments_to_markdown` path. Delete and replace.
- **viz "Per-region extraction" card revival** — that card relied on per-region segment data. Mineru's middle.json has the data, but rendering it back into the viz is out of scope for #1.

## 4. Architecture (after this spec)

```
bench loop (existing flags --cascade / --full-pipeline / --vlm preserved)
  │
  ├─ Stage-A router (XGBoost on doc features)
  │     → MUPDF (text-ok)  or  needs-OCR
  │
  ├─ layout-analyser (doclayout-yolo) — runs once on needs-OCR
  │     → emits LayoutDocument (cached) — used by Stage-B routing only
  │
  ├─ Stage-B decider — reads LayoutDocument complexity → PIPELINE | VLM
  │
  └─ Parser dispatch (one of, never a mix):
       MupdfParser.extract(pdf_path)                        → ExtractedDoc
       PipelineParser.extract(pdf_path)                     → ExtractedDoc
           └─ mineru.cli.common.do_parse(backend="pipeline", ...)
       VlmParser.extract(pdf_path, engine="vlm-transformers")  → ExtractedDoc
           └─ mineru.cli.common.do_parse(backend="vlm-<engine>", ...)
```

**Key invariant:** parsers no longer accept layout / region inputs. `layout-analyser` produces `LayoutDocument` for Stage-B's routing decision; parsers then independently do their own internal layout + extraction via mineru. The `LayoutDocument` is still serialized to the layout cache (existing behavior) for trace / debug / future use; parsers simply don't consume it.

## 5. Parser interface

Single method, identical signature on both ocr-needing parsers:

```python
class PipelineParser:
    def __init__(self, config: PipelineConfig | None = None) -> None: ...
    def extract(self, pdf_path: Path) -> ExtractedDoc: ...

class VlmParser:
    def __init__(self, config: VlmConfig | None = None) -> None: ...
    def extract(self, pdf_path: Path) -> ExtractedDoc: ...
```

`ExtractedDoc` (existing in `pdfsys-core`) carries:
- `sha256`, `backend` (Backend.PIPELINE / Backend.VLM), `markdown`, `segments` (empty for v1 — mineru produces structured data in sidecars), `stats` (dict).

`stats` gains a few keys for traceability:
- `mineru_backend`: the literal mineru backend string used (e.g. `"pipeline"`, `"vlm-transformers"`).
- `mineru_version`: from `mineru.__version__` if present.
- `middle_json_path`: relative path to the `<sha>_middle.json` sidecar (under `markdown-dir/<sha>/`), or `None` if `output_dir is None`.
- `content_list_path`: relative path to `<sha>_content_list.json`, or `None` if `output_dir is None`.

`sha256` is computed by the parser from `pdf_path.read_bytes()` (not accepted as a kwarg) — keeps the `extract(pdf_path)` signature minimal and prevents callers from passing inconsistent values.

**Output directory semantics** (used by both `PipelineConfig` and `VlmConfig`):

- `config.output_dir is None` (default): parser uses a temporary directory; only `markdown` is returned in `ExtractedDoc`, sidecar paths are `None`, and mineru's intermediate files are deleted on return.
- `config.output_dir = <Path>`: parser writes mineru's outputs to `<output_dir>/<sha>/<parse_method>/`. Sidecar paths in `stats` are relative to `config.output_dir` so they survive being moved across machines.

The bench loop wires `config.output_dir = args.markdown_dir` when `--markdown-dir` is set, mirroring the existing convention that `--markdown-dir` controls per-PDF artifact retention.

## 6. Mineru entry-point usage

Both parsers call:

```python
from mineru.cli.common import do_parse

do_parse(
    output_dir=<tmpdir or markdown-dir/<sha>/>,
    pdf_file_names=[<sha or doc stem>],
    pdf_bytes_list=[<PDF bytes>],
    p_lang_list=["ch"],  # mineru convention; PDF can be any language
    backend=<"pipeline" | "vlm-transformers" | ...>,
    parse_method="auto",
    formula_enable=True,
    table_enable=True,
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_content_list=True,
    f_dump_model_output=False,
    f_dump_orig_pdf=False,
    f_draw_layout_bbox=False,
    f_draw_span_bbox=False,
    image_analysis=True,
)
```

After `do_parse` returns, the parser:
1. Locates the markdown via `mineru.cli.common.prepare_env(output_dir, sha, parse_method)` (the same helper mineru itself uses) → returns the canonical `*_md_dir`. The markdown file is `<md_dir>/<sha>.md`. Falls back to `glob("*.md")` under the per-doc tree if the canonical layout shifts in a future mineru version.
2. Reads markdown file → `ExtractedDoc.markdown`.
3. Records `middle.json` and `content_list.json` paths in `stats` (relative to `config.output_dir`).
4. Acceptance criterion: if `config.output_dir is None`, the tmpdir is unconditionally removed before `extract()` returns; if set, mineru's intermediate-only artifacts (`_layout.pdf`, `_model.json`, `_origin.pdf`, `_span.pdf`) are removed but markdown + middle.json + content_list.json are kept.

## 7. Configuration

`pdfsys-core/config.py` already has `PipelineConfig` and `VlmConfig`. After this spec:

```python
@dataclass(slots=True)
class PipelineConfig:
    # No tunables for v1 — mineru pipeline mode has fixed sub-pipeline.
    formula_enable: bool = True
    table_enable: bool = True
    p_lang: str = "ch"
    output_dir: Path | None = None   # default: <markdown-dir>/<sha>/

@dataclass(slots=True)
class VlmConfig:
    # mineru VLM has multiple inference engines.
    engine: Literal["vlm-transformers", "vlm-mlx-engine", "vlm-vllm-engine"] = "vlm-transformers"
    formula_enable: bool = True
    table_enable: bool = True
    p_lang: str = "ch"
    output_dir: Path | None = None
```

Bench CLI gains one new flag:

```
--vlm-engine {transformers,mlx-engine,vllm-engine}   default: transformers
```

The flag value is prefixed with `vlm-` before passing to mineru.

## 8. Bench loop integration

`pdfsys-bench/loop.py` changes:

1. `PipelineParser()` and `VlmParser()` constructors take optional config (existing signature). No interface change at the call site.
2. Replace `vlm_parser.extract_complex_pages(pdf_path, layout, sha256=...)` (line 327, 437) with `vlm_parser.extract(pdf_path)`. Same for `pipeline_parser.extract_complex_pages(...)` if it exists.
3. The Stage-B routing decision and `LayoutDocument` flow are unchanged.
4. `LoopResult.layout_*` fields stay (they reflect what the layout-analyser decided, not what the parser saw). They remain useful for Stage-B trace and viz routing-trace card.

## 9. Removed code

| Path | Reason |
|------|--------|
| `pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/ocr_engine.py` | RapidOCR + PaddleOCR-classic abstraction — mineru pipeline mode replaces it end-to-end |
| Bulk of `pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/extract.py` | Region-by-region OCR loop — superseded by `do_parse(backend="pipeline")` |
| `pdfsys-parser-pipeline/pyproject.toml` dep `rapidocr-onnxruntime>=1.3` | Dead |
| Bulk of `pdfsys-parser-vlm/src/pdfsys_parser_vlm/extract.py` (region loop, `_run_vlm_per_region`, mineru `ModelSingleton` direct use, `merge_segments_to_markdown` callsite) | Superseded by `do_parse(backend="vlm-...")` |
| Tests targeting the region-based VLM path (`tests/parsers/test_vlm_*.py` if any) | Replaced by new tests on the `extract()` interface |

The packages' `__init__.py` files keep exporting the same parser class names so import sites don't churn.

## 10. Dependency cleanup

`pdfsys-parser-pipeline/pyproject.toml`:

```toml
# before
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "rapidocr-onnxruntime>=1.3",
    "numpy>=1.24",
    "Pillow>=10.0",
]

# after
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "mineru[pipeline]>=3.1,<4.0",
    "numpy>=1.24",
    "Pillow>=10.0",
]
```

`pdfsys-parser-vlm/pyproject.toml`:

```toml
# before
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "mineru>=3.1,<4.0",
    "accelerate>=1.0",
    "numpy>=1.24",
    "Pillow>=10.0",
]

# after
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.25",
    "mineru[vlm]>=3.1,<4.0",      # transformers, accelerate, torch all come via this extra
    "numpy>=1.24",
    "Pillow>=10.0",
]
```

`opencv-python` (NOT headless) is required by mineru. The workspace venv currently has `opencv-python-headless` reinstalled automatically by some transitive dep — a separate hygiene issue to investigate, but mineru itself imports fine after the venv self-heals. The plan's first task should verify the install is clean before any parser code runs.

## 11. Testing strategy

For both parsers, two test tiers:

**Tier A — unit tests with mocked `do_parse`** (no model load, fast):
- Mock `mineru.cli.common.do_parse` to write a known `.md` + `_middle.json` + `_content_list.json` to the output dir.
- Assert `parser.extract(p)` returns `ExtractedDoc` with the markdown content, correct `sha256`, correct `backend`, sidecar paths in `stats`.
- Edge cases: empty markdown file, missing sidecar (mineru sometimes skips middle.json if no content found), `do_parse` raises an exception.

**Tier B — integration smoke** (real mineru, gated by `MINERU_INTEGRATION=1` env):
- One real PDF from `packages/pdfsys-bench/omnidocbench_100/pdfs/`.
- For pipeline parser: `backend="pipeline"`, verify markdown is non-empty and contains expected words.
- For VLM parser: `backend="vlm-transformers"`, gated additionally on `--vlm-engine` availability (skip on CI without weights). Mark `@pytest.mark.slow`.

Tier A tests live in `tests/parsers/test_pipeline_parser.py` and `tests/parsers/test_vlm_parser.py`. Tier B tests live in `tests/parsers/integration/` and require explicit opt-in.

## 12. Risks & open questions

- **Model download on first run.** Mineru downloads weights to its cache (`~/.cache/modelscope` or `~/.cache/huggingface`) on first `do_parse(backend="vlm-transformers")`. ~7GB for VLM. Tests must NOT trigger download in CI — mocking is non-negotiable for Tier A.
- **`do_parse` output path layout.** Mineru's `prepare_env(output_dir, pdf_name, parse_method)` controls the actual subpath the markdown lands at. The plan's first task is a spike to nail this down (likely `<output_dir>/<pdf_name>/<parse_method>/<pdf_name>.md`). If the layout changes between mineru patch versions, the parser's read step needs to be defensive (glob for `*.md`).
- **`p_lang` value.** Mineru's pipeline OCR uses this for language hinting. Most OmniDocBench PDFs are Chinese; default to `"ch"`. Future per-PDF language detection is a separate concern.
- **Memory.** `vlm-transformers` loads a ~7GB model into RAM/VRAM. On a 16GB Mac this is borderline. `vlm-mlx-engine` is more memory-efficient on Apple Silicon. The plan should include a smoke test that documents observed memory on the dev machine before claiming "works".
- **Stage-B "vlm" decisions silently falling back to pipeline.** That's a separate pre-existing bug in loop.py and is NOT in scope for this spec. After this spec ships, if `vlm_enabled=False` and Stage-B says `vlm`, the behavior remains "fall through to pipeline" — but the pipeline path now itself is mineru, so the silent degrade is at least to a proper OCR pipeline, not RapidOCR. Fix is tracked as a follow-up.
- **opencv-python-headless reinstall.** Some transitive dep (probably one of the workspace packages or its lockfile) keeps pulling headless back. The plan must `grep -r opencv-python-headless` across all workspace `pyproject.toml`s and pin to `opencv-python` only.

## 13. Acceptance criteria

- [ ] `mineru.cli.common.do_parse` imports cleanly post-venv-sync (no cv2 errors). `from pdfsys_parser_pipeline import PipelineParser` and `from pdfsys_parser_vlm import VlmParser` both succeed.
- [ ] `PipelineParser().extract(<pdf>)` returns an `ExtractedDoc` with non-empty markdown when given a real PDF (Tier B smoke).
- [ ] `VlmParser(VlmConfig(engine="vlm-transformers")).extract(<pdf>)` returns an `ExtractedDoc` with non-empty markdown when weights are cached locally (Tier B smoke).
- [ ] `bench --cascade --vlm` on a 5-row slice of OmniDocBench produces at least one row with `backend == "vlm"` AND a non-empty markdown. (Demonstrates the silent-fallback bug observed earlier is gone for the proper case.)
- [ ] `rg "rapidocr"` and `rg "RapidOCR"` across the repo return no source-code hits (docs / specs / plans are OK).
- [ ] `rg "merge_segments_to_markdown" packages/pdfsys-parser-vlm` returns no hits.
- [ ] All Tier A unit tests pass (`uv run pytest tests/parsers/ -v`).
- [ ] Existing release-gate test suite still passes (`uv run pytest tests/bench/ tests/architecture/ -v`).
- [ ] `uv run ruff check packages/` clean.

## 14. Out of scope (re-stated for clarity)

- **Spec #2: HTTP services + Docker** — separate spec; this spec leaves parsers in-process and shipping in the workspace venv.
- **Layer 3 (visual verifier / consensus)** — separate.
- **Per-PDF language detection** — `p_lang="ch"` is the v1 default.
- **vLLM / lmdeploy / hybrid backends** — flag accepts `transformers` and `mlx-engine`; vllm/lmdeploy/hybrid are theoretically supported by mineru but require platform-specific deps not in this venv.
- **Mineru CLI passthrough (e.g. `mineru -i ... -o ...`)** — we call `do_parse` directly; the `mineru` shell command is not a dependency.

## 15. Post-build note (2026-05-22 / 2026-05-23)

Implementation landed across 9 planned tasks + 3 unplanned fix iterations
when in-process integration hit unanticipated macOS issues. Plan:
`docs/superpowers/plans/2026-05-22-mineru-parsers-migration.md`.

### Commits (in chronological order)

```
─── Planned 9 tasks ───
d3649a5  Task 1  test(parsers): lock mineru + cv2 import surface
b27d1d5  Task 1  fix(parsers): deepen import guards + pin opencv-python
0970640  Task 2  refactor(core): PipelineConfig + VlmConfig → mineru fields
823f262  Task 2  fix(core): serde Path support + tighter config tests
19e8e1d  Task 3  deps(parsers): swap rapidocr → mineru[pipeline]; mineru[vlm]
ed9f70a  Task 4  refactor(parser-pipeline): mineru pipeline-mode wrapper
c3cd758  Task 4  fix(parser-pipeline): glob fallback + version sync + sidecar test
2bce47a  Task 5  refactor(parser-vlm): mineru VLM-mode wrapper
4dcbdcc  Task 6  feat(bench): wire mineru parsers + --vlm-engine flag
bd5af21  Task 7  refactor(cli): align YAML schema + runner with mineru
af5c4a3  Task 8  test(parsers): Tier B integration tests (gated)

─── Architectural pivot: in-process → out-of-process ───
ffefec6  hack    feat(bench): --cascade-skip-pipeline flag for macOS workaround
16466f8  hack    fix(parsers): swap do_parse → aio_do_parse (didn't fully fix)
31ce715  hack    fix(parsers): ThreadPoolExecutor workaround + VLM subdir
5bf81fa  hack    fix(parsers): mp.set_start_method('fork') (still hung)
3847b6f  PIVOT   refactor(parsers): out-of-process HTTP client via mineru-api subprocess
a4fed31  fix     fix(parsers): HF_HUB_OFFLINE=1 + add mineru[mlx] for arm64-darwin
```

### Why the architectural pivot

The plan assumed in-process `mineru.cli.common.do_parse(...)` would work. On
macOS Apple Silicon (M4 Pro, 48GB) it didn't. Sequential debug:

1. **`do_parse` (sync) deadlocks.** Mineru's PDF render uses a
   `ProcessPoolExecutor` with `mp_context=spawn`. Spawn-mode workers must
   re-import the parent process; bench's heavy import surface (torch +
   MLX + transformers + parsers) makes them deadlock during re-import.
2. **`aio_do_parse` (async) also deadlocks** — same `ProcessPoolExecutor`
   under the hood. Async only changes how the parent waits, not how
   workers spawn.
3. **`ThreadPoolExecutor` monkey-patch** of mineru's PDF render fixed
   that one executor but **DocAnalysis init has its own mp.Pool** that
   also deadlocks. Each component in mineru could spawn its own pool.
4. **`mp.set_start_method('fork', force=True)`** unblocked DocAnalysis
   init and pipeline mode actually ran (25s/page in isolation). But VLM
   still hung at `Predict: 0%` — torch+MPS context from pipeline
   conflicts with MLX-VLM trying to allocate Metal afterwards in the
   same process.
5. **Out-of-process via mineru-api subprocess + HTTP.** Each parser
   spawns its own `mineru-api` subprocess on first `extract()`. Bench
   never imports mineru, so its heavy import surface cannot collide
   with mineru's machinery. mineru's PDF render mp.Pool now spawns
   from a clean `mineru-api` Python process — works correctly.

### End-to-end smoke (verified on M4 Pro / macOS)

```
uv run python -m pdfsys_bench \
    --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
    --out out/bench_mineru_smoke.jsonl \
    --cascade --vlm --vlm-engine mlx-engine \
    --limit 5 --no-quality
```

Result:
```
ROW 1  be=mupdf      cascade=['mupdf']             wall=16ms    md=399 chars
ROW 2  be=mupdf      cascade=['mupdf']             wall=10ms    md=441 chars
ROW 3  be=pipeline   cascade=['mupdf','pipeline']  wall=11.1s   md=2498 chars  MINERU=pipeline
ROW 4  be=mupdf      cascade=['mupdf']             wall=9ms     md=559 chars
ROW 5  be=mupdf      cascade=['mupdf']             wall=5ms     md=5734 chars
```

Standalone parser smokes (proving all three backends usable):

```
mupdf      (in-process)                : ~10ms/page,  399-5734 chars
pipeline   (mineru-api subprocess HTTP) : 27.9s/page, 2498 chars
vlm        (mineru-api + mlx-engine)    : 20.4s/page, 1773 chars
```

### Architectural changes from the original spec

The original spec assumed in-process calls. After the pivot:

1. **§5 Parser interface unchanged** — still `parser.extract(pdf_path) -> ExtractedDoc`.
   Bench code DID NOT change.
2. **§5 ExtractedDoc.stats: new key** — `mineru_api_url` records the
   subprocess URL. `middle_json_path` / `content_list_path` still recorded.
3. **§5 ExtractedDoc.stats: removed key** — `mineru_version` is now
   `payload.get("version")` from the HTTP response (server's mineru
   version), still semantically correct.
4. **Parser internals 100% rewritten** — no more `mineru.cli.common`
   import. Just `httpx` + `subprocess.Popen` for the `mineru-api` binary.
5. **`_macos_workaround.py` deleted** — fork start-method no longer
   needed because bench never imports mineru.
6. **New runtime dep**: `mineru-api` binary must be on PATH or next to
   `sys.executable`. Comes for free via `mineru[vlm]` / `mineru[pipeline]`
   extras.
7. **New runtime dep**: `mineru[mlx]>=3.1,<4.0` declared with
   `sys_platform == 'darwin' and platform_machine == 'arm64'` marker so
   Apple Silicon users get MLX automatically without breaking
   Linux/Intel installs.
8. **Subprocess env**: `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` set
   when spawning mineru-api — mineru otherwise hits HF Hub for revision
   checks at every cold start, which fails on flaky HF connectivity.
   Cached weights under `~/.cache/huggingface/hub/` are used directly.

### Known follow-ups

- **subprocess lifetime**: each parser keeps its mineru-api subprocess
  alive across all `extract()` calls and terminates on GC. For long-
  running benches this is correct. Short-lived scripts can leak the
  subprocess if `__del__` doesn't fire — explicit `parser.close()` is
  available and recommended.
- **Port collision race**: `_pick_free_port` binds to port 0 to find a
  free port, then closes. Theoretical race if another process grabs it
  before mineru-api binds. For single-user dev this is fine.
- **Pipeline OCR quality**: pipeline mode replaces periods with `?`
  characters in some numeric contexts (`14.70` → `14?? 70`). This is
  mineru pipeline mode's OCR engine — not our problem to fix; VLM mode
  produces clean output.
- **Spec #2** (Docker containers + service orchestration) is the
  natural next step. The HTTP-client architecture this pivot landed is
  essentially Spec #2's core; Dockerfile packaging is the remaining
  delta. The parser interface won't change.

### Test surface

```
tests/parsers/test_mineru_smoke.py       3 tests   (cv2 + mineru imports)
tests/parsers/test_pipeline_parser.py    9 tests   (Tier A, mocked httpx)
tests/parsers/test_vlm_parser.py        10 tests   (Tier A, mocked httpx)
tests/parsers/integration/               2 tests   (Tier B, gated, real HTTP)
tests/core/test_parser_configs.py        6 tests   (config dataclasses + serde)
─────────────────────────────────────────────────────────────────────────
Total                                   30 new tests; 94 passed + 2 skipped
                                        across the full suite.
```

Run: `uv run pytest tests/parsers/ tests/core/ -v`.

