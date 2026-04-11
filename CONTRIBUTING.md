# Contributing to pdfsys-mnbvc

## Dev environment setup

```bash
# Prerequisites: Python >= 3.11, uv >= 0.4
uv sync                             # installs all workspace packages in editable mode
python -m pdfsys_router.download_weights   # one-time: fetch XGBoost weights (257 KB)
```

If you'll be working on quality scoring, torch + transformers are pulled in by `pdfsys-bench`. The ModernBERT-large model (~800 MB) downloads on first scorer use. Set `HF_HOME` to control the cache location.

## Project structure

```
pdfsystem_mnbvc/
├── pyproject.toml              # uv workspace root (meta-package)
├── packages/
│   ├── pdfsys-core/            # shared types, enums, layout cache, serde
│   ├── pdfsys-router/          # Stage-A XGBoost classifier
│   │   ├── models/             # gitignored xgb_classifier.ubj lives here
│   │   └── src/pdfsys_router/
│   │       ├── feature_extractor.py   # 124-feature PyMuPDF extractor
│   │       ├── xgb_model.py           # lazy model loader
│   │       ├── classifier.py          # Router.classify() → RouterDecision
│   │       └── download_weights.py    # fetch weights from HF LFS
│   ├── pdfsys-parser-mupdf/    # text-ok fast path (PyMuPDF blocks → Markdown)
│   ├── pdfsys-parser-pipeline/ # OCR backend (stub)
│   ├── pdfsys-parser-vlm/      # VLM backend (stub)
│   ├── pdfsys-layout-analyser/ # layout model runner (stub)
│   └── pdfsys-bench/           # evaluation harness + quality scorer
│       ├── omnidocbench_100/   # gitignored bench dataset
│       └── src/pdfsys_bench/
│           ├── quality.py      # ModernBERT-large OCR quality scorer
│           ├── loop.py         # router → parser → scorer → JSONL runner
│           └── __main__.py     # CLI entry point
└── out/                        # gitignored run outputs
```

## Code conventions

### Naming

- Package dirs: `pdfsys-<name>` (kebab-case in pyproject.toml and directory names).
- Import names: `pdfsys_<name>` (snake_case, matching `src/pdfsys_<name>/`).
- All packages live under `packages/` and use the `[tool.uv.workspace]` editable pattern.

### Types and immutability

- Core data structures are `@dataclass(frozen=True, slots=True)`.
- Enums live in `pdfsys_core.types`.
- BBox coordinates are always normalized to `[0, 1]`; convert to pixels/points at the call site.
- Parser backends all emit `ExtractedDoc` with a `tuple[Segment, ...]` — the schema is backend-agnostic.

### Error handling

- `Router.classify()` never raises. Errors are surfaced via `RouterDecision.error`.
- Parser `extract_doc()` may raise; the bench loop catches and records errors in JSONL.
- Prefer explicit `except Exception` with a recorded message over silent swallowing.

### Feature extractor parity

The `feature_extractor.py` in `pdfsys-router` is a direct port of FinePDFs'
`blocks/predictor/ocr_predictor.py`. The 124-column feature vector MUST match
the upstream layout exactly — the XGBoost weights depend on column order. If you
change any feature extraction logic, verify against the FinePDFs reference output
before merging.

The feature ordering is:
1. `num_pages_successfully_sampled` (doc-level)
2. `garbled_text_ratio` (doc-level)
3. `is_form` (doc-level)
4. `creator_or_producer_is_known_scanner` (doc-level)
5. `page_level_unique_font_counts_page1` through `_page8`
6. ... (15 page-level features × 8 pages = 120 columns)

Total: 4 + 120 = 124 features.

### Dependencies

- `pdfsys-core` has **zero** external dependencies. Keep it that way.
- Heavy deps (torch, transformers) are lazy-imported so that `import pdfsys_bench` doesn't pull them in at module scope.
- XGBoost model weights are NOT committed to the repo. They're downloaded on demand via `download_weights.py`.

## Running the MVP

```bash
# Full run on OmniDocBench-100 (takes ~4 min on CPU)
python -m pdfsys_bench \
  --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
  --out out/bench_omnidoc100.jsonl \
  --markdown-dir out/bench_omnidoc100_md

# Fast smoke test (no quality scoring)
python -m pdfsys_bench \
  --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
  --out out/smoke.jsonl \
  --limit 5 --no-quality
```

Output: one JSONL file (per-doc results) + one `.summary.json` (aggregate stats).

## Adding a new parser backend

1. Implement the backend in its package under `packages/pdfsys-parser-<name>/`.
2. The entry point should accept a `Path` and return `ExtractedDoc` (from `pdfsys-core`).
3. Each `Segment` must have `page_index`, `type` (RegionType), `content`, and ideally a normalized `BBox`.
4. Call `merge_segments_to_markdown(segments)` from `pdfsys-core` to produce the `markdown` field.
5. Wire it into `loop.py` by handling the corresponding `Backend` enum value.

## Adding new features to the router

**Do not** modify `feature_extractor.py` unless you're also retraining the XGBoost model. The weights and feature layout are coupled. If you need additional routing signals, add them as post-classification heuristics in `classifier.py` rather than changing the feature vector.

## Commit conventions

Commit messages follow conventional commits:

```
feat(router): add scanner metadata detection
fix(parser-mupdf): handle zero-width bbox on empty pages
docs: update quickstart for new deps
chore: bump pymupdf to 1.25
```

Scope is the package name without the `pdfsys-` prefix (e.g. `router`, `core`, `bench`, `parser-mupdf`).
