# pdfsystem-mnbvc — Agent Orientation Map

> PB-scale PDF → pretraining-data pipeline for the MNBVC Chinese corpus. FinePDFs-inspired, dual-path (CPU text + GPU OCR/VLM).

## Stack

| Layer          | Tech                        |
|----------------|-----------------------------|
| Language       | Python 3.11+                |
| Package mgr   | uv (workspace monorepo)     |
| Build          | hatchling                   |
| PDF engine     | PyMuPDF 1.27+               |
| ML: router     | XGBoost (FinePDFs port)     |
| ML: layout     | DocLayout-YOLO (ONNX)       |
| ML: OCR        | RapidOCR (ONNX)             |
| ML: VLM        | MinerU 2.5 Pro (magic-pdf)  |
| ML: quality    | ModernBERT-large (HF)       |

## Architecture Layers

Dependency flows **downward only**. Never import upward.

```
pdfsys-core            ← zero external deps, stdlib only
    ↑
pdfsys-router          ← pymupdf, xgboost, numpy
pdfsys-layout-analyser ← doclayout-yolo, pymupdf
pdfsys-parser-mupdf    ← pymupdf
pdfsys-parser-pipeline ← rapidocr, pymupdf
pdfsys-parser-vlm      ← magic-pdf, torch
    ↑
pdfsys-bench           ← torch, transformers (quality scorer)
pdfsys-cli             ← pyyaml (orchestration layer)
```

## Key Conventions

- `pdfsys-core` MUST have zero external dependencies — see `docs/golden-principles/ZERO_DEP_CORE.md`
- All parser backends emit identical `ExtractedDoc` / `Segment` schema — see `docs/golden-principles/UNIFORM_OUTPUT.md`
- BBox coordinates always normalized to `[0, 1]` — never raw pixels or points
- Frozen dataclasses (`@dataclass(frozen=True, slots=True)`) for all data contracts
- Heavy deps (torch, transformers, magic-pdf) imported lazily

## Commands

```sh
uv sync                                    # install all workspace packages
pdfsys init-config > pdfsys.yaml           # generate example config
pdfsys run -c pdfsys.yaml --stages router  # run specific stage
pdfsys run -c pdfsys.yaml                  # run full pipeline
python -m pdfsys_bench --pdf-dir ... --out ... # legacy bench CLI
uv run ruff check .                        # lint
uv run pytest tests/                       # test
```

## Documentation Map

```
ARCHITECTURE.md                       Top-level domain map
docs/
├── PRD.md                            Product requirements (Chinese, 440 lines)
├── architecture/
│   └── LAYERS.md                     Layer rules + dependency enforcement
├── golden-principles/                Canonical patterns (DO/DON'T)
│   ├── ZERO_DEP_CORE.md
│   ├── UNIFORM_OUTPUT.md
│   └── LAZY_IMPORTS.md
├── SECURITY.md                       Secrets & data handling
└── guides/                           Setup, adding backends
```

## Where to Look First

| Task                         | Start here                             |
|------------------------------|----------------------------------------|
| Architecture overview        | `ARCHITECTURE.md`                      |
| Layer rules                  | `docs/architecture/LAYERS.md`          |
| Add a new parser backend     | `CONTRIBUTING.md` § Adding Backends    |
| Data contracts (types/enums) | `packages/pdfsys-core/src/pdfsys_core/`|
| Router feature parity        | `packages/pdfsys-router/src/pdfsys_router/feature_extractor.py` |
| Run the pipeline             | `pdfsys run -c pdfsys.yaml`            |
| Product requirements         | `docs/PRD.md`                          |

## Constraints (Machine-Readable)

- MUST: `pdfsys-core` has zero external deps → enforced by `tests/architecture/test_boundary.py`
- MUST NOT: parser backends import each other → see `docs/architecture/LAYERS.md`
- MUST NOT: bench/cli import core internals not in `__init__.py`
- PREFER: lazy imports for torch/transformers/magic-pdf
- VERIFY: `uv run ruff check . && uv run pytest tests/` before PR
