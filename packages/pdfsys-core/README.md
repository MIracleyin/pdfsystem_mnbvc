# pdfsys-core

Shared data contracts for the pdfsys pipeline. Every other package depends on this one.

## What's in here

- **Enums**: `RegionType` (TEXT / IMAGE / TABLE / FORMULA), `Backend` (MUPDF / PIPELINE / VLM / DEFERRED).
- **PdfRecord**: Frozen dataclass for per-PDF metadata (sha256, source_uri, size, provenance).
- **Layout schema**: `BBox` (normalized [0,1]), `LayoutRegion`, `LayoutPage`, `LayoutDocument` — the contract between layout-analyser and every parser backend.
- **ExtractedDoc / Segment**: Backend-agnostic output schema. All three parser backends emit these.
- **LayoutCache**: Content-addressable on-disk cache for LayoutDocuments, keyed by `sha256 + model_tag`.
- **PdfsysConfig**: Hierarchical configuration (paths, router, layout, per-backend settings, runtime).
- **Serde**: Generic `to_dict()` / `from_dict()` for all the above dataclasses.

## Key design rule

This package has **zero external dependencies** — stdlib only. Do not add pymupdf, torch, or anything else here. The types must be importable everywhere without pulling in heavy ML libraries.
