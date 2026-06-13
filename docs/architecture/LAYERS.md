# Architecture Layers

Dependency flows **downward only**. A package may only import from packages in its "Allowed imports" column.

## Layer Matrix

| Layer | Package | Allowed Imports |
|-------|---------|-----------------|
| **Foundation** | `pdfsys-core` | stdlib only (zero external deps) |
| **Processing** | `pdfsys-router` | `pdfsys-core` |
| **Processing** | `pdfsys-layout-analyser` | `pdfsys-core` |
| **Processing** | `pdfsys-parser-mupdf` | `pdfsys-core` |
| **Processing** | `pdfsys-parser-pipeline` | `pdfsys-core` |
| **Processing** | `pdfsys-parser-vlm` | `pdfsys-core` |
| **Evaluation** | `pdfsys-bench` | `pdfsys-core`, `pdfsys-router`, `pdfsys-parser-mupdf` |
| **Orchestration** | `pdfsys-cli` | all packages |

## Dependency Diagram

```
                    pdfsys-core  (Foundation — zero deps)
                         ↑
        ┌────────────────┼────────────────────────────────┐
        │                │                                │
  pdfsys-router  pdfsys-parser-*  pdfsys-layout-analyser  │
        │                │                                │
        ↑                ↑                                ↑
        └───── pdfsys-bench ──────────────────────────────┘
                         ↑
                    pdfsys-cli  (Orchestration — top)
```

## Prohibited Imports

These imports are **never allowed** and enforced by `tests/architecture/test_boundary.py`:

| From | Cannot Import | Why |
|------|--------------|-----|
| `pdfsys-core` | any external package | Core must remain stdlib-only |
| `pdfsys-parser-mupdf` | `pdfsys-router`, `pdfsys-parser-pipeline`, `pdfsys-parser-vlm` | Parsers are siblings, not parent-child |
| `pdfsys-parser-pipeline` | `pdfsys-router`, `pdfsys-parser-mupdf`, `pdfsys-parser-vlm` | Same as above |
| `pdfsys-parser-vlm` | `pdfsys-router`, `pdfsys-parser-mupdf`, `pdfsys-parser-pipeline` | Same as above |
| `pdfsys-router` | `pdfsys-bench`, `pdfsys-cli`, any parser | Router only depends on core |
| `pdfsys-layout-analyser` | `pdfsys-bench`, `pdfsys-cli`, any parser, `pdfsys-router` | Layout only depends on core |

## Remediation Guide

When a violation is detected:

```
VIOLATION: packages/pdfsys-router/src/pdfsys_router/foo.py:5 imports pdfsys_parser_mupdf
  → pdfsys-router (Processing) cannot import pdfsys-parser-mupdf (Processing sibling).
  → Move the shared logic to pdfsys-core, or pass it as a parameter from the orchestration layer.
  → See: docs/architecture/LAYERS.md
```

**Common fixes:**
1. **Shared types/utils** → move to `pdfsys-core`
2. **Cross-package call** → inject via the orchestration layer (`pdfsys-cli` or `pdfsys-bench`)
3. **Test helper** → `conftest.py` in the test directory, not in src

## Component versioning

This project ships as a system release — a tuple of (main repo commit, pinned
component commits). Each independently-versioned component lives in its own
git repo (current or future), mounted as a submodule under `external/`. A
machine-readable manifest at `system_release.toml` (repo root) pins each
component to a 40-char commit SHA + human-readable tag.

### Why independent versions

See `docs/ocr-quality-strategy.md §13`: any component whose output is used as
an optimization target (the quality scorer first, but parsers as a second-order
case) must be versioned independently from the pipeline that consumes it.
Otherwise pipeline-level optimization will silently game the consumer's internal
signals (Goodhart's law).

### Components

| Component | Today | After Spec #1 / #2 | Pin source |
|---|---|---|---|
| `parsers` | `in-tree:packages` (collects 4 parser packages) | `external/parsers` (submodule) | `[components.parsers]` |
| `quality-scorer` | `in-tree:packages/pdfsys-bench` | `external/quality-scorer` (Spec #2, TBD) | `[components.quality-scorer]` |

The `parsers` component currently bundles four in-tree packages:
- `packages/pdfsys-parser-mupdf`
- `packages/pdfsys-parser-pipeline`
- `packages/pdfsys-parser-vlm`
- `packages/pdfsys-layout-analyser`

They will move to `external/parsers/packages/*` once Spec #1 completes (see
`docs/superpowers/specs/2026-05-30-parsers-submodule-design.md`).

The `quality-scorer` is currently in-tree at `packages/pdfsys-bench` (the
`OcrQualityScorer` / `QualityScore` classes plus `quality_*.py` modules); a
future Spec #2 will extract it into its own repo with sealed-holdout
governance.

### Schema version

The `parsers` component declares `schema_version = "extracted_doc.v1"` in
`system_release.toml`. The JSON schema for the wire contract lives at
`docs/schema/extracted_doc.v1.json` (frozen by Task 0.1); a self-contained
Python mirror at `docs/schema/_extracted_doc_v1_mirror.py` is generated from
that JSON by `docs/schema/generate_dataclass.py`. The mirror gets vendored
into the parsers submodule (Task 2.3) so the submodule does not depend on
main-repo `pdfsys-core`.

### CLI

| Command | Behavior |
|---|---|
| `pdfsys release status` | Read `system_release.toml`, walk each submodule path, print a pin-vs-HEAD diff. Exit 0. |
| `pdfsys release lock` | Read submodule HEADs, write `system_release.toml` in-place (comment-preserving via `tomlkit`). Refuses if working tree dirty. Exit 0 / 1. |
| `pdfsys release verify` | CI guard. Exit 1 if any non-in-tree component is drifted or missing. |

In-tree components are skipped by `lock` and `verify` — they are user-managed
and only bumped when their host package cuts a release.

### TOML schema

See `docs/superpowers/specs/2026-05-30-parsers-submodule-design.md` §5 for the
full schema. The data model dataclasses live in `packages/pdfsys-cli/src/pdfsys_cli/release.py`
(`SystemRelease`, `Component`, `Runtime`). Validation rules (40-char lowercase
hex SHA, required keys, `t_publish` rejection, etc.) are enforced at parse
time and tested in `tests/release/test_toml_schema.py`.

### Transitional state (as of 2026-06-06)

Both components are pinned `in-tree:` in `system_release.toml`. The submodule
extraction (Spec #1 → `parsers`) is in progress. Once Task 2.4 lands the
submodule, `pdfsys release lock` will produce the first real pin diff
(`components.parsers.commit: <main-repo-sha> → <parsers-repo-sha>`).
