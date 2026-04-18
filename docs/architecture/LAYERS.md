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
