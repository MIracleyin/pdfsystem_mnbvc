# Lazy Imports for Heavy Dependencies

## Rule
Heavy ML dependencies (torch, transformers, mineru, doclayout-yolo) MUST be imported inside functions or methods, not at module top level.

## DO

```python
# Good: lazy import inside method
class OcrQualityScorer:
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch                                          # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415

        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
```

## DON'T

```python
# Bad: top-level import of heavy dependency
import torch
from transformers import AutoModelForSequenceClassification

class OcrQualityScorer:
    ...
```

## Why
`import torch` takes 2-3 seconds and loads 500 MB of shared libraries. If every module imports torch at the top, `from pdfsys_bench import loop` becomes a 5-second penalty even when the user only wants to read a config. Lazy imports keep startup instant for CLI operations that don't need ML inference.

## Stronger pattern: out-of-process isolation
Lazy import keeps a heavy dep out of *startup*, but it still loads into the host process on first use — and on macOS that triggered spawn-pool / MPS-vs-MLX deadlocks during the mineru migration. The parsers (`mineru`) and the quality scorer (`torch`/`transformers`) therefore go one step further: they never import the heavy dep at all, but spawn a subprocess (`mineru-api`, `_quality_server`) and call it over HTTP. Prefer this for any dep that conflicts with another heavy runtime in the same process. See `docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md §15`.

## Exceptions
- Test files can import eagerly (they always run the full stack).
- `pdfsys-core` has no heavy deps to lazy-import — this rule doesn't apply to it.
