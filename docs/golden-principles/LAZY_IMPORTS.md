# Lazy Imports for Heavy Dependencies

## Rule
Heavy ML dependencies (torch, transformers, magic-pdf, doclayout-yolo) MUST be imported inside functions or methods, not at module top level.

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

## Exceptions
- Test files can import eagerly (they always run the full stack).
- `pdfsys-core` has no heavy deps to lazy-import — this rule doesn't apply to it.
