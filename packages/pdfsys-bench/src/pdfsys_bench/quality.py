"""OCR quality scorer backed by the FinePDFs ModernBERT classifier.

Wraps ``HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn`` — a
single-head regression fine-tune of ModernBERT-large (~0.4 B params)
that emits a float in ``[0, 3]`` where:

* 0 → garbage / unreadable OCR
* 1 → formatting issues but mostly readable
* 2 → minor problems
* 3 → clean text

The scorer takes raw extracted text (Markdown or plain), truncates to at
most ``max_chars`` characters before tokenization, tokenizes with the
model's own tokenizer, runs one forward pass, and returns the scalar.

Heavy dependencies (``torch`` + ``transformers``) are imported lazily so
that merely importing :mod:`pdfsys_bench` does not pull them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn"
DEFAULT_MAX_CHARS = 10_000
# Upstream FinePDFs uses max_tokens=2048, but ModernBERT-large activations
# at that length need ≈ 3 GB of RAM — too much for a 4 GB dev box. 512
# tokens is enough to give a stable quality signal in practice and keeps
# peak memory well under a gig.
DEFAULT_MAX_TOKENS = 512


@dataclass(slots=True)
class QualityScore:
    """Result of scoring one document."""

    score: float
    num_chars: int
    num_tokens: int
    model: str

    def as_record(self) -> dict[str, Any]:
        return {
            "quality_score": self.score,
            "quality_num_chars": self.num_chars,
            "quality_num_tokens": self.num_tokens,
            "quality_model": self.model,
        }


class OcrQualityScorer:
    """Lazy ModernBERT regression scorer. Re-uses model/tokenizer across calls."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self._device_name = device
        self.dtype_name = dtype
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415 — lazy import is intentional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._device = torch.device(
            self._device_name
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Use bfloat16 on CPU to halve the model's memory footprint —
        # ModernBERT-large is ~0.4 B params, so fp32 weights alone take
        # ~1.6 GB and OOM a 4 GB-RAM dev box. bf16 inference is
        # numerically stable enough for a regression head like this.
        torch_dtype = getattr(torch, self.dtype_name, torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # ``dtype`` is the transformers≥5 name; ``torch_dtype`` was the
        # transformers<5 name. Pass ``dtype`` and fall back for older releases.
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                dtype=torch_dtype,
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
            )
        model.eval()
        model.to(self._device)
        self._model = model

    def score(self, text: str) -> QualityScore:
        """Score a single document. Empty input returns 0.0."""
        if not text or not text.strip():
            return QualityScore(
                score=0.0, num_chars=0, num_tokens=0, model=self.model_name
            )

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        torch = self._torch

        clipped = text[: self.max_chars]
        enc = self._tokenizer(
            clipped,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        num_tokens = int(enc["input_ids"].shape[1])
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.inference_mode():
            out = self._model(**enc)
            logits = out.logits  # shape [1, 1] for regression
            raw = float(logits.squeeze().item())
        # Drop the forward-pass tensors eagerly so large-seq runs on CPU
        # don't hold onto activations between calls.
        del enc, out, logits

        # Clamp to the documented [0, 3] range.
        clamped = max(0.0, min(3.0, raw))

        return QualityScore(
            score=clamped,
            num_chars=len(clipped),
            num_tokens=num_tokens,
            model=self.model_name,
        )

    def score_many(self, texts: list[str]) -> list[QualityScore]:
        """Serial scoring — tiny MVP harness, not a batched hot path."""
        return [self.score(t) for t in texts]
