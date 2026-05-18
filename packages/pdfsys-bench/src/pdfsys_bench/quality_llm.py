"""LLM-based OCR quality scorer.

Companion to :mod:`pdfsys_bench.quality` (FinePDFs ModernBERT regression).
Where the BERT scorer emits a black-box scalar, this one asks an
OpenAI-compatible chat model to rate extracted markdown on the same
``0-3`` scale **and** return a short natural-language reason — useful
for annotation review and bad-case triage.

Output shape mirrors ``QualityScore.as_record()`` so the viz layer can
display both side by side.

The model is configured via ``.env`` (see ``LlmConfig.from_env``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm_client import LlmClient, LlmConfig

# Match the BERT scorer's truncation budget — keeps token cost bounded
# and gives the model roughly the same window the BERT model sees.
DEFAULT_MAX_CHARS = 8_000

_SYSTEM_PROMPT = (
    "You are an OCR/extraction quality auditor for a PDF→Markdown pipeline. "
    "You will be shown the Markdown extracted from one PDF. Rate the "
    "OCR/extraction quality on the integer scale 0-3:\n"
    "  0 = garbage / unreadable, content cannot be recovered\n"
    "  1 = significant formatting or OCR errors, only partly readable\n"
    "  2 = minor problems (some broken layout, missing whitespace, a few "
    "typos), but the content is essentially recoverable\n"
    "  3 = clean text, well-structured Markdown, no visible OCR errors\n\n"
    "Judge ONLY the extraction quality — not whether the document itself "
    "is interesting. Empty / near-empty input is 0.\n\n"
    'Reply with a single JSON object on one line: '
    '{"score": <0|1|2|3>, "reason": "<one short sentence in the same '
    'language as the input>"}.\n'
    "Do not wrap in code fences. Do not output anything else."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class LlmQualityScore:
    """Result of one LLM scoring call."""

    score: float
    reason: str
    model: str
    num_chars: int
    raw_content: str
    parse_error: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "quality_score_llm": self.score,
            "quality_reason_llm": self.reason,
            "quality_model_llm": self.model,
            "quality_num_chars_llm": self.num_chars,
            "quality_parse_error_llm": self.parse_error,
        }


class LlmQualityScorer:
    """OCR quality scorer that delegates judgment to a chat model."""

    def __init__(
        self,
        client: LlmClient | None = None,
        config: LlmConfig | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.client = client or LlmClient(config or LlmConfig.from_env())
        self.max_chars = max_chars

    def score(self, text: str) -> LlmQualityScore:
        """Score a single document. Empty input short-circuits to 0."""
        model_name = self.client.config.model

        if not text or not text.strip():
            return LlmQualityScore(
                score=0.0,
                reason="empty input",
                model=model_name,
                num_chars=0,
                raw_content="",
                parse_error=None,
            )

        clipped = text[: self.max_chars]
        raw = self.client.chat(
            prompt=clipped,
            system=_SYSTEM_PROMPT,
            temperature=0.0,
        )

        score, reason, parse_error = _parse_response(raw)
        return LlmQualityScore(
            score=score,
            reason=reason,
            model=model_name,
            num_chars=len(clipped),
            raw_content=raw,
            parse_error=parse_error,
        )

    def score_many(self, texts: list[str]) -> list[LlmQualityScore]:
        return [self.score(t) for t in texts]


def _parse_response(raw: str) -> tuple[float, str, str | None]:
    """Extract score + reason from the model's reply.

    The system prompt asks for a one-line JSON object; the model sometimes
    wraps it in code fences or prefixes prose. We try strict JSON first,
    then a regex to find the first ``{...}`` block.
    """
    text = (raw or "").strip()
    if not text:
        return 0.0, "", "empty response"

    # Strip common code-fence wrappers.
    if text.startswith("```"):
        text = text.strip("`")
        # remove leading "json\n"
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # First attempt: parse as-is.
    obj: Any
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            return 0.0, raw.strip()[:200], "no JSON object in reply"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return 0.0, raw.strip()[:200], f"json parse failed: {e}"

    if not isinstance(obj, dict):
        return 0.0, str(obj)[:200], "reply is not a JSON object"

    score_raw = obj.get("score")
    reason = obj.get("reason") or ""
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        return 0.0, str(reason)[:500], f"score not numeric: {score_raw!r}"

    score = max(0.0, min(3.0, score))
    return score, str(reason)[:500], None
