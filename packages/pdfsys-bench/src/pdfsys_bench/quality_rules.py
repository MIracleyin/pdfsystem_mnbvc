"""Layer-1 deterministic hard-rule checks for extracted text.

Cheap, pure-stdlib, model-free gate that runs BEFORE any BERT/LLM
scorer. Designed for a cascade-with-early-exit pipeline: the cheapest
parser (e.g. MuPDF) runs, its output is hard-checked, and if any
blocker fires the orchestrator escalates to the next parser without
ever loading a model.

Each blocker is intentionally permissive — defaults are tuned to catch
only "obviously broken" output, leaving graded judgements to the
BERT/LLM scorers downstream.

Blockers emitted:
* ``empty_output``           — stripped text is empty
* ``too_short``              — chars/page below threshold
* ``high_replacement_chars`` — U+FFFD density above threshold
* ``high_garbage_chars``     — control / unassigned / private-use density above threshold
* ``repetition_loop``        — same non-empty line repeated consecutively above threshold
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

DEFAULT_MIN_CHARS_PER_PAGE = 50
DEFAULT_MAX_REPLACEMENT_RATIO = 0.005
DEFAULT_MAX_GARBAGE_RATIO = 0.01
DEFAULT_REPETITION_MIN_RUN = 10

REPLACEMENT_CHAR = "�"

# Cc/Cn/Co/Cs Unicode categories mark broken / encoding-artefact chars,
# but \t \n \r are legitimate Cc whitespace in extracted text.
_LEGITIMATE_CONTROL_CHARS = frozenset({"\t", "\n", "\r"})


@dataclass(slots=True, frozen=True)
class HardCheckResult:
    blockers: dict[str, bool]
    metrics: dict[str, float]

    @property
    def any_blocker(self) -> bool:
        return any(self.blockers.values())

    def as_record(self) -> dict[str, Any]:
        return {
            "hard_check_blockers": dict(self.blockers),
            "hard_check_metrics": dict(self.metrics),
            "hard_check_any_blocker": self.any_blocker,
        }


def check_extracted_text(
    text: str,
    page_count: int = 1,
    *,
    min_chars_per_page: int = DEFAULT_MIN_CHARS_PER_PAGE,
    max_replacement_ratio: float = DEFAULT_MAX_REPLACEMENT_RATIO,
    max_garbage_ratio: float = DEFAULT_MAX_GARBAGE_RATIO,
    repetition_min_run: int = DEFAULT_REPETITION_MIN_RUN,
) -> HardCheckResult:
    text = text or ""
    pages = max(page_count, 1)
    total_chars = len(text)
    stripped_len = len(text.strip())

    chars_per_page = stripped_len / pages
    replacement_count = text.count(REPLACEMENT_CHAR)
    replacement_ratio = (replacement_count / total_chars) if total_chars else 0.0

    garbage_count = _count_garbage(text)
    garbage_ratio = (garbage_count / total_chars) if total_chars else 0.0

    max_run = _max_consecutive_line_run(text)

    blockers = {
        "empty_output": stripped_len == 0,
        "too_short": chars_per_page < min_chars_per_page,
        "high_replacement_chars": replacement_ratio > max_replacement_ratio,
        "high_garbage_chars": garbage_ratio > max_garbage_ratio,
        "repetition_loop": max_run > repetition_min_run,
    }
    metrics = {
        "chars_per_page": chars_per_page,
        "replacement_ratio": replacement_ratio,
        "garbage_ratio": garbage_ratio,
        "max_repetition_run": float(max_run),
    }
    return HardCheckResult(blockers=blockers, metrics=metrics)


def _count_garbage(text: str) -> int:
    n = 0
    for ch in text:
        if ch in _LEGITIMATE_CONTROL_CHARS:
            continue
        if ch == REPLACEMENT_CHAR:
            # Counted separately as replacement_ratio; avoid double-count.
            continue
        # Cc=control, Cn=unassigned, Co=private-use, Cs=surrogate
        if unicodedata.category(ch).startswith("C"):
            n += 1
    return n


def _max_consecutive_line_run(text: str) -> int:
    best = 0
    current = 0
    prev: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            prev = None
            current = 0
            continue
        if line == prev:
            current += 1
        else:
            current = 1
            prev = line
        if current > best:
            best = current
    return best
