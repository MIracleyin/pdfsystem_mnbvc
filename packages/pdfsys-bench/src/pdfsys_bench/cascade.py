"""Cascade-with-early-exit orchestration for the PDF extraction pipeline.

The cascade runs a sequence of parser stages (cheapest first), checks
each output against :func:`pdfsys_bench.quality_rules.check_extracted_text`,
and either *publishes* the first acceptable result or *escalates* to the
next stage. If all stages fail their Layer-1 gate, the cascade returns
``reject`` and keeps the last attempted output for human review.

The module is intentionally parser-agnostic: a :class:`CascadeStage` is
just a name plus a callable ``(Path) -> ExtractedDoc``. The bench
runner (or any caller) wires real MuPDF / Pipeline / VLM parsers behind
this interface, but the cascade engine itself is pure logic and fully
unit-testable without heavy ML dependencies.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdfsys_core import ExtractedDoc

from .quality_rules import check_extracted_text

PUBLISH = "publish"
ESCALATE = "escalate"
REJECT = "reject"


@dataclass(slots=True)
class CascadeStage:
    """One parser stage in the cascade.

    Attributes
    ----------
    name:
        Short identifier used in logs and the ``CascadeAttempt`` record
        (e.g. ``"mupdf"``, ``"pipeline"``, ``"vlm"``).
    extract:
        Callable invoked with the PDF path; must return an ``ExtractedDoc``.
        Exceptions are caught by the cascade and recorded as escalations.
    rules_kwargs:
        Per-stage overrides forwarded to ``check_extracted_text``.
        Use this to relax / tighten thresholds for parsers with known
        characteristics (e.g. allow shorter output from a VLM cover-page
        extractor).
    """

    name: str
    extract: Callable[[Path], ExtractedDoc]
    rules_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CascadeAttempt:
    """One stage's outcome — what we extracted, what the gate said, why."""

    stage: str
    decision: str  # PUBLISH | ESCALATE
    blockers: dict[str, bool]
    metrics: dict[str, float]
    error: str | None
    wall_ms: float

    def as_record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision,
            "blockers": dict(self.blockers),
            "metrics": dict(self.metrics),
            "error": self.error,
            "wall_ms": self.wall_ms,
        }


@dataclass(slots=True)
class CascadeResult:
    """Final cascade outcome plus full audit trail of every attempt."""

    decision: str  # PUBLISH | REJECT
    final_stage: str | None
    extracted: ExtractedDoc | None
    attempts: list[CascadeAttempt]
    total_wall_ms: float

    def as_record(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "final_stage": self.final_stage,
            "attempts": [a.as_record() for a in self.attempts],
            "total_wall_ms": self.total_wall_ms,
        }


def run_cascade(
    pdf_path: Path,
    stages: Sequence[CascadeStage],
    *,
    num_pages: int = 1,
) -> CascadeResult:
    """Run ``stages`` in order, publishing the first one that passes the gate.

    Parameters
    ----------
    pdf_path:
        Forwarded to each stage's extractor.
    stages:
        Stages tried in order (cheapest first). Must be non-empty.
    num_pages:
        Page count used by the Layer-1 ``chars_per_page`` check. Passing
        the real page count lets the cascade reject "mupdf returned 50
        chars from a 30-page scanned PDF" without paying for layout.

    Returns
    -------
    CascadeResult
        ``decision == PUBLISH`` and a non-None ``extracted`` if any
        stage passed; ``decision == REJECT`` otherwise (with the last
        attempted ``extracted`` preserved for human review).
    """
    if not stages:
        raise ValueError("run_cascade requires at least one stage")

    attempts: list[CascadeAttempt] = []
    last_extracted: ExtractedDoc | None = None
    last_stage_name: str | None = None
    total = 0.0

    for stage in stages:
        t0 = time.perf_counter()
        extracted: ExtractedDoc | None = None
        error: str | None = None
        try:
            extracted = stage.extract(pdf_path)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        wall = (time.perf_counter() - t0) * 1000.0
        total += wall

        if extracted is not None:
            last_extracted = extracted
            last_stage_name = stage.name

        # An exception means "this stage couldn't produce output at all",
        # so we always escalate — no point gating against missing text.
        if error is not None:
            attempts.append(
                CascadeAttempt(
                    stage=stage.name,
                    decision=ESCALATE,
                    blockers={},
                    metrics={},
                    error=error,
                    wall_ms=wall,
                )
            )
            continue

        assert extracted is not None  # for type-checkers
        check = check_extracted_text(
            extracted.markdown,
            page_count=num_pages,
            **stage.rules_kwargs,
        )

        if not check.any_blocker:
            attempts.append(
                CascadeAttempt(
                    stage=stage.name,
                    decision=PUBLISH,
                    blockers=dict(check.blockers),
                    metrics=dict(check.metrics),
                    error=None,
                    wall_ms=wall,
                )
            )
            return CascadeResult(
                decision=PUBLISH,
                final_stage=stage.name,
                extracted=extracted,
                attempts=attempts,
                total_wall_ms=total,
            )

        attempts.append(
            CascadeAttempt(
                stage=stage.name,
                decision=ESCALATE,
                blockers=dict(check.blockers),
                metrics=dict(check.metrics),
                error=None,
                wall_ms=wall,
            )
        )

    return CascadeResult(
        decision=REJECT,
        final_stage=last_stage_name,
        extracted=last_extracted,
        attempts=attempts,
        total_wall_ms=total,
    )
