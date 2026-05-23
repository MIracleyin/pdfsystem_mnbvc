"""Tier B: real mineru pipeline-mode invocation. Slow, gated.

Runs only when MINERU_INTEGRATION=1. Downloads ~3GB of weights on
first run; subsequent runs use the local cache. Run via:

    MINERU_INTEGRATION=1 uv run pytest tests/parsers/integration/ -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_GATE = os.environ.get("MINERU_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(
    not _GATE, reason="set MINERU_INTEGRATION=1 to run mineru integration tests"
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PDF_DIR = _REPO_ROOT / "packages" / "pdfsys-bench" / "omnidocbench_100" / "pdfs"


def _first_pdf() -> Path:
    pdfs = sorted(p for p in _PDF_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        pytest.skip(f"no PDFs found under {_PDF_DIR}")
    return pdfs[0]


def test_pipeline_real_extraction(tmp_path: Path) -> None:
    from pdfsys_core import Backend, PipelineConfig
    from pdfsys_parser_pipeline import PipelineParser

    pdf = _first_pdf()
    parser = PipelineParser(PipelineConfig(output_dir=tmp_path))
    doc = parser.extract(pdf)

    assert doc.backend == Backend.PIPELINE
    assert doc.sha256
    assert len(doc.markdown.strip()) > 0, "expected non-empty markdown"
    assert doc.stats["mineru_backend"] == "pipeline"
    assert doc.stats["middle_json_path"] is not None
