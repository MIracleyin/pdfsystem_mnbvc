"""Tier B: real mineru VLM-mode invocation. Slow, gated.

Runs only when MINERU_INTEGRATION=1. Downloads ~7GB on first run.
Defaults to engine=transformers (most portable). Override via
MINERU_VLM_ENGINE=mlx-engine on Apple Silicon for speed.

    MINERU_INTEGRATION=1 uv run pytest tests/parsers/integration/test_vlm_integration.py -v -s
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
_ENGINE = os.environ.get("MINERU_VLM_ENGINE", "transformers")


def _first_pdf() -> Path:
    pdfs = sorted(p for p in _PDF_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        pytest.skip(f"no PDFs found under {_PDF_DIR}")
    return pdfs[0]


def test_vlm_real_extraction(tmp_path: Path) -> None:
    from pdfsys_core import Backend, VlmConfig
    from pdfsys_parser_vlm import VlmParser

    pdf = _first_pdf()
    parser = VlmParser(VlmConfig(engine=_ENGINE, output_dir=tmp_path))
    doc = parser.extract(pdf)

    assert doc.backend == Backend.VLM
    assert doc.sha256
    assert len(doc.markdown.strip()) > 0
    assert doc.stats["mineru_backend"] == f"vlm-{_ENGINE}"
