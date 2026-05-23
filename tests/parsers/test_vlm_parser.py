"""Tier A: VlmParser unit tests with mocked mineru.cli.common.do_parse.

These tests must NEVER load real mineru VLM weights (~7GB).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pdfsys_core import Backend, ExtractedDoc, VlmConfig
from pdfsys_parser_vlm import VlmParser


def _make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\n%stub\n") -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(content)
    return p


def _fake_do_parse(expected_md: str):
    """Returns a side_effect writing mineru-shaped outputs."""
    async def _side_effect(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                           backend, **kwargs):
        parse_method = "vlm"  # mineru writes vlm output here regardless of parse_method kwarg
        for name in pdf_file_names:
            md_dir = Path(output_dir) / name / parse_method
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / "images").mkdir(exist_ok=True)
            (md_dir / f"{name}.md").write_text(expected_md, encoding="utf-8")
            (md_dir / f"{name}_middle.json").write_text(
                json.dumps({"pages": []}), encoding="utf-8"
            )
            (md_dir / f"{name}_content_list.json").write_text(
                json.dumps([]), encoding="utf-8"
            )
    return _side_effect


def test_vlm_extract_returns_doc(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()

    fake = _fake_do_parse("# VLM Output\n\n$$E=mc^2$$\n")
    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=fake) as m:
        parser = VlmParser(VlmConfig(output_dir=tmp_path / "out"))
        doc = parser.extract(pdf)

    assert isinstance(doc, ExtractedDoc)
    assert doc.backend == Backend.VLM
    assert doc.sha256 == expected_sha
    assert doc.markdown == "# VLM Output\n\n$$E=mc^2$$\n"

    _, kwargs = m.call_args
    assert kwargs["backend"] == "vlm-transformers"


def test_vlm_extract_with_mlx_engine(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("md")
    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=fake) as m:
        parser = VlmParser(VlmConfig(engine="mlx-engine", output_dir=tmp_path / "o"))
        parser.extract(pdf)

    _, kwargs = m.call_args
    assert kwargs["backend"] == "vlm-mlx-engine"


def test_vlm_extract_with_vllm_engine(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("md")
    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=fake) as m:
        parser = VlmParser(VlmConfig(engine="vllm-engine", output_dir=tmp_path / "o"))
        parser.extract(pdf)

    _, kwargs = m.call_args
    assert kwargs["backend"] == "vlm-vllm-engine"


def test_vlm_extract_records_sidecars(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    fake = _fake_do_parse("md")
    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=fake):
        parser = VlmParser(VlmConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    assert doc.stats["mineru_backend"] == "vlm-transformers"
    assert doc.stats["middle_json_path"] is not None
    assert doc.stats["content_list_path"] is not None


def test_vlm_extract_tmpdir_when_output_dir_none(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("# Y")
    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=fake):
        parser = VlmParser(VlmConfig(output_dir=None))
        doc = parser.extract(pdf)

    assert doc.markdown == "# Y"
    assert doc.stats["middle_json_path"] is None


def test_vlm_extract_propagates_errors(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    async def _raise(*a, **kw):
        raise RuntimeError("simulated vlm failure")

    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=_raise):
        parser = VlmParser(VlmConfig(output_dir=tmp_path / "o"))
        with pytest.raises(RuntimeError, match="simulated vlm failure"):
            parser.extract(pdf)


def test_vlm_extract_raises_when_markdown_missing(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    async def _do_nothing(*a, **kw):
        pass

    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=_do_nothing):
        parser = VlmParser(VlmConfig(output_dir=tmp_path / "o"))
        with pytest.raises(FileNotFoundError, match="markdown"):
            parser.extract(pdf)


def test_vlm_extract_sidecar_none_when_mineru_skips_middle_json(tmp_path: Path) -> None:
    """Mineru sometimes omits ``_middle.json`` for empty PDFs (spec §11)."""
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    async def _fake_no_middle(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                              backend, **kwargs):
        parse_method = "vlm"  # mineru writes vlm output here regardless of parse_method kwarg
        for name in pdf_file_names:
            md_dir = Path(output_dir) / name / parse_method
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / f"{name}.md").write_text("sparse vlm content", encoding="utf-8")
            (md_dir / f"{name}_content_list.json").write_text("[]", encoding="utf-8")
            # no _middle.json

    with patch("pdfsys_parser_vlm.extract.aio_do_parse", side_effect=_fake_no_middle):
        parser = VlmParser(VlmConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    assert doc.markdown == "sparse vlm content"
    assert doc.stats["middle_json_path"] is None
    assert doc.stats["content_list_path"] is not None
