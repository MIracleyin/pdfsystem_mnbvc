"""Tier A: PipelineParser unit tests with mocked mineru.cli.common.do_parse.

These tests must NEVER load real mineru models. The mock writes a known
.md + sidecars to the output_dir so the parser's read-back logic is
exercised end-to-end without touching the network or disk-cache models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pdfsys_core import Backend, ExtractedDoc, PipelineConfig
from pdfsys_parser_pipeline import PipelineParser


def _make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\n%stub\n") -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(content)
    return p


def _fake_do_parse(expected_md: str, expected_middle: dict, expected_content: list):
    """Returns a side_effect that writes mineru-shaped outputs to the dir
    the parser would pass in."""
    def _side_effect(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
                     backend, **kwargs):
        # mineru lays files at <output_dir>/<pdf_name>/<parse_method>/<pdf_name>.md
        parse_method = "auto"
        for name in pdf_file_names:
            md_dir = Path(output_dir) / name / parse_method
            md_dir.mkdir(parents=True, exist_ok=True)
            (md_dir / "images").mkdir(exist_ok=True)
            (md_dir / f"{name}.md").write_text(expected_md, encoding="utf-8")
            (md_dir / f"{name}_middle.json").write_text(
                json.dumps(expected_middle), encoding="utf-8"
            )
            (md_dir / f"{name}_content_list.json").write_text(
                json.dumps(expected_content), encoding="utf-8"
            )
    return _side_effect


def test_extract_returns_doc_with_markdown(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()

    fake = _fake_do_parse("# Hello\n\nWorld.\n", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake) as m:
        parser = PipelineParser(PipelineConfig(output_dir=tmp_path / "out"))
        doc = parser.extract(pdf)

    assert isinstance(doc, ExtractedDoc)
    assert doc.backend == Backend.PIPELINE
    assert doc.sha256 == expected_sha
    assert doc.markdown == "# Hello\n\nWorld.\n"

    # mineru received the right backend argument
    assert m.call_count == 1
    _, kwargs = m.call_args
    assert kwargs["backend"] == "pipeline"
    assert kwargs["p_lang_list"] == ["ch"]
    assert kwargs["formula_enable"] is True
    assert kwargs["table_enable"] is True


def test_extract_records_sidecar_paths(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    fake = _fake_do_parse("md", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake):
        parser = PipelineParser(PipelineConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    assert doc.stats["mineru_backend"] == "pipeline"
    assert doc.stats["middle_json_path"] is not None
    assert doc.stats["content_list_path"] is not None
    # Paths are relative to output_dir
    middle_full = out_dir / doc.stats["middle_json_path"]
    assert middle_full.exists()


def test_extract_tmpdir_when_output_dir_none(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("# X", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake):
        parser = PipelineParser(PipelineConfig(output_dir=None))
        doc = parser.extract(pdf)

    assert doc.markdown == "# X"
    # Sidecar paths are null when no persistent output_dir
    assert doc.stats["middle_json_path"] is None
    assert doc.stats["content_list_path"] is None


def test_extract_uses_config_p_lang(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    fake = _fake_do_parse("md", {"pages": []}, [])
    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=fake) as m:
        parser = PipelineParser(PipelineConfig(p_lang="en", output_dir=tmp_path / "o"))
        parser.extract(pdf)

    _, kwargs = m.call_args
    assert kwargs["p_lang_list"] == ["en"]


def test_extract_propagates_do_parse_errors(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    def _raise(*a, **kw):
        raise RuntimeError("simulated mineru failure")

    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=_raise):
        parser = PipelineParser(PipelineConfig(output_dir=tmp_path / "o"))
        with pytest.raises(RuntimeError, match="simulated mineru failure"):
            parser.extract(pdf)


def test_extract_raises_when_markdown_missing(tmp_path: Path) -> None:
    """If mineru returns without writing a .md, surface a clear error."""
    pdf = _make_pdf(tmp_path)

    def _do_nothing(*a, **kw):
        pass  # mineru wrote nothing

    with patch("pdfsys_parser_pipeline.extract.do_parse", side_effect=_do_nothing):
        parser = PipelineParser(PipelineConfig(output_dir=tmp_path / "o"))
        with pytest.raises(FileNotFoundError, match="markdown"):
            parser.extract(pdf)
