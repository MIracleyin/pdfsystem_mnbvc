"""Tier A: PipelineParser unit tests.

Mock the HTTP layer + subprocess startup so tests don't spawn
``mineru-api`` or hit the network. Each test runs in well under 100ms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdfsys_core import Backend, ExtractedDoc, PipelineConfig
from pdfsys_parser_pipeline import PipelineParser


def _make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\n%stub\n") -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(content)
    return p


def _fake_response(
    *,
    status_code: int = 200,
    status: str = "completed",
    markdown: str | None = "# Hello\n\nWorld.\n",
    middle_json: dict | None = None,
    content_list: list | None = None,
    error: str | None = None,
    version: str = "3.1.14",
) -> MagicMock:
    """Build a MagicMock that mimics ``httpx.Response`` for ``/file_parse``."""
    resp = MagicMock()
    resp.status_code = status_code
    if status_code != 200:
        resp.text = error or "server error"
        return resp
    result = {}
    if markdown is not None:
        result["md_content"] = markdown
    if middle_json is not None:
        result["middle_json"] = middle_json
    if content_list is not None:
        result["content_list"] = content_list
    resp.json.return_value = {
        "task_id": "fake-task-id",
        "status": status,
        "backend": "pipeline",
        "version": version,
        "error": error,
        "results": {"fake.pdf": result} if status == "completed" else {},
    }
    return resp


def _patched_parser(parser: PipelineParser, response: MagicMock) -> tuple:
    """Skip subprocess startup, mock httpx.post; returns the two patches."""
    server_patch = patch.object(
        PipelineParser, "_ensure_server", return_value="http://test"
    )
    post_patch = patch(
        "pdfsys_parser_pipeline.extract.httpx.post", return_value=response
    )
    return server_patch, post_patch


# ---------------------------------------------------------------- happy path


def test_extract_returns_doc_with_markdown(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    response = _fake_response(markdown="# Hello\n\nWorld.\n")

    sp, pp = _patched_parser(None, response)
    with sp, pp as post:
        parser = PipelineParser(PipelineConfig())
        doc = parser.extract(pdf)

    assert isinstance(doc, ExtractedDoc)
    assert doc.backend == Backend.PIPELINE
    assert doc.sha256 == expected_sha
    assert doc.markdown == "# Hello\n\nWorld.\n"

    # Verify the HTTP request shape
    assert post.call_count == 1
    _, kwargs = post.call_args
    assert kwargs["data"]["backend"] == "pipeline"
    assert kwargs["data"]["lang_list"] == "ch"
    assert kwargs["data"]["return_md"] == "true"


def test_extract_uses_config_p_lang(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response()

    sp, pp = _patched_parser(None, response)
    with sp, pp as post:
        parser = PipelineParser(PipelineConfig(p_lang="en"))
        parser.extract(pdf)

    _, kwargs = post.call_args
    assert kwargs["data"]["lang_list"] == "en"


def test_extract_disabled_formula_and_table(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response()

    sp, pp = _patched_parser(None, response)
    with sp, pp as post:
        parser = PipelineParser(
            PipelineConfig(formula_enable=False, table_enable=False)
        )
        parser.extract(pdf)

    _, kwargs = post.call_args
    assert kwargs["data"]["formula_enable"] == "false"
    assert kwargs["data"]["table_enable"] == "false"


# ---------------------------------------------------------------- sidecars


def test_extract_persists_sidecars_when_output_dir_set(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    middle = {"pages": [{"a": 1}]}
    content = [{"type": "text", "text": "hi"}]
    response = _fake_response(middle_json=middle, content_list=content)

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        parser = PipelineParser(PipelineConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    sha = doc.sha256
    assert doc.stats["middle_json_path"] == f"{sha}/{sha}_middle.json"
    assert doc.stats["content_list_path"] == f"{sha}/{sha}_content_list.json"
    assert (out_dir / sha / f"{sha}_middle.json").exists()
    assert (out_dir / sha / f"{sha}_content_list.json").exists()


def test_extract_no_sidecars_when_output_dir_none(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(middle_json={"pages": []}, content_list=[])

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        parser = PipelineParser(PipelineConfig(output_dir=None))
        doc = parser.extract(pdf)

    assert doc.stats["middle_json_path"] is None
    assert doc.stats["content_list_path"] is None


# ---------------------------------------------------------------- errors


def test_extract_raises_on_http_error(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(status_code=500, error="internal boom")

    sp, pp = _patched_parser(None, response)
    with sp, pp, pytest.raises(RuntimeError, match="500"):
        PipelineParser().extract(pdf)


def test_extract_raises_on_task_failure(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(status="failed", error="model crashed")

    sp, pp = _patched_parser(None, response)
    with sp, pp, pytest.raises(RuntimeError, match="model crashed"):
        PipelineParser().extract(pdf)


def test_extract_raises_when_markdown_empty(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(markdown="")

    sp, pp = _patched_parser(None, response)
    with sp, pp, pytest.raises(RuntimeError, match="empty markdown"):
        PipelineParser().extract(pdf)


# ---------------------------------------------------------------- stats


def test_stats_include_mineru_backend_and_url(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(version="3.1.14")

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        parser = PipelineParser()
        doc = parser.extract(pdf)

    assert doc.stats["mineru_backend"] == "pipeline"
    assert doc.stats["mineru_api_url"] == "http://test"
    assert doc.stats["mineru_version"] == "3.1.14"
