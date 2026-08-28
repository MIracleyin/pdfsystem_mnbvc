"""Tier A: PipelineParser unit tests.

Mock the HTTP layer + subprocess startup so tests don't spawn
``mineru-api`` or hit the network. Each test runs in well under 100ms.
"""

from __future__ import annotations

import base64
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
    images: dict | None = None,
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
    if images is not None:
        result["images"] = images
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


# ---------------------------------------------------------------- image crops
#
# The crops are what `content_list.json`'s `img_path` entries point at. When
# the client asked for return_images=false they existed only inside the
# mineru-api process's own filesystem — which under docker-compose is not a
# volume the client shares, so `pdfsys dataset --images crops` had nothing to
# read.

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 3


def _data_uri(payload: bytes = PNG, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def test_crops_are_requested_by_default() -> None:
    """The whole point: the request has to ask for them."""
    assert PipelineConfig().return_images is True


def test_extract_writes_crops_next_to_the_sidecars(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    response = _fake_response(
        content_list=[{"type": "image", "img_path": "images/fig1.png"}],
        images={"fig1.png": _data_uri(), "fig2.jpg": _data_uri(mime="image/jpeg")},
    )

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        doc = PipelineParser(PipelineConfig(output_dir=out_dir)).extract(pdf)

    images_dir = out_dir / doc.sha256 / "images"
    assert doc.stats["images_written"] == 2
    assert (images_dir / "fig1.png").read_bytes() == PNG
    assert (images_dir / "fig2.jpg").exists()


def test_the_request_carries_the_return_images_flag(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response()

    sp, pp = _patched_parser(None, response)
    with sp, pp as post:
        PipelineParser(PipelineConfig(return_images=False)).extract(pdf)

    assert post.call_args.kwargs["data"]["return_images"] == "false"


def test_a_crop_filename_cannot_escape_the_output_directory(tmp_path: Path) -> None:
    """The filename comes from the server. A figure crop is never worth
    letting a path traversal out."""
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    response = _fake_response(images={"../../escaped.png": _data_uri()})

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        doc = PipelineParser(PipelineConfig(output_dir=out_dir)).extract(pdf)

    assert (out_dir / doc.sha256 / "images" / "escaped.png").exists()
    assert not (tmp_path / "escaped.png").exists()


def test_a_malformed_crop_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One unreadable crop must not cost us the document's text."""
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    response = _fake_response(
        images={"good.png": _data_uri(), "bad.png": "not-a-data-uri", "none.png": None}
    )

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        doc = PipelineParser(PipelineConfig(output_dir=out_dir)).extract(pdf)

    assert doc.stats["images_written"] == 1
    assert doc.markdown, "the markdown still has to come back"


def test_no_output_dir_means_no_crops_and_no_crash(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(images={"fig1.png": _data_uri()})

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        doc = PipelineParser(PipelineConfig(output_dir=None)).extract(pdf)

    assert doc.stats["images_written"] == 0


def test_a_response_without_images_is_fine(tmp_path: Path) -> None:
    """Older mineru-api builds, or return_images=false, simply omit the key."""
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    response = _fake_response(content_list=[])

    sp, pp = _patched_parser(None, response)
    with sp, pp:
        doc = PipelineParser(PipelineConfig(output_dir=out_dir)).extract(pdf)

    assert doc.stats["images_written"] == 0


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
