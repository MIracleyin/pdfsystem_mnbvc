"""Tier A: VlmParser unit tests.

Mock the HTTP layer + subprocess startup so tests don't spawn
``mineru-api`` or load MLX weights. Each test runs in well under 100ms.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdfsys_core import Backend, ExtractedDoc, VlmConfig
from pdfsys_parser_vlm import VlmParser


def _make_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\n%stub\n") -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(content)
    return p


def _fake_response(
    *,
    status_code: int = 200,
    status: str = "completed",
    backend: str = "vlm-transformers",
    markdown: str | None = "# VLM\n",
    middle_json: dict | None = None,
    content_list: list | None = None,
    images: dict | None = None,
    error: str | None = None,
    version: str = "3.1.14",
) -> MagicMock:
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
        "backend": backend,
        "version": version,
        "error": error,
        "results": {"fake.pdf": result} if status == "completed" else {},
    }
    return resp


def _patched_parser(response: MagicMock) -> tuple:
    server_patch = patch.object(
        VlmParser, "_ensure_server", return_value="http://test"
    )
    post_patch = patch(
        "pdfsys_parser_vlm.extract.httpx.post", return_value=response
    )
    return server_patch, post_patch


# ---------------------------------------------------------------- happy path


def test_vlm_extract_returns_doc(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    expected_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    response = _fake_response(backend="vlm-transformers", markdown="# VLM\n")

    sp, pp = _patched_parser(response)
    with sp, pp as post:
        parser = VlmParser(VlmConfig())
        doc = parser.extract(pdf)

    assert isinstance(doc, ExtractedDoc)
    assert doc.backend == Backend.VLM
    assert doc.sha256 == expected_sha
    assert doc.markdown == "# VLM\n"

    _, kwargs = post.call_args
    assert kwargs["data"]["backend"] == "vlm-transformers"


def test_vlm_extract_with_mlx_engine(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(backend="vlm-mlx-engine")

    sp, pp = _patched_parser(response)
    with sp, pp as post:
        parser = VlmParser(VlmConfig(engine="mlx-engine"))
        parser.extract(pdf)

    _, kwargs = post.call_args
    assert kwargs["data"]["backend"] == "vlm-mlx-engine"


def test_vlm_extract_with_vllm_engine(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(backend="vlm-vllm-engine")

    sp, pp = _patched_parser(response)
    with sp, pp as post:
        parser = VlmParser(VlmConfig(engine="vllm-engine"))
        parser.extract(pdf)

    _, kwargs = post.call_args
    assert kwargs["data"]["backend"] == "vlm-vllm-engine"


def test_vlm_extract_uses_config_p_lang(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response()

    sp, pp = _patched_parser(response)
    with sp, pp as post:
        parser = VlmParser(VlmConfig(p_lang="en"))
        parser.extract(pdf)

    _, kwargs = post.call_args
    assert kwargs["data"]["lang_list"] == "en"


# ---------------------------------------------------------------- sidecars


def test_vlm_extract_persists_sidecars(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    middle = {"pages": [{"a": 1}]}
    content = [{"type": "text", "text": "hi"}]
    response = _fake_response(middle_json=middle, content_list=content)

    sp, pp = _patched_parser(response)
    with sp, pp:
        parser = VlmParser(VlmConfig(output_dir=out_dir))
        doc = parser.extract(pdf)

    sha = doc.sha256
    assert doc.stats["middle_json_path"] == f"{sha}/{sha}_middle.json"
    assert (out_dir / sha / f"{sha}_middle.json").exists()
    assert (out_dir / sha / f"{sha}_content_list.json").exists()


def test_vlm_extract_no_sidecars_when_output_dir_none(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(middle_json={"pages": []}, content_list=[])

    sp, pp = _patched_parser(response)
    with sp, pp:
        parser = VlmParser(VlmConfig(output_dir=None))
        doc = parser.extract(pdf)

    assert doc.stats["middle_json_path"] is None
    assert doc.stats["content_list_path"] is None


# ---------------------------------------------------------------- image crops
#
# Same change as the pipeline lane: the crops `content_list.json` points at
# used to stay inside the mineru-api process's own filesystem. Kept symmetric
# on purpose — the two parsers share this code path verbatim.

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 3


def _data_uri(payload: bytes = PNG, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def test_vlm_crops_are_requested_by_default() -> None:
    assert VlmConfig().return_images is True


def test_vlm_extract_writes_crops_next_to_the_sidecars(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"
    response = _fake_response(images={"fig1.png": _data_uri()})

    sp, pp = _patched_parser(response)
    with sp, pp:
        doc = VlmParser(VlmConfig(output_dir=out_dir)).extract(pdf)

    assert doc.stats["images_written"] == 1
    assert (out_dir / doc.sha256 / "images" / "fig1.png").read_bytes() == PNG


def test_vlm_request_carries_the_return_images_flag(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)

    sp, pp = _patched_parser(_fake_response())
    with sp, pp as post:
        VlmParser(VlmConfig(return_images=False)).extract(pdf)

    assert post.call_args.kwargs["data"]["return_images"] == "false"


# ---------------------------------------------------------------- errors


def test_vlm_extract_raises_on_http_error(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(status_code=500, error="boom")

    sp, pp = _patched_parser(response)
    with sp, pp, pytest.raises(RuntimeError, match="500"):
        VlmParser().extract(pdf)


def test_vlm_extract_raises_on_task_failure(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(status="failed", error="OOM")

    sp, pp = _patched_parser(response)
    with sp, pp, pytest.raises(RuntimeError, match="OOM"):
        VlmParser().extract(pdf)


def test_vlm_extract_raises_when_markdown_empty(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(markdown="")

    sp, pp = _patched_parser(response)
    with sp, pp, pytest.raises(RuntimeError, match="empty markdown"):
        VlmParser().extract(pdf)


# ---------------------------------------------------------------- stats


def test_vlm_stats_include_backend_and_url(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path)
    response = _fake_response(backend="vlm-mlx-engine")

    sp, pp = _patched_parser(response)
    with sp, pp:
        parser = VlmParser(VlmConfig(engine="mlx-engine"))
        doc = parser.extract(pdf)

    assert doc.stats["mineru_backend"] == "vlm-mlx-engine"
    assert doc.stats["mineru_api_url"] == "http://test"
