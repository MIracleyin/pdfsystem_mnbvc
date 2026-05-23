"""Mineru VLM parser — talks to a ``mineru-api`` subprocess over HTTP.

**Out-of-process design.** Each :class:`VlmParser` instance owns a
``mineru-api`` subprocess that loads the mineru VLM stack (torch +
MLX/transformers) in isolation. The bench / CLI client never imports
mineru, so the torch+MLX+Metal conflicts that plague in-process VLM
inference on macOS Apple Silicon (Predict-stage hangs at 0% CPU) cannot
happen.

Lifecycle:

* ``__init__``: stash config, do nothing else.
* First ``extract()``: start ``mineru-api`` subprocess, poll ``/health``
  until ready, then POST the PDF.
* Subsequent ``extract()``: reuse the running subprocess.
* ``close()`` (or GC): terminate the subprocess.

See ``docs/superpowers/specs/2026-05-22-mineru-parsers-migration-design.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from pdfsys_core import Backend, ExtractedDoc, VlmConfig

_LOG = logging.getLogger(__name__)

_READY_TIMEOUT_S = 120.0
_READY_POLL_S = 1.0
_EXTRACT_TIMEOUT_S = 600.0


def _resolve_mineru_api_bin() -> str:
    """Locate the ``mineru-api`` entry-point script."""
    on_path = shutil.which("mineru-api")
    if on_path:
        return on_path
    sibling = Path(sys.executable).parent / "mineru-api"
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError(
        "mineru-api not found on PATH or next to sys.executable. "
        "Install via `mineru[vlm]` (or `mineru[pipeline]`)."
    )


def _pick_free_port() -> int:
    """Reserve a free localhost port. Race-prone but fine for single-user dev."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VlmParser:
    """Mineru VLM-mode parser (out-of-process HTTP client)."""

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig()
        self._proc: subprocess.Popen | None = None
        self._base_url: str | None = None

    def _ensure_server(self) -> str:
        """Lazy-start the mineru-api subprocess; return its base URL."""
        if self._base_url is not None:
            return self._base_url

        bin_path = _resolve_mineru_api_bin()
        port = _pick_free_port()
        _LOG.info("starting mineru-api at 127.0.0.1:%d", port)
        # Force offline model resolution — mineru otherwise hits HF Hub for
        # revision checks at every cold start, which fails when HF is down
        # or behind a flaky proxy. Our weights live in ~/.cache/huggingface
        # already so this is the safe default.
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        self._proc = subprocess.Popen(
            [bin_path, "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"mineru-api exited early (rc={self._proc.returncode})"
                )
            try:
                r = httpx.get(f"{base}/health", timeout=2.0)
                if r.status_code == 200:
                    _LOG.info(
                        "mineru-api ready at %s (pid=%s)", base, self._proc.pid
                    )
                    self._base_url = base
                    return base
            except httpx.HTTPError:
                pass
            time.sleep(_READY_POLL_S)

        # Clean up on timeout.
        self.close()
        raise TimeoutError(
            f"mineru-api did not become ready at {base} in {_READY_TIMEOUT_S}s"
        )

    def extract(self, pdf_path: Path) -> ExtractedDoc:
        """Extract markdown from ``pdf_path`` via mineru VLM mode over HTTP."""
        pdf_path = Path(pdf_path)
        pdf_bytes = pdf_path.read_bytes()
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        backend = f"vlm-{self.config.engine}"

        url = self._ensure_server()

        with pdf_path.open("rb") as f:
            resp = httpx.post(
                f"{url}/file_parse",
                files={"files": (f"{sha}.pdf", f, "application/pdf")},
                data={
                    "backend": backend,
                    "lang_list": self.config.p_lang,
                    "formula_enable": str(self.config.formula_enable).lower(),
                    "table_enable": str(self.config.table_enable).lower(),
                    "return_md": "true",
                    "return_middle_json": "true",
                    "return_content_list": "true",
                    "return_layout_pdf": "false",
                    "return_images": "false",
                },
                timeout=_EXTRACT_TIMEOUT_S,
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"mineru-api /file_parse returned {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        if payload.get("status") != "completed":
            raise RuntimeError(
                f"mineru-api task failed: {payload.get('error') or payload}"
            )

        results = payload.get("results") or {}
        if not results:
            raise RuntimeError("mineru-api returned no results")
        file_key = next(iter(results))
        result = results[file_key]
        markdown = result.get("md_content") or ""
        if not markdown:
            raise RuntimeError(
                f"mineru-api returned empty markdown for {file_key}"
            )

        stats: dict[str, Any] = {
            "mineru_backend": backend,
            "mineru_api_url": url,
            "mineru_version": payload.get("version"),
        }
        stats["middle_json_path"] = _persist_sidecar(
            result.get("middle_json"),
            self.config.output_dir,
            sha,
            f"{sha}_middle.json",
        )
        stats["content_list_path"] = _persist_sidecar(
            result.get("content_list"),
            self.config.output_dir,
            sha,
            f"{sha}_content_list.json",
        )

        return ExtractedDoc(
            sha256=sha,
            backend=Backend.VLM,
            segments=(),
            markdown=markdown,
            stats=stats,
        )

    def close(self) -> None:
        """Terminate the mineru-api subprocess, if any."""
        proc = self._proc
        self._proc = None
        self._base_url = None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover  — GC-time cleanup
        self.close()


def _persist_sidecar(
    payload: Any,
    output_dir: Path | None,
    sha: str,
    filename: str,
) -> str | None:
    """Write a sidecar JSON to ``<output_dir>/<sha>/<filename>`` if provided.

    Returns the relative path (under ``output_dir``) or ``None`` when
    output_dir is unset or the payload is missing.
    """
    if output_dir is None or payload is None:
        return None
    out_dir = Path(output_dir) / sha
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    out_path.write_text(text, encoding="utf-8")
    return str(out_path.relative_to(output_dir))
