"""OCR quality scorer — out-of-process HTTP client to ``_quality_server``.

The actual ModernBERT / transformers stack lives in a subprocess (see
``pdfsys_bench._quality_server``). The bench / CLI client posts text to
it over HTTP and receives a float score. Lifecycle mirrors the parser
HTTP-client design.

Why a subprocess for a model the bench process used to load in-process?

* Cold-start ``transformers.from_pretrained`` triggers HF Hub revision
  probes which retry on flaky network — we observed 8000+ s wall clock
  for a 20-PDF run that should have taken minutes.
* Heavy import surface (torch + transformers + safetensors) ballooned
  the bench parent process and slowed cascade workflows that don't
  need quality scoring at all.

Subprocess + ``HF_HUB_OFFLINE=1`` ensures predictable behavior and a
lean bench parent.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

_LOG = logging.getLogger(__name__)

# Final scoring model: ModernBERT-base fine-tune (4 ordinal classes,
# 8192-token context). Legacy fallback: the FinePDFs regression model
# HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn (512 tokens,
# max_chars 10_000).
DEFAULT_MODEL = "miracleyin/mnbvc-pdf-quality-scorer-modernbert"
DEFAULT_MAX_CHARS = 40_000
DEFAULT_MAX_TOKENS = 8192

_READY_TIMEOUT_S = 180.0  # cold-start incl. model load
_READY_POLL_S = 1.0
_SCORE_TIMEOUT_S = 60.0


@dataclass(slots=True)
class QualityScore:
    """Result of scoring one document."""

    score: float
    num_chars: int
    num_tokens: int
    model: str

    def as_record(self) -> dict[str, Any]:
        return {
            "quality_score": self.score,
            "quality_num_chars": self.num_chars,
            "quality_num_tokens": self.num_tokens,
            "quality_model": self.model,
        }


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OcrQualityScorer:
    """Lazy HTTP client to a ModernBERT scoring subprocess."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.device = device
        self.dtype = dtype
        self._proc: subprocess.Popen | None = None
        self._base_url: str | None = None

    def _ensure_server(self) -> str:
        if self._base_url is not None:
            return self._base_url

        # External server: when QUALITY_URL is set, the client connects
        # to that URL directly and skips the subprocess lifecycle (used
        # by container deployments where a sibling pdfsys-quality
        # service hosts ModernBERT). `close()` becomes a no-op because
        # `self._proc` stays None.
        external = os.environ.get("QUALITY_URL")
        if external:
            base = external.rstrip("/")
            _LOG.info("using external quality scorer at %s", base)
            self._base_url = base
            return base

        port = _pick_free_port()
        cmd = [
            sys.executable, "-m", "pdfsys_bench._quality_server",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--model", self.model_name,
            "--dtype", self.dtype,
            "--max-tokens", str(self.max_tokens),
            "--max-chars", str(self.max_chars),
        ]
        if self.device:
            cmd += ["--device", self.device]

        # Inherit env + force HF offline so cold-start can't block on
        # flaky network. Models are expected to be cached locally.
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

        _LOG.info("starting quality scorer subprocess on 127.0.0.1:%d", port)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"quality-server exited early (rc={self._proc.returncode})"
                )
            try:
                r = httpx.get(f"{base}/health", timeout=2.0)
                if r.status_code == 200:
                    _LOG.info(
                        "quality-server ready at %s (pid=%s)",
                        base, self._proc.pid,
                    )
                    self._base_url = base
                    return base
            except httpx.HTTPError:
                pass
            time.sleep(_READY_POLL_S)

        self.close()
        raise TimeoutError(
            f"quality-server did not become ready at {base} in {_READY_TIMEOUT_S}s"
        )

    def score(self, text: str) -> QualityScore:
        """Score a single document. Empty input returns 0.0 without RPC."""
        if not text or not text.strip():
            return QualityScore(
                score=0.0,
                num_chars=0,
                num_tokens=0,
                model=self.model_name,
            )

        url = self._ensure_server()
        resp = httpx.post(
            f"{url}/score",
            content=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=_SCORE_TIMEOUT_S,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"quality-server /score returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        return QualityScore(
            score=float(data["score"]),
            num_chars=int(data["num_chars"]),
            num_tokens=int(data["num_tokens"]),
            model=str(data.get("model") or self.model_name),
        )

    def score_many(self, texts: list[str]) -> list[QualityScore]:
        """Serial scoring — tiny MVP harness, not a batched hot path."""
        return [self.score(t) for t in texts]

    def close(self) -> None:
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

    def __del__(self) -> None:  # pragma: no cover
        self.close()
