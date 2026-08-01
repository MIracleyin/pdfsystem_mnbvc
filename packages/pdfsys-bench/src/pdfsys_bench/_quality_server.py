"""Quality scorer subprocess — runs ModernBERT in an isolated process.

Started by :class:`pdfsys_bench.quality.OcrQualityScorer`. Loads
``HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn`` (or any
HuggingFace regression head) once, then serves
``POST /score {text}`` over HTTP.

Why a subprocess?
* Torch + transformers initialization in the bench process slows every
  startup and pulls in heavy deps. Isolating into a subprocess keeps
  the bench client small.
* HF Hub probes (which retry on flaky network) cannot block the bench
  parent — the subprocess inherits ``HF_HUB_OFFLINE=1`` from the
  parent's ``OcrQualityScorer._ensure_server`` so we always use cached
  weights.

Run manually:
    python -m pdfsys_bench._quality_server --port 8765 --model <hf-id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_MODEL = "HuggingFaceFW/finepdfs_ocr_quality_classifier_eng_Latn"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_CHARS = 10_000

_LOG = logging.getLogger("pdfsys_bench._quality_server")

# Globals set during _init(); accessed by the handler.
_MODEL: Any = None
_TOKENIZER: Any = None
_DEVICE: Any = None
_TORCH: Any = None
_MODEL_NAME: str = ""
_MAX_TOKENS: int = DEFAULT_MAX_TOKENS
_MAX_CHARS: int = DEFAULT_MAX_CHARS


def _init(model_name: str, device_pref: str | None, dtype_name: str) -> None:
    global _MODEL, _TOKENIZER, _DEVICE, _TORCH, _MODEL_NAME
    _MODEL_NAME = model_name

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TORCH = torch
    if device_pref:
        _DEVICE = torch.device(device_pref)
    elif torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")

    dtype = getattr(torch, dtype_name, torch.float32)

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=dtype
        )
    except TypeError:
        # transformers < 5
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype
        )
    model.eval()
    model.to(_DEVICE)
    _MODEL = model


def _logits_to_score(logits: Any) -> float:
    """Map a classification-head output to a scalar quality score in [0, 3].

    Regression head (1 logit): the raw value, clamped. Ordinal multi-class
    head (N logits for classes 0..N-1): softmax expectation over class
    indices — continuous, so threshold-based consumers keep working.
    Tensor-method-only on purpose: module top imports stay stdlib.
    """
    flat = logits.reshape(-1).float()
    if flat.numel() == 1:
        return max(0.0, min(3.0, float(flat.item())))
    probs = flat.softmax(-1)
    expectation = sum(i * float(p) for i, p in enumerate(probs))
    return max(0.0, min(3.0, expectation))


def _score(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {
            "score": 0.0,
            "num_chars": 0,
            "num_tokens": 0,
            "model": _MODEL_NAME,
        }

    clipped = text[:_MAX_CHARS]
    enc = _TOKENIZER(
        clipped,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_TOKENS,
    )
    num_tokens = int(enc["input_ids"].shape[1])
    enc = {k: v.to(_DEVICE) for k, v in enc.items()}

    with _TORCH.inference_mode():
        out = _MODEL(**enc)
        score = _logits_to_score(out.logits)
    return {
        "score": score,
        "num_chars": len(clipped),
        "num_tokens": num_tokens,
        "model": _MODEL_NAME,
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return  # silent — we have our own logging

    def _send_json(self, status: HTTPStatus, body: Any) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, status: HTTPStatus, msg: str) -> None:
        payload = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "model": _MODEL_NAME})
            return
        self._send_text(HTTPStatus.NOT_FOUND, "no such endpoint")

    def do_POST(self) -> None:
        if self.path != "/score":
            self._send_text(HTTPStatus.NOT_FOUND, "no such endpoint")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_text(HTTPStatus.BAD_REQUEST, "body required")
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_text(HTTPStatus.BAD_REQUEST, f"invalid JSON: {e}")
            return

        text = body.get("text")
        if not isinstance(text, str):
            self._send_text(HTTPStatus.BAD_REQUEST, "missing 'text' string field")
            return

        try:
            result = _score(text)
        except Exception as e:
            self._send_text(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{type(e).__name__}: {e}",
            )
            return

        self._send_json(HTTPStatus.OK, result)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pdfsys_bench._quality_server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None,
                   help="Force device (cuda|mps|cpu). Default: auto.")
    p.add_argument("--dtype", default="bfloat16",
                   help="Torch dtype name (e.g. bfloat16, float32).")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = p.parse_args(argv)

    global _MAX_TOKENS, _MAX_CHARS
    _MAX_TOKENS = args.max_tokens
    _MAX_CHARS = args.max_chars

    # Force offline mode by default — caller (OcrQualityScorer) already
    # sets these in subprocess env, but setting here too is harmless and
    # supports manual launches.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"[quality-server] loading {args.model} ...", flush=True)
    _init(args.model, args.device, args.dtype)
    print(
        f"[quality-server] ready on http://{args.host}:{args.port}/ "
        f"(device={_DEVICE}, dtype={args.dtype})",
        flush=True,
    )

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
