"""Fetch the FinePDFs XGBoost router weights from upstream.

The weights file (``xgb.ubj``, ~257 KB) is not committed to this repo —
it's external IP owned by HuggingFace/FinePDFs and lives on their Git-LFS
bucket. Running this module downloads it once into ``models/xgb_classifier.ubj``
next to this package.

Usage::

    python -m pdfsys_router.download_weights
"""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path

# GitHub raw download URL for XGBoost router weights
WEIGHTS_URLS = [
    "https://github.com/huggingface/finepdfs/raw/main/models/xgb_ocr_classifier/xgb_classifier.ubj",
    "https://raw.githubusercontent.com/huggingface/finepdfs/main/models/xgb_ocr_classifier/xgb_classifier.ubj",
]


def target_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "models" / "xgb_classifier.ubj"


def download(force: bool = False, timeout: int = 30) -> Path:
    dst = target_path()
    if dst.exists() and not force:
        print(f"[download_weights] already present: {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    last_error = None
    for url in WEIGHTS_URLS:
        print(f"[download_weights] fetching {url}")
        try:
            # 设置超时
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — pinned URL
                data = r.read()
            if len(data) < 10_000:
                raise RuntimeError(
                    f"downloaded blob is suspiciously small ({len(data)} bytes) — "
                    "likely an LFS pointer, not the binary"
                )
            dst.write_bytes(data)
            print(f"[download_weights] wrote {len(data)} bytes -> {dst}")
            return dst
        except (urllib.error.URLError, socket.timeout) as e:
            last_error = e
            print(f"[download_weights] failed for {url}: {e}")
            continue
    
    raise RuntimeError(f"Failed to download weights from all URLs: {last_error}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
